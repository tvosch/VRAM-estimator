from dataclasses import dataclass
import argparse

from transformers import AutoConfig


PRECISION_TO_BYTES = {"float32": 4,
                      "fp32": 4,
                      "float16": 2,
                      "fp16": 2,
                      "bfloat16": 2,
                      "bf16": 2,
                      "int8": 1}

@dataclass
class ModelConfig:
    model_size: float
    hidden_size: int
    sequence_length: int
    num_layers: int
    num_heads: int

    def overwrite_with_hf_config(self, auto_config: AutoConfig):
        self.model_size = round(get_model_size_from_autoconfig(auto_config) / 10**9, 2)
        self.hidden_size = auto_config.hidden_size
        self.sequence_length = auto_config.max_position_embeddings
        self.num_layers = auto_config.num_hidden_layers
        self.num_heads = auto_config.num_attention_heads

@dataclass
class TrainingConfig:
    micro_batch_size: int
    num_gpus: int
    optimizer: str
    zero_stage: int
    gradient_checkpointing: False
    mixed_precision: False


def parse_args():
    parser = argparse.ArgumentParser(description="Parser for VRAM estimator")

    parser.add_argument("--repo_id", type=str, default=None, help="HuggingFace repo id to automatically determine model settings")
    parser.add_argument("--model_size", type=float, default=7, help="Model size (in billion parameters)")
    parser.add_argument("--hidden_size", type=int, default=4096, help="Hidden size")
    parser.add_argument("--sequence_length", type=int, default=8192, help="Sequence length")
    parser.add_argument("--num_layers", type=int, default=32, help="Number of layers")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Micro batch size (batch size per device/GPU)")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of heads")
    parser.add_argument("--zero_stage", type=int, default=0, choices=[0, 1, 2, 3], help="ZeRO optimization stage")
    parser.add_argument("--gradient_checkpointing", type=bool, default=False, help="Enable gradient checkpointing")
    parser.add_argument("--mixed_precision", type=bool, default=False, help="Enable mixed precision for model training")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Type of optimizer")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs. Necessary for estimating ZeRO stages")
    parser.add_argument("--cache_dir", type=str, default=".huggingface_configs", help="HuggingFace cache directory to download config from")

    return parser.parse_args()

def get_model_size_from_autoconfig(config: AutoConfig):
    # Embedding parameters:
    embedding_params = config.vocab_size * config.hidden_size

    # Transformer layer parameters
    def transformer_layer_params(hidden_size, intermediate_size, num_key_value_heads):
            input_layernorm_params = hidden_size
            mlp_down_proj_params = hidden_size * intermediate_size
            mlp_gate_proj_params = intermediate_size * hidden_size
            mlp_up_proj_params = intermediate_size * hidden_size
            post_attention_layernorm_params = hidden_size
            self_attn_k_proj_params = (hidden_size // (num_key_value_heads // 2)) * hidden_size
            self_attn_o_proj_params = hidden_size * hidden_size
            self_attn_q_proj_params = hidden_size * hidden_size
            self_attn_v_proj_params = (hidden_size // (num_key_value_heads // 2)) * hidden_size
            
            total_layer_params = (
                input_layernorm_params + mlp_down_proj_params + mlp_gate_proj_params + mlp_up_proj_params +
                post_attention_layernorm_params + self_attn_k_proj_params + self_attn_o_proj_params +
                self_attn_q_proj_params + self_attn_v_proj_params
            )

            return total_layer_params
    
    # Total parameters for all transformer layers
    single_layer_params = transformer_layer_params(config.hidden_size, config.intermediate_size, config.num_key_value_heads)
    total_transformer_params = config.num_hidden_layers * single_layer_params
    
    # Output layer parameters
    output_params = config.vocab_size * config.hidden_size
    
    # Total parameters
    total_params = embedding_params + total_transformer_params + output_params
    return total_params


def download_config_from_hub(repo_id: str, cache_dir: str):
    return AutoConfig.from_pretrained(pretrained_model_name_or_path=repo_id, cache_dir=cache_dir)

def model_memory(parameters, precision = "bf16", mixed_precision = False):
    if mixed_precision:
        return parameters * (PRECISION_TO_BYTES["fp32"] + PRECISION_TO_BYTES["fp16"])
    return parameters * PRECISION_TO_BYTES[precision]
    

def gradients_memory(parameters, precision = "fp32"):
    return parameters * PRECISION_TO_BYTES[precision]

def optimizer_memory(parameters, optimizer= "adamw", precision = "fp32"):
    optimizer_choices = {"adam": 3,
                         "adamw": 2,
                         "sgd": 1}
    return optimizer_choices[optimizer] * parameters * PRECISION_TO_BYTES[precision]

def activations_memory(num_layers, sequence_length, micro_batch_size, hidden_size, num_heads):
    # Reference: https://arxiv.org/pdf/2205.05198
    # Activations assumed to be in 16-bit floating precision
    bytes_per_layer = sequence_length * micro_batch_size * hidden_size * (34 + 5 * (num_heads * sequence_length / hidden_size))
    bytes_model = bytes_per_layer * num_layers
    return round(bytes_model / 10**9, 2)

def vram_required(model_size, hidden_size, sequence_length, num_layers, num_heads, micro_batch_size, num_gpus, optimizer, zero_stage, gradient_checkpointing, mixed_precision):
    # Reference: https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/
    model_vram = model_memory(model_size, mixed_precision=mixed_precision)
    gradients_vram = gradients_memory(model_size)
    optimizer_vram = optimizer_memory(model_size, optimizer=optimizer)

    # Baseline
    if zero_stage == 0:
        aggregated_vram = model_vram + gradients_vram + optimizer_vram
    # Optimizer state partitioning
    if zero_stage == 1:
        aggregated_vram = model_vram + gradients_vram + (optimizer_vram / num_gpus)
    # Gradient + Optimzer state partitioning
    if zero_stage == 2:
        aggregated_vram = model_vram + ((gradients_vram + optimizer_vram) / num_gpus)
    # Parameter partitioning + Gradient + Optimizer partitioning
    if zero_stage == 3:
        aggregated_vram = (model_vram / num_gpus) + (gradients_vram / num_gpus) + (optimizer_vram / num_gpus)

    print(f"ZeRO stage {zero_stage} takes {aggregated_vram} GB")

    activations_vram = activations_memory(num_layers, sequence_length, micro_batch_size, hidden_size, num_heads)
    if gradient_checkpointing:
        activations_vram = activations_vram ** 0.5
    
    print(f"Activations require {activations_vram} GB with gradient checkpointing: {gradient_checkpointing}")
    total_vram = aggregated_vram + activations_vram
    print(f"Estimated 'minimal' VRAM requirement on {num_gpus} GPUs per GPU is {total_vram} GB")
    return total_vram


if __name__ == "__main__":
    args = parse_args()
    print(args)
    model_config = ModelConfig(args.model_size, args.hidden_size, args.sequence_length, args.num_layers, args.num_heads)
    training_config = TrainingConfig(args.micro_batch_size, args.num_gpus, args.optimizer, args.zero_stage, args.gradient_checkpointing, args.mixed_precision)
    if args.repo_id:
        config = download_config_from_hub(args.repo_id, args.cache_dir)
        model_config.overwrite_with_hf_config(config)

    total_vram = vram_required(**vars(model_config), **vars(training_config))

    # 1.1B model actually takes 31982 Mib so 33,5 GB
