from pydantic import BaseModel, Field
from src.config.settings import settings


class LoRAConfig(BaseModel):
    base_model: str = Field(default_factory=lambda: settings.base_model_name)
    r: int = Field(default_factory=lambda: settings.lora_r)
    lora_alpha: int = Field(default_factory=lambda: settings.lora_alpha)
    lora_dropout: float = Field(default_factory=lambda: settings.lora_dropout)
    target_modules: list[str] = Field(default_factory=lambda: settings.lora_target_modules)
    num_train_epochs: int = Field(default_factory=lambda: settings.lora_training_epochs)
    learning_rate: float = Field(default_factory=lambda: settings.lora_learning_rate)
    per_device_train_batch_size: int = Field(default_factory=lambda: settings.lora_batch_size)
    gradient_accumulation_steps: int = Field(default_factory=lambda: settings.lora_gradient_accumulation)
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    use_gradient_checkpointing: bool = True
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    bf16: bool = False
    optim: str = "adamw_8bit"
    logging_steps: int = 10
    save_steps: int = 50
    output_dir: str = "/tmp/lora_output"
