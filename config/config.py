from dataclasses import dataclass
from omegaconf import MISSING
from typing import Tuple

@dataclass
class ClearMLConfig:
    project_name: str = 'e2e-selfsupervised-asr/xlsr-finetune'
    task_name: str = 'cv_magichub'
    output_uri: str = 's3://experiment-logging/storage'
    task_type:str  = 'training'
    tags: Tuple[str] = ('Wav2Vec2', 'Not Fairseq')

    base_docker: str = 'dleongsh/huggingface_asr:v4.16.2'
    queue_name: str = 'compute'

@dataclass
class DatasetConfig:
    dataset_project: str = 'indonesian_corpus'
    dataset_name: str = 'cv_magichub'

    feature_size : int = 1
    sample_rate : int = 16000
    padding_value : float = 0.0
    do_normalize: bool = True
    return_attention_mask: bool = True

@dataclass
class ModelConfig:
    pretrained_model_name: str = 'facebook/wav2vec2-xls-r-300m'
    attention_dropout : float = 0.1
    hidden_dropout : float = 0.1
    feat_proj_dropout : float = 0.2
    mask_time_prob : float = 0.05
    layerdrop : float = 0.1
    ctc_loss_reduction: str = 'mean'
    ctc_zero_infinity: bool = True
    pad_token_id: int = MISSING
    vocab_size: int = MISSING
    
@dataclass
class TrainingArgumentsConfig:
    output_dir: str = 'output'
    group_by_length = True
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    evaluation_strategy: str = 'steps'
    num_train_epochs: int = 100
    gradient_checkpointing: bool = False
    fp16: bool = True
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    learning_rate: float = 5e-4
    lr_scheduelr_type: str = 'linear'
    warmup_steps: int = 1000
    weight_decay: float = 0.005
    load_best_model_at_end: bool = True
    metric_for_best_model: str = 'wer'
    greater_is_better: bool = False
    dataloader_num_workers : int = 8
    save_total_limit: int = 1
    push_to_hub: bool = False
