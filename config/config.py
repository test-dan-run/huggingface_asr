class Config:
    hello=1

    @staticmethod
    def get_dict():
        return {key:value for key, value in Config.__dict__.items() if not key.startswith('__') and not callable(key)}

class ClearMLConfig(Config):
    project_name = 'e2e-selfsupervised-asr/xlsr-finetune'
    task_name = 'commonvoice_id_v8.0_dropout'
    output_uri = 's3://experiment-logging/storage'
    task_type = 'training'
    tags = ['Wav2Vec2', 'Not Fairseq']

    base_docker = 'dleongsh/huggingface_asr:v4.16.2'
    queue_name = 'compute'

class DatasetConfig(Config):
    dataset_name = 'mozilla-foundation/common_voice_8_0'
    language = 'id'

    columns_to_remove = ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"]
    chars_to_remove = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\！\–\—\，\\\]'
    chars_to_replace = {
        '[’]': "'",
        '[á]': 'a',
        '[é]': 'e',
        '[ł]': 'z',
        '[ń]': 'n',
        '[ō]': 'o',
        '[&]': 'and',
    }

    feature_size = 1
    sample_rate = 16000
    padding_value = 0.0
    do_normalize = True
    return_attention_mask = True

class ModelConfig(Config):
    pretrained_model_name = "facebook/wav2vec2-xls-r-300m"
    attention_dropout = 0.1
    hidden_dropout = 0.1
    feat_proj_dropout = 0.0
    mask_time_prob = 0.05
    layerdrop = 0.1
    ctc_loss_reduction = "mean"

class TrainingConfig(Config):
    output_dir = 'output'
    group_by_length=True
    per_device_train_batch_size=32
    per_device_eval_batch_size=32
    gradient_accumulation_steps=2
    evaluation_strategy="steps"
    num_train_epochs=100
    gradient_checkpointing=True
    fp16=True
    save_steps=500
    eval_steps=500
    logging_steps=500
    learning_rate=3e-4
    warmup_steps=500
    load_best_model_at_end = True
    metric_for_best_model = 'wer'
    dataloader_num_workers = 8
    save_total_limit=1
    push_to_hub=False
