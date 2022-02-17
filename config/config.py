class ClearMLConfig:
    project_name = 'e2e-selfsupervised-asr/xlsr-finetune'
    task_name = 'cv_magichub'
    output_uri = 's3://experiment-logging/storage'
    task_type = 'training'
    tags = ['Wav2Vec2', 'Not Fairseq']

    base_docker = 'dleongsh/huggingface_asr:v4.16.2'
    queue_name = 'compute'

class DatasetConfig:

    dataset_project = 'indonesian_corpus'
    dataset_name = 'cv_magichub'

    feature_size = 1
    sample_rate = 16000
    padding_value = 0.0
    do_normalize = True
    return_attention_mask = True

class ModelConfig:
    pretrained_model_name = "facebook/wav2vec2-xls-r-300m"
    attention_dropout = 0.1
    hidden_dropout = 0.1
    feat_proj_dropout = 0.1
    mask_time_prob = 0.05
    layerdrop = 0.1
    ctc_loss_reduction = "mean"
    ctc_zero_infinity = True

class TrainingConfig:
    output_dir = 'output'
    group_by_length=True
    per_device_train_batch_size=16
    per_device_eval_batch_size=16
    gradient_accumulation_steps=2
    evaluation_strategy="steps"
    num_train_epochs=100
    gradient_checkpointing=True
    fp16=True
    save_steps=500
    eval_steps=500
    logging_steps=100
    learning_rate=3e-4
    warmup_steps=1000
    load_best_model_at_end = True
    metric_for_best_model = 'wer'
    greater_is_better = False
    dataloader_num_workers = 8
    save_total_limit=1
    push_to_hub=False
