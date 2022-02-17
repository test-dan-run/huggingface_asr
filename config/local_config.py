class DatasetConfig:

    train_datasets = {
        'cv_8.0': {
            'path': '/datasets/asr/processed/id/cv_8.0',
            'split': 'train+dev',
        },
        'magichub_scripted': {
            'path': '/datasets/asr/processed/id/magichub_scripted',
            'split': 'train',
        },
        'magichub_conversational': {
            'path': '/datasets/asr/processed/id/magichub_conversational',
            'split': 'train',
        }
    }

    test_datasets = {
        'cv_8.0': {
            'path': '/datasets/asr/processed/id/cv_8.0',
            'split': 'test',
        }
    }

    feature_size = 1
    sample_rate = 16000
    padding_value = 0.0
    do_normalize = True
    return_attention_mask = True

class ModelConfig:
    pretrained_model_name = "facebook/wav2vec2-xls-r-300m"
    attention_dropout = 0.1
    hidden_dropout = 0.1
    feat_proj_dropout = 0.0
    mask_time_prob = 0.05
    layerdrop = 0.1
    ctc_loss_reduction = "mean"

class TrainingConfig:
    output_dir = 'output'
    group_by_length=True
    per_device_train_batch_size=8
    per_device_eval_batch_size=8
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
    dataloader_num_workers = 4
    save_total_limit=1
    push_to_hub=False
