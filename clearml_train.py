
from clearml import Task
from config.hf_env import HF_AUTH_TOKEN
from config.config import ClearMLConfig as c_cfg

task = Task.init(
    project_name=c_cfg.project_name, 
    task_name=c_cfg.task_name, 
    output_uri=c_cfg.output_uri, 
    task_type=c_cfg.task_type)
task.set_base_docker(f'{c_cfg.base_docker} --env HF_AUTH_TOKEN={HF_AUTH_TOKEN}')
task.execute_remotely(queue_name=c_cfg.queue_name, clone=False, exit_process=True)

# self-defined packages
from metrics import compute_metrics
from dataloader import DataCollatorCTCWithPadding
from dataset_utils import clean_dataset, generate_vocab_json, prepare_dataset

import os
from datasets import load_dataset, load_metric, Audio
from config.config import DatasetConfig as ds_cfg, \
     ModelConfig as m_cfg, TrainingConfig as t_cfg
from transformers import Trainer, TrainingArguments, \
     Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, \
     Wav2Vec2Processor, Wav2Vec2ForCTC

def main():
    print('Loading Datasets...')
    train_dataset = load_dataset(
        ds_cfg.dataset_name,
        ds_cfg.language,
        split = 'train+validation',
        use_auth_token=os.environ.get('HF_AUTH_TOKEN')
    )

    test_dataset = load_dataset(
        ds_cfg.dataset_name,
        ds_cfg.language,
        split = 'test',
        use_auth_token=os.environ.get('HF_AUTH_TOKEN')
    )

    print('Datasets loaded. Cleaning Datasets...')
    train_dataset = clean_dataset(
        train_dataset, 
        ds_cfg.columns_to_remove,
        ds_cfg.chars_to_remove, 
        ds_cfg.chars_to_replace,
    )

    test_dataset = clean_dataset(
        test_dataset, 
        ds_cfg.columns_to_remove,
        ds_cfg.chars_to_remove, 
        ds_cfg.chars_to_replace,
    )

    train_dataset = train_dataset.cast_column('audio', Audio(sampling_rate=ds_cfg.sample_rate))
    test_dataset = test_dataset.cast_column('audio', Audio(sampling_rate=ds_cfg.sample_rate))

    print('Datasets cleaned.')
    vocab_path = generate_vocab_json([train_dataset, test_dataset])

    print('Preparing Tokenizer...')
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path, 
        unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )

    print('Tokenizer initiated. Preparing Feature Extractor...')
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size = ds_cfg.feature_size, 
        sampling_rate = ds_cfg.sample_rate, 
        padding_value = ds_cfg.padding_value, 
        do_normalize = ds_cfg.do_normalize, 
        return_attention_mask = ds_cfg.return_attention_mask
    )

    print('Feature Extractor initiated. Preparing Processor...')
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, 
        tokenizer=tokenizer
    )

    print('Processor initiated. Processing datasets...')
    train_dataset = train_dataset.map(
        lambda x: prepare_dataset(x, processor), 
        remove_columns= train_dataset.column_names
    )

    test_dataset = test_dataset.map(
        lambda x: prepare_dataset(x, processor), 
        remove_columns= test_dataset.column_names
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    wer_metric = load_metric('wer')

    print('Datasets prepared. Initiating model...')
    model = Wav2Vec2ForCTC.from_pretrained(
        m_cfg.pretrained_model_name, 
        attention_dropout = m_cfg.attention_dropout,
        hidden_dropout = m_cfg.hidden_dropout,
        feat_proj_dropout = m_cfg.feat_proj_dropout,
        mask_time_prob = m_cfg.mask_time_prob,
        layerdrop = m_cfg.layerdrop,
        ctc_loss_reduction = m_cfg.ctc_loss_reduction, 
        pad_token_id = processor.tokenizer.pad_token_id,
        vocab_size = len(processor.tokenizer),
    )

    model.freeze_feature_extractor()
    print('Model initiated. Preparing Trainer...')

    training_args = TrainingArguments(
        output_dir = t_cfg.output_dir,
        group_by_length = t_cfg.group_by_length,
        per_device_train_batch_size = t_cfg.per_device_train_batch_size,
        gradient_accumulation_steps = t_cfg.gradient_accumulation_steps,
        evaluation_strategy = t_cfg.evaluation_strategy,
        num_train_epochs = t_cfg.num_train_epochs,
        gradient_checkpointing = t_cfg.gradient_checkpointing,
        fp16 = t_cfg.fp16,
        save_steps = t_cfg.save_steps,
        eval_steps = t_cfg.eval_steps,
        logging_steps = t_cfg.logging_steps,
        learning_rate = t_cfg.learning_rate,
        warmup_steps = t_cfg.warmup_steps,
        save_total_limit = t_cfg.save_total_limit,
        push_to_hub = t_cfg.push_to_hub,
    )

    trainer = Trainer(
        model = model,
        data_collator = data_collator,
        args = training_args,
        compute_metrics = lambda pred: compute_metrics(pred, processor, wer_metric),
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        tokenizer = processor.feature_extractor,
    )

    trainer.train()

if __name__ == '__main__':
    main()