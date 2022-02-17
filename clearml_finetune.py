
from clearml import Task, Dataset
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
from dataset_utils import generate_vocab_json, prepare_dataset

import os
from transformers.integrations import TensorBoardCallback
from datasets import load_dataset, load_metric, Audio
from config.config import DatasetConfig as ds_cfg, \
     ModelConfig as m_cfg, TrainingConfig as t_cfg
from transformers import Trainer, TrainingArguments, \
     Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, \
     Wav2Vec2Processor, Wav2Vec2ForCTC

def main():
    print('Downloading Dataset from S3 via ClearML...')
    dataset = Dataset.get(dataset_project=ds_cfg.dataset_project, dataset_name=ds_cfg.dataset_name)
    dataset_path = dataset.get_local_copy()

    print('Loading Datasets...')
    train_dataset = load_dataset(dataset_path, split = 'train+validation')
    train_dataset = train_dataset.map(lambda x: {'audio_filepath': os.path.join(dataset_path, x['audio_filepath'])})
    train_dataset = train_dataset.cast_column('audio_filepath', Audio(sampling_rate=ds_cfg.sample_rate))

    test_dataset = load_dataset(dataset_path, split = 'test')
    test_dataset = test_dataset.map(lambda x: {'audio_filepath': os.path.join(dataset_path, x['audio_filepath'])})
    test_dataset = test_dataset.cast_column('audio_filepath', Audio(sampling_rate=ds_cfg.sample_rate))

    print('Datasets cleaned.')
    vocab_path = generate_vocab_json([train_dataset, test_dataset], column_name='audio_filepath')

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
        ctc_zero_infinity = m_cfg.ctc_zero_infinity,
        pad_token_id = processor.tokenizer.pad_token_id,
        vocab_size = len(processor.tokenizer),
    )

    model.freeze_feature_extractor()
    print('Model initiated. Preparing Trainer...')

    training_args = TrainingArguments(
        output_dir = t_cfg.output_dir,
        group_by_length = t_cfg.group_by_length,
        per_device_train_batch_size = t_cfg.per_device_train_batch_size,
        per_device_eval_batch_size = t_cfg.per_device_eval_batch_size,
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
        load_best_model_at_end = t_cfg.load_best_model_at_end,
        metric_for_best_model = t_cfg.metric_for_best_model,
        greater_is_better = t_cfg.greater_is_better,
        save_total_limit = t_cfg.save_total_limit,
        dataloader_num_workers = t_cfg.dataloader_num_workers,
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
        tb_writer = TensorBoardCallback()
    )

    trainer.train()

if __name__ == '__main__':
    main()