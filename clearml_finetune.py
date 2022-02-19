
from clearml import Task, Dataset
from config.config import ClearMLConfig as c_cfg

Task.add_requirements('librosa', 'omegaconf')
task = Task.init(
    project_name=c_cfg.project_name, 
    task_name=c_cfg.task_name, 
    output_uri=c_cfg.output_uri, 
    task_type=c_cfg.task_type)
task.set_base_docker(c_cfg.base_docker)
task.execute_remotely(queue_name=c_cfg.queue_name, clone=False, exit_process=True)

# self-defined packages
from metrics import compute_metrics
from dataloader import DataCollatorCTCWithPadding
from dataset_utils import generate_vocab_json, prepare_dataset

import os
from omegaconf import OmegaConf
from transformers.integrations import TensorBoardCallback
from datasets import load_dataset, load_metric, Audio
from config.config import DatasetConfig, ModelConfig, TrainingArgumentsConfig
from transformers import Trainer, TrainingArguments, \
     Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, \
     Wav2Vec2Processor, Wav2Vec2ForCTC

def main():

    # import confs
    ds_cfg = OmegaConf.structured(DatasetConfig)
    m_cfg = OmegaConf.structured(ModelConfig)
    t_cfg = OmegaConf.structured(TrainingArgumentsConfig)

    print('Downloading Dataset from S3 via ClearML...')
    dataset = Dataset.get(dataset_project=ds_cfg.dataset_project, dataset_name=ds_cfg.dataset_name)
    dataset_path = dataset.get_local_copy()

    print('Loading Datasets...')
    train_dataset = load_dataset('csv', data_files=[os.path.join(dataset_path, 'train.csv'), os.path.join(dataset_path, 'validation.csv')], split='train')
    train_dataset = train_dataset.map(lambda x: {'audio_filepath': os.path.join(dataset_path, x['audio_filepath'])})
    train_dataset = train_dataset.cast_column('audio_filepath', Audio(sampling_rate=ds_cfg.sample_rate))

    test_dataset = load_dataset('csv', data_files=os.path.join(dataset_path, 'test.csv'), split='train')
    test_dataset = test_dataset.map(lambda x: {'audio_filepath': os.path.join(dataset_path, x['audio_filepath'])})
    test_dataset = test_dataset.cast_column('audio_filepath', Audio(sampling_rate=ds_cfg.sample_rate))

    print('Datasets cleaned.')
    vocab_path = generate_vocab_json([train_dataset, test_dataset], column_name='text')

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
    m_cfg['pad_token_id'] = processor.tokenizer.pad_token_id
    m_cfg['vocab_size'] = len(processor.tokenizer)

    model = Wav2Vec2ForCTC.from_pretrained(**m_cfg)

    model.freeze_feature_extractor()
    print('Model initiated. Preparing Trainer...')

    training_args = TrainingArguments(**t_cfg)

    trainer = Trainer(
        model = model,
        data_collator = data_collator,
        args = training_args,
        compute_metrics = lambda pred: compute_metrics(pred, processor, wer_metric),
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        tokenizer = processor.feature_extractor,
        callbacks = [TensorBoardCallback(),]
    )

    trainer.train()

if __name__ == '__main__':
    main()