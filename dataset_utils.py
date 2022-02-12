import re
import json

def remove_special_characters(batch, chars_to_remove_regex):
    batch['sentence'] = re.sub(chars_to_remove_regex, '', batch['sentence']).lower()
    return batch

def replace_hatted_characters(batch, chars_to_replace):
    for char, sub in chars_to_replace.items():
        batch['sentence'] = re.sub(char, sub, batch['sentence'])
    return batch

def remove_rows_with_brackets(x):
    brackets = '()[]'
    for bracket in brackets:
        if bracket in x:
            return False
    return True

def clean_dataset(
    dataset, 
    columns_to_remove,
    chars_to_remove_regex, 
    chars_to_replace, 
    input_columns=['sentence',],
    ):
    dataset = dataset.remove_columns(columns_to_remove)
    dataset = dataset.map(lambda x: remove_special_characters(x, chars_to_remove_regex))
    dataset = dataset.map(lambda x: replace_hatted_characters(x, chars_to_replace))
    dataset = dataset.filter(remove_rows_with_brackets, input_columns=input_columns)
    return dataset

def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

def generate_vocab_json(datasets, output_path='vocab.json'):
    print('Generating vocabulary of characters...')
    vocab_set = set()
    for dataset in datasets:
        vocab = dataset.map(
            extract_all_chars, batched=True, batch_size=-1, 
            keep_in_memory=True, remove_columns=dataset.column_names
        )
        vocab_set.update(set(vocab['vocab'][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(list(vocab_set)))}
    vocab_dict['|'] = vocab_dict[' ']
    del vocab_dict[' ']

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    print('Final vocab dict:')
    print(vocab_dict)

    with open(output_path, 'w', encoding='utf8') as vocab_file:
        json.dump(vocab_dict, vocab_file)
    print(f'Vocabulary json generated. Output path: {output_path}')

    return output_path

def prepare_dataset(batch, processor):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch