from torch.utils.data import Dataset
import torch


class SacDataset(Dataset):
    def __init__(self,
                 dataset_filepath,
                 max_len,
                 tokenizer,
                 label_separator="##"):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.labels = []
        self.sentences = []

        for _, line in enumerate(open(dataset_filepath, 'r',
                                      encoding='utf-8')):
            line = line.strip().split(label_separator)
            print("get line from file: ", line)
            if len(line) == 1:
                sentence = line[0]
                label = [
                    -1,
                ]
            elif len(line) == 2:
                sentence = line[0]
                label = int(line[1])
            else:
                raise NotImplementedError(
                    "text not support more than 2 inputs")

            self.labels.append(label)
            self.sentences.append(sentence)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sentence = self.sentences[idx]
        label = self.labels[idx]

        input_token = self.tokenizer.tokenize(sentence)
        input_token = ['[CLS]'] + input_token + ['SEP']
        input_id = self.tokenizer.convert_tokens_to_ids(input_token)
        print("sentence: ", sentence, "input_token: ", input_token,
              "input_id: ", input_id)

        return {"input_ids": input_id}, label


def pad_sequence(input_id, max_len, padding_token):
    cur_len = len(input_id)
    if cur_len > max_len:
        return input_id[:max_len], [1] * max_len
    else:
        return input_id + [padding_token] * (max_len-cur_len), [1] * cur_len + [0] * (max_len-cur_len)


def pad_collate_fn(batch, max_len=32, padding_token=0):
    input_ids, token_type_ids, attention_mask, labels = [], [], [], []
    for input_data, label in batch:
        input_id = input_data["input_ids"]
        input_id_pad, attention_mask_id_pad = pad_sequence(input_id, max_len, padding_token)
        token_type_id_pad = [0] * len(input_id_pad)

        input_ids.append(input_id_pad)
        token_type_ids.append(token_type_id_pad)
        attention_mask.append(attention_mask_id_pad)
        labels.append(label)

    return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "attention_mask": torch.tensor(attention_mask, dtype=torch.long), "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long)}, torch.tensor(labels, dtype=torch.long)


