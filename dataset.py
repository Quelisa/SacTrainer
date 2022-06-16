from torch.utils.data import Dataset
import torch


class SacDataset(Dataset):
    def __init__(self,
                 dataset_filepath,
                 max_len,
                 tokenizer,
                 pad_to_max_len=True,
                 doc_id=None,
                 is_train=True,
                 label_separator="\t"):
        self.max_len = max_len
        self.pad_to_max_len = pad_to_max_len
        self.is_train = is_train
        self.tokenizer = tokenizer
        self.labels = []
        self.sentences = []

        for _, line in enumerate(open(dataset_filepath, 'r',
                                      encoding='utf-8')):
            line = line.strip().split(label_separator)
            if len(line) == 1:
                sentence = line[0]
                label = [
                    -1,
                ]
            elif len(line) == 2:
                sentence = line[0]
                label = line[1]
            else:
                raise NotImplementedError(
                    "text not support more than 2 inputs")

            if self.mode == "retrieve":
                label = [self.id2label.get(i, -1) for i in label]
            self.labels.append(label)
            self.sentences.append(sentence)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sentences = self.sentences[idx]
        labels = self.labels[idx]

        input_ids = []
        attention_mask_ids = []
        token_type_ids = []

        for _, sent in enumerate(sentences):
            input_token = self.tokenizer.tokenize(sent)
            input_id = self.tokenizer.convert_token_to_ids(input_token)

            if len(input_id) > self.max_len:
                input_id = input_id[:self.max_len]
                attention_mask_id = [1] * self.max_len
            else:
                input_id = input_id + [0] * (self.max_len - len(input_id))
                attention_mask_id = [1] * len(input_id) + [0] * (self.max_len -
                                                                 len(input_id))
            token_type_id = [0] * len(self.max_len)
            input_ids.append(input_id)
            attention_mask_ids.append(attention_mask_id)
            token_type_ids.append(token_type_id)

            input_ids.append(input_id)

        return {
            "input_ids":
            torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids":
            torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask_ids":
            torch.tensor(attention_mask_ids, dtype=torch.long),
            "labels":
            labels
        }
