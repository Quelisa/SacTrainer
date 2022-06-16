from config import args
from model import SacBert
from trainer import SacTrainer
from pretrained_model import BertConfig
import torch
import os
from transformer import AutoTokenizer
from dataset import SacDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    config_file = os.path.join(args.model_name_or_path, "config.json")
    config = BertConfig.from_json_file(config_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = SacBert.from_pretrained(args.model_name_or_path, config=config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    if args.do_train:
        train_dataset = SacDataset(os.path.join(args.data_path, 'train.txt'),
                                   args.max_len,
                                   tokenizer,
                                   is_train=True)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True)
        if args.mode == 'retrieve':
            valid_intent_dataset = SacDataset(os.path.join(
                args.data_path, 'intent.txt'),
                                              args.max_len,
                                              tokenizer,
                                              is_train=False)
            valid_intent_dataloader = DataLoader(valid_intent_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=True)
            valid_query_dataset = SacDataset(os.path.join(
                args.data_path, 'intent.txt'),
                                             args.max_len,
                                             tokenizer,
                                             is_train=False)
            valid_query_dataloader = DataLoader(valid_query_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True)
        else:
            valid_dataset = SacDataset(os.path.join(args.data_path,
                                                    'test.txt'),
                                       args.max_len,
                                       tokenizer,
                                       is_train=False)
            valid_dataloader = DataLoader(valid_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True)
    if args.do_eval:
        if args.mode == 'retrieve':
            test_intent_dataset = SacDataset(os.path.join(
                args.data_path, 'intent.txt'),
                                             args.max_len,
                                             tokenizer,
                                             is_train=False)
            test_intent_dataloader = DataLoader(test_intent_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True)
            test_query_dataset = SacDataset(os.path.join(
                args.data_path, 'intent.txt'),
                                            args.max_len,
                                            tokenizer,
                                            is_train=False)
            test_query_dataloader = DataLoader(test_query_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
        else:
            test_dataset = SacDataset(os.path.join(args.data_path, 'test.txt'),
                                      args.max_len,
                                      tokenizer,
                                      is_train=False)
            test_dataloader = DataLoader(test_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True)

    trainer = SacTrainer(
        args=args,
        device=device,
        model=model,
        train_dataset=train_dataloader if args.do_train else None,
        valid_intent_dataset=valid_intent_dataloader
        if args.do_train and args.mode == 'retrieve' else None,
        valid_query_dataset=valid_query_dataloader
        if args.do_train and args.mode == 'retrieve' else None,
        valid_dataset=valid_dataloader
        if args.do_train and args.mode == 'classify' else None,
        optimizers=(optimizer, lr_scheduler),
    )

    if args.do_train:
        trainer.train()
    if args.do_eval:
        if args.mode == 'retrieve':
            trainer.eval_retrieve(intent_dataloader=test_intent_dataloader, query_dataloader=test_query_dataloader)
        elif args.mode == 'classify':
            trainer.eval_classify(dataloader=test_dataloader)
        else:
            raise ValueError(
                "mode type error, only support retrieve and classify")


if __name__ == '__main__':
    main()
