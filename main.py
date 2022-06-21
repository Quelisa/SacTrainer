from config import args
from model import SacBert
from trainer import SacTrainer
import torch
import os
from transformers import BertConfig, AutoTokenizer
from dataset import SacDataset, pad_collate_fn
from torch.utils.data import DataLoader
from functools import partial

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    config = BertConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = SacBert.from_pretrained(args.model_name_or_path, config=config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    collate_fn = partial(pad_collate_fn, max_len=args.max_len, padding_token=0)

    print(" ======== begin to load data ======== ")
    if args.do_train:
        train_dataset = SacDataset(os.path.join(args.data_path, 'train.txt'),
                                   args.max_len,
                                   tokenizer)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True, collate_fn=collate_fn)
        if args.mode == 'retrieve':
            valid_intent_dataset = SacDataset(os.path.join(
                args.data_path, 'intent.txt'),
                                              args.max_len,
                                              tokenizer)
            valid_intent_dataloader = DataLoader(valid_intent_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=True, collate_fn=collate_fn)
            valid_query_dataset = SacDataset(os.path.join(
                args.data_path, 'intent.txt'),
                                             args.max_len,
                                             tokenizer)
            valid_query_dataloader = DataLoader(valid_query_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True, collate_fn=collate_fn)
        else:
            valid_dataset = SacDataset(os.path.join(args.data_path,
                                                    'test.txt'),
                                       args.max_len,
                                       tokenizer)
            valid_dataloader = DataLoader(valid_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True, collate_fn=collate_fn)
    if args.do_eval:
        if args.mode == 'retrieve':
            test_intent_dataset = SacDataset(os.path.join(
                args.data_path, 'intent.txt'),
                                             args.max_len,
                                             tokenizer)
            test_intent_dataloader = DataLoader(test_intent_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True, collate_fn=collate_fn)
            test_query_dataset = SacDataset(os.path.join(
                args.data_path, 'intent.txt'),
                                            args.max_len,
                                            tokenizer)
            test_query_dataloader = DataLoader(test_query_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True, collate_fn=collate_fn)
        else:
            test_dataset = SacDataset(os.path.join(args.data_path, 'test.txt'),
                                      args.max_len,
                                      tokenizer)
            test_dataloader = DataLoader(test_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True, collate_fn=collate_fn)

    trainer = SacTrainer(
        args=args,
        device=device,
        model=model,
        train_dataloader=train_dataloader if args.do_train else None,
        valid_intent_dataloader=valid_intent_dataloader
        if args.do_train and args.mode == 'retrieve' else None,
        valid_query_dataloader=valid_query_dataloader
        if args.do_train and args.mode == 'retrieve' else None,
        valid_dataloader=valid_dataloader
        if args.do_train and args.mode == 'classify' else None,
        optimizers=(optimizer, lr_scheduler),
    )

    if args.do_train:
        print(" ======== begin to train model ======== ")
        trainer.train()
    if args.do_eval:
        print(" ======== begin to eval model ======== ")
        if args.mode == 'retrieve':
            trainer.eval_retrieve(intent_dataloader=test_intent_dataloader, query_dataloader=test_query_dataloader)
        elif args.mode == 'classify':
            trainer.eval_classify(dataloader=test_dataloader)
        else:
            raise ValueError(
                "mode type error, only support retrieve and classify")


if __name__ == '__main__':
    main()
