import torch
from torch import nn
import os
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm


class SacTrainer(object):
    def __init__(self, args, device, model, train_dataloader, valid_dataloader, valid_intent_dataloader, valid_query_dataloader, optimizers):
        super().__init__()
        self.model = model
        self.device = device
        self.args = args
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.valid_intent_dataloader = valid_intent_dataloader
        self.valid_query_dataloader = valid_query_dataloader
        self.optimizer = optimizers[0]
        self.lr_scheduler = optimizers[1]
        self.index = None

    def train(self):
        for epoch in range(self.args.epoch_num):
            train_losses = []
            self.model.train()
            for i, batch_data in enumerate(tqdm(self.train_dataloader)):
                input_data, labels = batch_data
                for d in input_data:
                    input_data[d] = input_data[d].to(self.device)
                labels = labels.to(self.device)

                if self.args.mode == 'retrieve':
                    intent_vector = self.model(input_data["input_ids"], input_data["attention_mask"], input_data["token_type_ids"], self.args.mode)
                    logits = cosine_similarity(intent_vector, intent_vector)
                elif self.args.mode == 'classify':
                    logits = self.model(input_data, self.args.mode)

                loss_func = nn.CrossEntropyLoss()
                loss = loss_func(logits, labels)

                loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
                train_losses.append(loss.item())

                if ((i+1) % self.args.gradient_accumulation_steps) == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

            train_loss = train_losses.mean()
            if self.args.mode == 'retrieve':
                acc = self.eval_retrieve(intent_dataloader=self.valid_intent_dataloader, query_dataloader=self.query_dataloader)
            elif self.args.mode == 'classify':
                acc = self.eval_classify(dataloader=self.valid_dataloader)
            print("epoch:%d | train_loss:%f | valid_acc:%f " % (epoch, train_loss, acc))
        torch.save(self.model.state_dict(), os.path.join(self.args.output_dir, "pytorch_model.bin"))

    def eval_retrieve(self, top_k=1, threshold=0.1, intent_dataloader=None, query_dataloader=None):
        self.model.eval()
        with torch.no_grad():

            embeddings_list = []
            labels_list = []
            for _, batch_data in enumerate(tqdm(intent_dataloader)):
                input_data, labels = batch_data
                for d in input_data:
                    input_data[d] = input_data[d].to(self.device)
                labels = labels.to(self.device)

                intent_vector = self.model(input_data["input_ids"], input_data["attention_mask"], input_data["token_type_ids"], self.args.mode)

                embeddings_list.append(intent_vector.cpu())
                labels_list.append(labels.cpu())
            embeddings = torch.cat(embeddings_list, 0)

            self.index["index"] = embeddings
            self.index["labels"] = torch.cat(labels_list, 0)

            total_num = 0
            acc_num = 0
            for _, batch_data in enumerate(tqdm(query_dataloader)):
                input_data, labels = batch_data
                for d in input_data:
                    input_data[d] = input_data[d].to(self.device)
                labels = labels.to(self.device)
                
                query_vector = self.model(input_data["input_ids"], input_data["attention_mask"], input_data["token_type_ids"], self.args.mode)

                similarities = cosine_similarity(query_vector, self.index["index"])
                id_and_score = []
                for i, s in enumerate(similarities):
                    if s >= threshold:
                        id_and_score.append((i, s))
                id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[:top_k]
                results = [(self.index["labels"][idx], score) for idx, score in id_and_score]

                total_num += len(labels)
                for i, res in enumerate(results):
                    idx, score = res
                    if self.index["labels"][idx] == labels[i]:
                        acc_num += 1
            acc = acc_num/total_num
            print("top_k:%d | acc:%f" % (top_k, acc))
            return acc

    def eval_classify(self, dataloader=None):
        self.model.eval()
        with torch.no_grad():

            losses = []
            total_num = 0
            acc_num = 0
            for _, batch_data in enumerate(tqdm(dataloader)):
                input_data, labels = batch_data
                for d in input_data:
                    input_data[d] = input_data[d].to(self.device)
                labels = labels.to(self.device)

                logits = self.model(input_data["input_ids"], input_data["attention_mask"], input_data["token_type_ids"], self.args.mode)

                loss_func = nn.CrossEntropyLoss()
                loss = loss_func(logits, labels)
                losses.append(loss.item())

                pred = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).cpu().numpy()
                total_num += len(labels)
                for predict, label in zip(pred, labels.cpu()):
                    if predict == label:
                        acc_num += 1

            acc = acc_num/total_num
            print("loss:%f | acc:%f" % (losses.mean(), acc))
            return acc
