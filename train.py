from model import GPT3ForCausalLM
from transformers import BertTokenizerFast

import torch
import transformers
import config
from preprocess import load_dataset

torch.manual_seed(2023)

model = GPT3ForCausalLM.from_pretrained("HuiHuang/gpt3-damo-base-zh")
tokenizer = BertTokenizerFast.from_pretrained("HuiHuang/gpt3-damo-base-zh")

train_loader, valid_loader = load_dataset('./pkl_data/train.pkl', './pkl_data/valid.pkl')

def accuracy(y,y_hat,ign_idx):
    y_hat = y_hat[...,:-1,:].contiguous().view(-1, y_hat.size(-1))
    y = y[..., 1:].contiguous().view(-1)
    _, y_hat = y_hat.max(dim=-1)
    mask = y.ne(ign_idx)
    n_correct = y_hat.eq(y).masked_select(mask).sum().item()
    n_word = mask.sum().item()
    return n_correct, n_word


def train_epoch(optimizer, scheduler, epoch_idx):
    model.train()
    device = config.device
    ign_idx = config.ignore_idx
    total_loss = 0
    all_correct, all_word = 0,0
    for i, (input_ids, y) in enumerate(train_loader):
        # predict
        input_ids = input_ids.to(device)
        y = y.to(device)
        out = model.forward(input_ids,labels=y)
        y_hat = out.logits

        # loss & backward
        loss = out.loss
        loss = loss.mean()
        total_loss += loss.item()
        loss /= config.grad_accumulation_step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        # accuracy
        batch_correct, batch_word = accuracy(y,y_hat,ign_idx)
        all_correct += batch_correct
        all_word += batch_word
        batch_acc = batch_correct / batch_word

        if (i + 1) % config.grad_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        print(f'Epoch {epoch_idx}, batch {i}, loss {config.grad_accumulation_step * loss.item()}, '
              f'batch_acc {batch_acc}, lr {scheduler.get_lr()}')

def valid_epoch():
    pass


def train():
    pass

if __name__ == '__main__':






















if __name__ == '__main__':

