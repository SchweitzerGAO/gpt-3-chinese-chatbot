from model import GPT3ForCausalLM
from transformers import BertTokenizerFast

import torch
import transformers
import config
from preprocess import load_dataset
from matplotlib import pyplot as plt
import os

torch.manual_seed(2023)

net = GPT3ForCausalLM.from_pretrained("HuiHuang/gpt3-damo-base-zh")
tokenizer = BertTokenizerFast.from_pretrained("HuiHuang/gpt3-damo-base-zh")

train_loader, valid_loader = load_dataset('./pkl_data/train.pkl', './pkl_data/valid.pkl')

train_losses, valid_losses, valid_accuracies = [], [], []


def plot(figure_id, to_plot, title, file_name):
    plt.figure(figure_id)
    plt.plot(to_plot)
    plt.title(title)
    plt.savefig(file_name)


def accuracy(y, y_hat, ign_idx):
    y_hat = y_hat[..., :-1, :].contiguous().view(-1, y_hat.size(-1))
    y = y[..., 1:].contiguous().view(-1)
    _, y_hat = y_hat.max(dim=-1)
    mask = y.ne(ign_idx)
    n_correct = y_hat.eq(y).masked_select(mask).sum().item()
    n_word = mask.sum().item()
    return n_correct, n_word


def train_epoch(optimizer, scheduler, epoch_idx):
    print(f'Start training epoch {epoch_idx + 1}')
    net.train()
    device = config.device
    ign_idx = config.ignore_idx
    total_loss = 0
    all_correct, all_word = 0, 0
    for i, (input_ids, y) in enumerate(train_loader):
        # predict
        input_ids = input_ids.to(device)
        y = y.to(device)
        out = net.forward(input_ids, labels=y)
        y_hat = out.logits

        # loss & backward
        loss = out.loss
        loss = loss.mean()
        total_loss += loss.item()
        loss /= config.grad_accumulation_step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)

        # accuracy
        batch_correct, batch_word = accuracy(y, y_hat, ign_idx)
        all_correct += batch_correct
        all_word += batch_word
        batch_acc = batch_correct / batch_word

        if (i + 1) % config.grad_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        print(f'Epoch: {epoch_idx + 1}, batch: {i + 1}, loss: {config.grad_accumulation_step * loss.item()}, '
              f'batch_acc: {batch_acc * 100}%, lr: {scheduler.get_lr()}')
    epoch_mean_loss = total_loss / len(train_loader)
    epoch_mean_acc = all_correct / all_word
    print(f'Training epoch: {epoch_idx + 1} loss: {epoch_mean_loss}, acc: {epoch_mean_acc * 100}%\n')
    return epoch_mean_loss


def valid_epoch(epoch_idx):
    print(f'Start validating epoch {epoch_idx + 1}')
    net.eval()
    device = config.device
    ign_idx = config.ignore_idx
    total_loss = 0
    all_correct, all_word = 0, 0
    with torch.no_grad():
        for i, (input_ids, y) in enumerate(train_loader):
            # predict
            input_ids = input_ids.to(device)
            y = y.to(device)
            out = net.forward(input_ids, labels=y)
            y_hat = out.logits

            # loss & backward
            loss = out.loss
            loss = loss.mean()
            total_loss += loss.item()

            # accuracy
            batch_correct, batch_word = accuracy(y, y_hat, ign_idx)
            all_correct += batch_correct
            all_word += batch_word
        epoch_mean_loss = total_loss / len(valid_loader)
        epoch_mean_acc = all_correct / all_word
    print(f'Validating epoch: {epoch_idx}, loss: {epoch_mean_loss}, acc: {epoch_mean_acc}\n')
    return epoch_mean_loss, epoch_mean_acc


def train():
    training_steps = len(train_loader) // config.grad_accumulation_step * config.epoch
    optimizer = transformers.AdamW(net.parameters(), lr=config.lr, eps=config.weight_decay)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=config.warm_up_steps,
                                                             num_training_steps=training_steps)
    best_valid_acc = 0
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(f'Number of parameters: {num_params}')
    for ep in range(config.epoch):
        train_loss = train_epoch(optimizer, scheduler, ep)  # train
        val_loss, val_acc = valid_epoch(ep)  # validation
        train_losses.append(train_loss)
        valid_losses.append(val_loss)
        valid_accuracies.append(val_acc)
        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            save_path = f'./saved_models/best_{ep + 1}'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            model_to_save = net.module if hasattr(net, 'module') else net
            model_to_save.save_pretrained(save_path)
    plot(0, train_losses, 'Training Loss', './plots/train_loss.png')
    plot(1, valid_losses, 'Validation Loss', './plots/valid_loss.png')
    plot(2, valid_accuracies, 'Validation Accuracy', './plots/valid_accuracy.png')


if __name__ == '__main__':
    train()
