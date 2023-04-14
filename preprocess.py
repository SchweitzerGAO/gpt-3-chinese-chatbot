import json
import pickle as pkl

from tqdm import tqdm
from transformers import BertTokenizerFast

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import config
import torch.nn.utils.rnn as rnn_utils

tokenizer = BertTokenizerFast.from_pretrained("HuiHuang/gpt3-damo-base-zh")


def json_to_txt(rpath, wpath):
    clean_data = []
    with open(rpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for dialog in data:
        for i in range(len(dialog)):
            dialog[i] = dialog[i].replace(' ', '')
        clean_data.append('\n'.join(dialog))
    with open(wpath, 'w', encoding='utf-8') as f:
        for d in clean_data:
            f.write(d)
            f.write('\n\n')


def preprocess_lccc(rpath, wpath):
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    with open(rpath, 'r', encoding='utf-8') as f:
        data = f.read()
    data = data.split('\n\n')
    proceeded_dialogs = []
    for i, dialog in enumerate(tqdm(data[0:int(2**21)])):
        sentences = dialog.split('\n')
        input_ids = [cls_id]
        for s in sentences:
            input_ids += tokenizer.encode(s, add_special_tokens=False)
            input_ids.append(sep_id)
        proceeded_dialogs.append(input_ids)
    with open(wpath, 'wb') as f:
        pkl.dump(proceeded_dialogs, f)
    print('Finished')


def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels


class LCCCDataset(Dataset):
    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __getitem__(self, index):
        input_ids = self.input_list[index]
        input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def __len__(self):
        return len(self.input_list)


def load_dataset(train_path, valid_path):
    with open(train_path, 'rb') as f:
        train_data = pkl.load(f)
    with open(valid_path, 'rb') as f:
        valid_data = pkl.load(f)
    pass
    train_set = LCCCDataset(train_data, config.max_len)
    valid_set = LCCCDataset(valid_data, config.max_len)
    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers, drop_last=True,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers, drop_last=True,
                              collate_fn=collate_fn)
    return train_loader, valid_loader


if __name__ == '__main__':
    # json_to_txt('./LCCC-base-split/LCCC-base_valid.json','./data/valid.txt')
    preprocess_lccc('./data/train.txt', './pkl_data/train.pkl')
