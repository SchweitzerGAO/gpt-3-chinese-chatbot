import torch
"""
hyper-parameters
"""
epoch = 15
batch_size = 32768
max_len = 150
lr = 2.6e-5
weight_decay = 1e-9
grad_accumulation_step = 5
max_grad_norm = 2
num_workers = 0
ignore_idx = -100
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
warm_up_steps = 4000
