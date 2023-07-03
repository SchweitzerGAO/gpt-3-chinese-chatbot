# gpt-3-chinese-chatbot

A [GPT-3](https://github.com/huanghuidmml/gpt3_zh) based Chinese chatbot finetuned by [LCCC dataset](https://github.com/thu-coai/CDial-GPT)

## Fine-tuning Methods

### 1. Traditional Fine-tuning with LoRA

I fine-tuned the last `nn.Linear` layer with [PEFT](https://github.com/huggingface/peft) = Huggingface+LoRA and the preprocessing method in [this repo](https://github.com/yangjianxin1/GPT2-chitchat/) 

### 2. Instruction Fine-tuning

To be done
