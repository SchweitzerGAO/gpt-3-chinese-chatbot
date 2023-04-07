from gpt3 import GPT3ForCausalLM
from transformers import BertTokenizerFast


model = GPT3ForCausalLM.from_pretrained("HuiHuang/gpt3-damo-large-zh")
tokenizer = BertTokenizerFast.from_pretrained("HuiHuang/gpt3-damo-large-zh")


if __name__ == '__main__':

    pass

