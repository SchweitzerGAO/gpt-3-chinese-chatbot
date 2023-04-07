

from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("HuiHuang/gpt3-damo-large-zh")


def json_to_txt(rpath, wpath):
    pass


def preprocess_lccc(rpath, wpath):
    pass


if __name__ == '__main__':
    print(tokenizer.sep_token_id)
