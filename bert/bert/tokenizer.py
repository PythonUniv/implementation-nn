from tokenizers import Tokenizer
from tokenizers import Encoding


tokenizer: Tokenizer = Tokenizer.from_pretrained('bert-base-cased')


if __name__ == '__main__':
    # tokenizer.enable_padding()
    print(tokenizer.encode('[UNK] [CLS] [PAD] [SEP] [MASK]').ids)
    encoding: list[Encoding] = tokenizer.encode_batch(['War', 'Be your friend!'])
    print(encoding[0].ids)
    print(encoding[0].attention_mask)
