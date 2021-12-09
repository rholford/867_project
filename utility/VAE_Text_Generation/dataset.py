import re
#break resolved by installing torch text
#just a data structure, should be fine
from torchtext.legacy import data
#attempted alt: https://www.tensorflow.org/guide/data   
#https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
#from tensorflow  import data

my_punc = "!\"#$%&\()*+?_/:;[]{}|~,`"
table = dict((ord(char), u' ') for char in my_punc)


def clean_str(string):
    string = re.sub(r"\'s ", " ", string)
    string = re.sub(r"\'m ", " ", string)
    string = re.sub(r"\'ve ", " ", string)
    string = re.sub(r"n\'t ", " not ", string)
    string = re.sub(r"\'re ", " ", string)
    string = re.sub(r"\'d ", " ", string)
    string = re.sub(r"\'ll ", " ", string)
    string = re.sub("-", " ", string)
    string = re.sub(r"@", " ", string)
    string = re.sub('\'', '', string)
    string = string.translate(table)
    string = string.replace("..", "").strip()
    return string


def tokenizer_function(text):                   # create a tokenizer function for data field
    # text = clean_str(text)
    text = [x for x in text.split(" ") if x != "" and x.find(" ") == -1]
    return text


class MyDataset(data.Dataset):
    def __init__(self, path, text_field, **kwargs):
        fields = [('text', text_field)]
        examples = []
        with open(path, 'r') as f:
            for text in f:
                examples.append(data.Example.fromlist([text], fields))
        super(MyDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, train='train', **kwargs):
        return super(MyDataset, cls).splits(text_field=text_field, train=train, **kwargs)


def get_iterators(opt, path, fname):
    text_field = data.Field(init_token=opt.start_token, eos_token=opt.end_token, lower=True, tokenize=tokenizer_function, batch_first=True)
    train_data, val_data = MyDataset.splits(path=path, train=fname, test=fname, text_field=text_field)
    text_field.build_vocab(train_data, val_data, max_size=opt.n_vocab-4, vectors='glove.6B.' + str(opt.n_embed) + 'd')
    train_vocab = text_field.vocab
    train_iter, val_iter = data.BucketIterator.splits((train_data, val_data), batch_size=opt.batch_size, sort_key=lambda x: len(x.text), repeat=False)
    return train_iter, val_iter, train_vocab
