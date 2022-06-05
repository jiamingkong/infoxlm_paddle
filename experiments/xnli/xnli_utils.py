# this script extends some utility function to train the datasets from paddlenlp.datasets.xnli

from paddlenlp.datasets import load_dataset

__all__ = ["XNLI_Dataset", "convert_examples", "convert_example"]


class XNLI_Dataset(object):
    def __init__(self, lang, split):
        self.lang = lang
        self.split = split
        self.data = load_dataset("xnli", lang, splits=[split])

    def __getitem__(self, index):
        return self.data[index]

    def collate(self, start, end=None):
        if end is None:
            end = start + 1
        raw_data = self.data[start:end]
        # collected into sentence pairs
        # (premise, hypothesis, label)
        premises = [i["premise"] for i in raw_data]
        hypotheses = [i["hypothesis"] for i in raw_data]
        labels = [i["label"] for i in raw_data]
        return premises, hypotheses, labels

    def __len__(self):
        return len(self.data)

    def get_batch_iterator(self, batch_size):
        total_batches = len(self) // batch_size + 1
        for i in range(total_batches):
            start = i * batch_size
            end = start + batch_size
            yield self.collate(start, end)


# also provide the convert examples function to utilize the convenient Trainer API
def convert_examples(examples, tokenizer):
    premises = [i["premise"] for i in examples]
    hypotheses = [i["hypothesis"] for i in examples]
    labels = [i["label"] for i in examples]
    tokenized_examples = tokenizer(
        text=premises, text_pair=hypotheses, pad_to_max_seq_len=True, max_seq_len=128
    )
    for idx in range(len(tokenized_examples)):
        tokenized_examples[idx]["label"] = [examples[idx]["label"]]


def convert_example(example, tokenizer, max_seq_len=128):
    encoded_inputs = tokenizer(
        text=example["premise"],
        text_pair=example["hypothesis"],
        max_seq_len=max_seq_len,
        pad_to_max_seq_len=True,
    )
    encoded_inputs["label"] = int(example["label"])
    return encoded_inputs


if __name__ == "__main__":
    a = XNLI_Dataset("en", "train")
    print(a.collate(0, 10))
