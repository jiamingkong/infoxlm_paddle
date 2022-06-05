# this script extends some utility function to train the datasets from paddlenlp.datasets.xnli

from paddlenlp.datasets import load_dataset

train_ds, test_ds = load_dataset("xnli", "ar", splits=["train", "test"])

