# converting the XNLI datasets from huggingface into a format that is readable for paddlepaddle
from paddlenlp.datasets import load_dataset
train_ds, test_ds = load_dataset("xnli", splits=("train", "test"))