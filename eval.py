# to test the task specific models

from infoxlm_paddle import (
    InfoXLMTokenizer,
    InfoXLMModel,
    InfoXLMForSequenceClassification,
)

import os
import numpy as np
import paddle
from experiments.xnli import XNLI_Dataset
from paddle.metric import Accuracy
from utils import XNLI_LANGS

HERE = os.path.dirname(os.path.abspath(__file__))
# ROOT = os.path.dirname(HERE)


sentencepiece_model_file = os.path.join(
    HERE, "model_checkpoints", "original_pytorch_huggingface", "sentencepiece.bpe.model"
)
do_lower_case = False
remove_space = True

my_tokenizer = InfoXLMTokenizer(
    sentencepiece_model_file=sentencepiece_model_file,
    do_lower_case=do_lower_case,
    remove_space=remove_space,
)

finetuned_xnli = InfoXLMForSequenceClassification.from_pretrained(
    os.path.join(HERE, "model_checkpoints/finetuned_paddle")
)
finetuned_xnli.eval()

dataset = XNLI_Dataset("ar", "test")

metric = Accuracy()


@paddle.no_grad()
def eval_accuracy(dataset, model, batch_size=8):
    model.eval()
    metric.reset()
    for pre, hyp, lbl in dataset.get_batch_iterator(batch_size):
        encoded_inputs = my_tokenizer(pre, hyp, padding=True)
        input_token_ids = paddle.to_tensor(encoded_inputs["input_ids"])
        logits = model(input_token_ids)
        correct = metric.compute(logits, paddle.to_tensor(lbl))
        metric.update(correct)
    acc = metric.accumulate()
    return acc


if __name__ == "__main__":
    for lang in XNLI_LANGS:
        dataset = XNLI_Dataset(lang, "test")
        acc = eval_accuracy(dataset, finetuned_xnli)
        print(f"{lang} acc: {acc:.4f}")
