# this script trains the XNLI model on English train set, then it predicts on Chinese test set
# python .\train.py --do_train --warmup_ratio=0.1 --output_dir=output_dir --per_device_train_batch_size=1


from paddlenlp.datasets import load_dataset
from experiments.xnli.xnli_utils import XNLI_Dataset, convert_examples, convert_example
from infoxlm_paddle import (
    InfoXLMTokenizer,
    InfoXLMModel,
    InfoXLMForSequenceClassification,
)
from paddlenlp.data import Dict, Stack, Pad
from paddle.io import DataLoader, BatchSampler, DistributedBatchSampler
from functools import partial
import paddle
from paddlenlp.trainer import Trainer, TrainingArguments, PdArgumentParser
import os
import numpy as np
import paddle
from training_args import training_args, data_args

# set up environment

print("Setting up environment...")
HERE = os.path.dirname(os.path.abspath(__file__))
PADDLE_WEIGHT = os.path.join(HERE, "model_checkpoints/converted_paddle")
SPM_MODEL = os.path.join(PADDLE_WEIGHT, "sentencepiece.bpe.model")

print(f"\tPADDLE_WEIGHT: {PADDLE_WEIGHT}\nSPM_MODEL: {SPM_MODEL}")

print("Setting up the tokenizer, base model and task model")

tokenizer = InfoXLMTokenizer(
    sentencepiece_model_file=SPM_MODEL, do_lower_case=False, remove_space=True
)

base_model = InfoXLMModel.from_pretrained(PADDLE_WEIGHT)
xnli_model = InfoXLMForSequenceClassification(base_model, num_classes=3, dropout=0.05)

print(
    f"\tTokenizer, base model and task model setup, with dropout = {xnli_model.dropout.p}"
)

print("Setting up the dataset")

train_batchify_fn = lambda samples, fn=Dict(
    {
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        "label": Stack(dtype="int64"),
    }
): fn(samples)

train_dataset = load_dataset("xnli", "en", splits=["train"])
train_dataset = train_dataset.map(
    partial(convert_example, tokenizer=tokenizer, max_seq_len=data_args.max_seq_len)
)

print("Train!")

trainer = Trainer(
    model=xnli_model,
    criterion=paddle.nn.loss.CrossEntropyLoss(),
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    tokenizer=tokenizer,
)

if training_args.do_train:
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_state()
