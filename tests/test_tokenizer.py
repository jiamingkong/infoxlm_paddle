# set up pytest
import pytest
from transformers import XLMRobertaTokenizer, AutoTokenizer

import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# up one level
ROOT = os.path.dirname(HERE)

test_sentences = [
    "This is a test sentence.",  # [0.2496]
    "这是一个测试句子。",
    "Das ist ein Test.",
    "C'est un test.",
    "Esto es una prueba.",
    "இவ்வில் ஒரு நிர்வாகம்.",
]


def test_get_tokenizer():
    tokenizer = XLMRobertaTokenizer.from_pretrained(
        os.path.join(ROOT, "model_checkpoints/original_pytorch_huggingface"),
        local_files_only=True,
    )
    assert isinstance(tokenizer, XLMRobertaTokenizer)


def test_results_are_correct():
    my_tokenizer = XLMRobertaTokenizer.from_pretrained(
        os.path.join(ROOT, "model_checkpoints/original_pytorch_huggingface"),
        local_files_only=True,
    )
    true_tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
    for sentence in test_sentences:
        my_tokens = my_tokenizer(sentence, return_tensors="pt")
        true_tokens = true_tokenizer(sentence, return_tensors="pt")
        print(my_tokens["input_ids"])
        assert torch.equal(my_tokens["input_ids"], true_tokens["input_ids"])
