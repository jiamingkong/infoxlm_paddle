# set up pytest
import pytest
from transformers import XLMRobertaTokenizer, AutoTokenizer

import os


from infoxlm_paddle import InfoXLMTokenizer, InfoXLMModel
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


sentencepiece_model_file = os.path.join(
    ROOT, "model_checkpoints", "original_pytorch_huggingface", "sentencepiece.bpe.model"
)
do_lower_case = False
remove_space = True

test_sentences = [
    "This is a test sentence.",  # [0.2496]
    "这是一个测试句子。",
    "Das ist ein Test.",
    "C'est un test.",
    "Esto es una prueba.",
    "இவ்வில் ஒரு நிர்வாகம்.",
]


def test_get_original_tokenizer():
    tokenizer = XLMRobertaTokenizer.from_pretrained(
        os.path.join(ROOT, "model_checkpoints/original_pytorch_huggingface"),
        local_files_only=True,
    )
    assert isinstance(tokenizer, XLMRobertaTokenizer)


def test_my_tokenizer():

    my_tokenizer = InfoXLMTokenizer(
        sentencepiece_model_file=sentencepiece_model_file,
        do_lower_case=do_lower_case,
        remove_space=remove_space,
    )

    assert True


def test_my_tokenizing_functions():
    my_tokenizer = InfoXLMTokenizer(
        sentencepiece_model_file=sentencepiece_model_file,
        do_lower_case=do_lower_case,
        remove_space=remove_space,
    )
    for sentence in test_sentences:
        tokens_lhs = my_tokenizer(sentence, max_seq_len=4)
        # assert isinstance(tokens_lhs, dict)
        # assert isinstance(tokens_lhs["input_ids"], list)
        lst = tokens_lhs["input_ids"]
        assert len(lst) == 4
    for sentence in test_sentences:
        tokens_lhs = my_tokenizer(sentence, max_seq_len=128, pad_to_max_seq_len=True)
        # assert isinstance(tokens_lhs, dict)
        # assert isinstance(tokens_lhs["input_ids"], list)
        lst = tokens_lhs["input_ids"]
        assert len(lst) == 128
        assert lst[-1] == my_tokenizer.pad_token_id


def test_compare_both_tokenizers():
    tokenizer = XLMRobertaTokenizer.from_pretrained(
        os.path.join(ROOT, "model_checkpoints/original_pytorch_huggingface"),
        local_files_only=True,
    )
    my_tokenizer = InfoXLMTokenizer(
        sentencepiece_model_file=sentencepiece_model_file,
        do_lower_case=do_lower_case,
        remove_space=remove_space,
    )
    for sentence in test_sentences:
        tokens_lhs = tokenizer(sentence, "and a premise")
        tokens_rhs = my_tokenizer(sentence, "and a premise")
        assert tokens_lhs["input_ids"] == tokens_rhs["input_ids"]


def test_get_tokenizer_from_pretrained():
    tokenizer = InfoXLMTokenizer.from_pretrained(
        os.path.join(ROOT, "model_checkpoints/finetuned_paddle")
    )
    assert tokenizer("This is a test")
