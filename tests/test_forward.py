#!/usr/bin/python
# -*- coding: UTF-8 -*-

from transformers import XLMRobertaTokenizer, XLMRobertaModel, AutoModel, AutoTokenizer

# from paddlenlp.transformers.fnet.tokenizer import FNetTokenizer
from infoxlm_paddle import InfoXLMTokenizer, dirty_fix, InfoXLMModel
import os
import torch
import numpy as np
import paddle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


sentencepiece_model_file = os.path.join(
    ROOT, "model_checkpoints", "original_pytorch_huggingface", "sentencepiece.bpe.model"
)
do_lower_case = False
remove_space = True

my_tokenizer = InfoXLMTokenizer(
    sentencepiece_model_file=sentencepiece_model_file,
    do_lower_case=do_lower_case,
    remove_space=remove_space,
)
my_model = InfoXLMModel.from_pretrained(
    os.path.join(ROOT, "model_checkpoints/converted_paddle")
)
my_model.eval()

local_directory = os.path.join(ROOT, "model_checkpoints/original_pytorch_huggingface")

true_tokenizer = AutoTokenizer.from_pretrained(local_directory, local_files_only=True)
true_model = AutoModel.from_pretrained(local_directory, local_files_only=True)
true_model.eval()

test_sentences = [
    "This is a test sentence.",
    "这是一个测试句子。",
    "Das ist ein Test.",
    "C'est un test.",
    "Esto es una prueba.",
    "இவ்வில் ஒரு நிர்வாகம்.",
    "产品 和 地理 是 什么 使 奶油 抹 霜 工作 .",
    "Man verliert die Dinge auf die folgende Ebene , wenn sich die Leute erinnern .",
    "अगर लोग याद करते हैं तो आप निम ् न स ् तर पर चीज ़ ें खो देते हैं ."
]


def run_a_sentence(sentence):
    tokens_lhs = true_tokenizer(sentence)
    tokens_rhs = dirty_fix(my_tokenizer(sentence))
    # assert the input_ids are the same using list equal
    assert np.array_equal(
        np.array(tokens_lhs["input_ids"]), np.array(tokens_rhs["input_ids"])
    )


def test_run_sentences():
    for sentence in test_sentences:
        run_a_sentence(sentence)


def forward_using_hf(sentence):
    # tokenize using true_tokenizer
    tokens_lhs = true_tokenizer(sentence, return_tensors="pt")
    # run it through the model
    outputs = true_model(**tokens_lhs)
    # get the last hidden state
    last_hidden_state = outputs[0]
    # to numpy
    last_hidden_state = last_hidden_state.detach().cpu().numpy()
    # return the last hidden state
    return last_hidden_state


def forward_using_paddle(sentence):
    # tokenize using true_tokenizer
    tokens_lhs = my_tokenizer(sentence)
    # run it through the model
    input_ids = paddle.to_tensor([dirty_fix(tokens_lhs)["input_ids"]])
    print(input_ids)
    outputs = my_model(input_ids)
    # get the last hidden state
    last_hidden_state = outputs[0]
    # to numpy
    last_hidden_state = last_hidden_state.detach().cpu().numpy()
    # return the last hidden state
    return last_hidden_state


def test_forward():
    for sentence in test_sentences:
        # run it through the model
        last_hidden_state_lhs = forward_using_hf(sentence)
        last_hidden_state_rhs = forward_using_paddle(sentence)
        # assert the last hidden states are the same
        assert np.abs(last_hidden_state_lhs - last_hidden_state_rhs).max() < 1e-2
