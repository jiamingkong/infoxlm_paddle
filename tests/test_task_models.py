# to test the task specific models

from infoxlm_paddle import (
    InfoXLMTokenizer,
    InfoXLMModel,
    InfoXLMForSequenceClassification,
)

import os
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

my_base_model = InfoXLMModel.from_pretrained(
    os.path.join(ROOT, "model_checkpoints/converted_paddle")
)


def test_sequence_classification_model():
    my_sequence_model = InfoXLMForSequenceClassification(
        my_base_model, num_classes=3, dropout=0.1
    )
    my_sequence_model.eval()
    test_sentences = [
        "This is a test sentence.",
        "这是一个测试句子。",
        "Das ist ein Test.",
        "C'est un test.",
        "Esto es una prueba.",
        "இவ்வில் ஒரு நிர்வாகம்.",
        "产品 和 地理 是 什么 使 奶油 抹 霜 工作 .",
    ]
    for sentence in test_sentences:
        tokens_rhs = my_tokenizer(
            sentence, "another sentence just to try out the text pair"
        )
        # tokens_rhs = dirty_fix(tokens_lhs)
        input_ids = paddle.to_tensor([tokens_rhs["input_ids"]])
        token_type_ids = paddle.to_tensor([tokens_rhs["token_type_ids"]])

        # run it through the model
        logits = my_sequence_model(input_ids, token_type_ids=token_type_ids)
        logits = logits.numpy()
        print(logits)
        assert logits.shape == (1, 3)
