from transformers import XLMRobertaTokenizer, XLMRobertaModel, AutoModel, AutoTokenizer
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


tokenizer = XLMRobertaTokenizer.from_pretrained(
    os.path.join(HERE, "model_checkpoints/original_pytorch_huggingface"),
    local_files_only=True
)

infoxlm = XLMRobertaModel.from_pretrained(
    os.path.join(HERE, "model_checkpoints/original_pytorch_huggingface"),
    local_files_only=True
)

true_tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/infoxlm-base")
# true_infoxlm = AutoModel.from_pretrained("microsoft/infoxlm-base")
# infoxlm.eval()

test_sentences = [
    "This is a test sentence.", # [0.2496]
    "这是一个测试句子。",
    "Das ist ein Test.",
    "C'est un test.",
    "Esto es una prueba.",
    "இவ்வில் ஒரு நிர்வாகம்."
]

def test_a_sentence(sentence):
    tokens = tokenizer(sentence, return_tensors="pt")
    print(tokens)
    with torch.no_grad():
        output = infoxlm(**tokens)
    return output["last_hidden_state"]

print(test_a_sentence(test_sentences[0]))