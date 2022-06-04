from paddlenlp.transformers.fnet.tokenizer import FNetTokenizer
import os

HERE = os.path.dirname(os.path.abspath(__file__))


sentencepiece_model_file = os.path.join(
    HERE, "model_checkpoints", "original_pytorch_huggingface", "sentencepiece.bpe.model"
)

do_lower_case = False

remove_space = True

tokenizer = FNetTokenizer(
    sentencepiece_model_file=sentencepiece_model_file,
    do_lower_case=do_lower_case,
    remove_space=remove_space,
)

tokenizer.tokenize("")
