import os
from paddle import inference
import numpy as np
from scipy.special import softmax
from infoxlm_paddle import InfoXLMTokenizer
from paddlenlp.data import Pad


class InferenceEngine(object):
    """
    The Inference Engine
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.predictor, self.config, self.input_tensors, self.output_tensors = self.load_predictor(
            os.path.join(args.model_dir, "inference.pdmodel"),
            os.path.join(args.model_dir, "inference.pdiparams"))

        # build transforms
        self.tokenizer = InfoXLMTokenizer.from_pretrained(args.model_dir)
        self.batchify_fn = Pad(axis=0,
                               pad_val=self.tokenizer.pad_token_id,
                               dtype="int64")

        # warm up
        if self.args.warmup > 0:
            for idx in range(args.warmup):
                input_ids = np.random.randint(
                    5, 200, size=(4, 32)).astype("int64")
                langs = np.ones_like(input_ids).astype(
                    "int64") * self.tokenizer.lang2id["en"]
                self.input_tensors[0].copy_from_cpu(input_ids)
                self.input_tensors[1].copy_from_cpu(langs)
                self.predictor.run()
                self.output_tensors[0].copy_to_cpu()
        return