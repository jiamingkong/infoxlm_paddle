import os
import argparse
import torch
import paddle
from collections import OrderedDict

DONT_TRANSPOSE = [
    ".layer_norm",  # verified
    ".position_embeddings",  # verified
    ".embeddings",  # verified
]

# https://paddlenlp.readthedocs.io/zh/latest/community/contribute_models/convert_pytorch_to_paddle.html#id4
keys_dict = {
    "encoder.layer": "encoder.layers",
    "attention.self.query": "self_attn.q_proj",
    "attention.self.key": "self_attn.k_proj",
    "attention.self.value": "self_attn.v_proj",
    "attention.output.dense": "self_attn.out_proj",
    "attention.output.LayerNorm.weight": "norm1.weight",
    "attention.output.LayerNorm.bias": "norm1.bias",
    "intermediate.dense": "linear1",
    "output.dense": "linear2",
    "output.LayerNorm.weight": "norm2.weight",
    "output.LayerNorm.bias": "norm2.bias",
}


def need_transpose(key, value_dim):
    # if ('linear' in key) or ('proj' in  key) or ('vocab' in  key and 'weight' in  key) or ("dense.weight" in key) or ('transform.weight' in key) or ('seq_relationship.weight' in key) and value_dim == 2:
    #     return True
    for d in DONT_TRANSPOSE:
        if d in key:
            return False
    # only transpose if value is 2d
    if value_dim == 2:
        return True


def convert_torch_to_paddle(torch_weight_path, paddle_weight_path):
    """
    Quickly convert torch state dict into a paddle weight
    """
    # Load the torch model and map to CPU
    torch_weights = torch.load(torch_weight_path, map_location="cpu")
    # prepare an empty OrderedDict
    paddle_weights = OrderedDict()
    # iterate through the keys and convert the tensors
    for key, value in torch_weights.items():
        _key = key
        transposed = False
        value_dim = value.ndim

        if key.endswith(".weight"):
            if need_transpose(key, value_dim):
                value = value.transpose(0, 1)
                transposed = True
        for k in keys_dict:
            if k in key:
                key = key.replace(k, keys_dict[k])
        if "pred_layer.proj.weight" in key:
            # skip the pred_layer if the model has any
            print(f"Skipping {key}")
            continue
        if "pred_layer.proj.bias" in key:
            key = key.replace("pred_layer.proj.bias", "pred_layer.bias")
        print(
            f"{_key} ==> {key} ({value.shape}, transposed = {transposed}, sum = {value.sum()})"
        )
        paddle_weights[key] = paddle.to_tensor(value.numpy())

    paddle.save(paddle_weights, paddle_weight_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch", type=str, required=True)
    parser.add_argument("--paddle", type=str, required=True)
    args = parser.parse_args()
    convert_torch_to_paddle(args.torch, args.paddle)
