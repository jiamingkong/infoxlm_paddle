# %%
from infoxlm_paddle import InfoXLMModel
from transformers import XLMRobertaTokenizer, AutoModel
import os
import paddle
import torch
import numpy as np

from typing import Union

HERE = os.path.dirname(os.path.abspath(__file__))
# up one level
ROOT = os.path.dirname(HERE)


def paddle2np(data: Union[paddle.Tensor, dict] = None):
    if isinstance(data, dict):
        np_data = {}
        for k, v in data.items():
            np_data[k] = v.numpy()
        return np_data
    else:
        return {"output": data.numpy()}


def torch2np(data):
    if isinstance(data, dict):
        np_data = {}
        for k, v in data.items():
            np_data[k] = v.detach().numpy()
        return np_data
    else:
        return {"output": data.detach().numpy()}


hf_model = AutoModel.from_pretrained(
    os.path.join(ROOT, "model_checkpoints/original_pytorch_huggingface"),
    local_files_only=True,
)

hf_model.eval()


torch_tensor = torch.tensor([[0, 186513, 8938, 3219, 164269, 938, 5, 2]])
paddle_tensor = paddle.to_tensor([[0, 186513, 8938, 3219, 164269, 938, 5, 2]])


my_model = InfoXLMModel.from_pretrained(
    os.path.join(ROOT, "model_checkpoints/converted_paddle")
)

my_model.eval()


# %% [markdown]
# ## Making sure the embeddings are OK

# %%


def test_embeddings_are_ok():
    a = my_model.embeddings.forward(paddle_tensor)
    b = hf_model.embeddings.forward(torch_tensor)
    a_np = paddle2np(a)["output"]
    b_np = torch2np(b)["output"]

    # numpy all close
    assert np.allclose(a_np, b_np, rtol=1e-3)


# %% [markdown]
# ## Making sure the forward is OK

# %%
def test_forward_is_ok():
    output = hf_model(torch_tensor)
    lhs = output["last_hidden_state"].detach().numpy()
    lhs2 = output["pooler_output"].detach().numpy()

    result = my_model(paddle_tensor)
    rhs = result[0].numpy()
    rhs2 = result[1].numpy()

    # numpy all close

    assert np.abs(lhs[0] - rhs[0]).max() < 1e-2

    assert np.abs(lhs2[0] - rhs2[0]).max() < 1e-2
