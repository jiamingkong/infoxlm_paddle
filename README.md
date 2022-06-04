# infoxlm_paddle
Implementing InfoXLM's code base and training process with PaddlePaddle

## 0. Environment setup

It's recommended to install paddlepaddle in a separate virtual environment, with pytorch cpu-only version accompanied for verification purposes.

```bash
conda create -n infoxlm_paddle python=3.9.6
source activate infoxlm_paddle

```

## 1. Prepare the weights

Please follow the steps below to prepare the weights for paddlepaddle.

1. Download all the files from [InfoXLM](https://huggingface.co/microsoft/infoxlm-base/tree/main), and put it under `model_checkpoints/original_pytorch_huggingface`. The folder should look like:
    
    ```
    model_checkpoints/
        original_pytorch_huggingface/
            config.json
            pytorch_model.bin
            sentencepiece.bpe.model
            tokenizer.json
    ```
2. Run the following command lines, **make sure you are in the infoxlm_paddle environment**:

    ```bash
    cd model_checkpoints
    python converter.py --torch original_pytorch_huggingface/pytorch_model.bin --paddle converted_paddle/model_state.pdparams
    ```

    The converted model will be saved in `converted_paddle/model_state.pdparams`, you should see the terminal output matched the file content in `model_checkpoints/conversion_log.txt`.

3. Run the following commands to make sure the paddlepaddle weights are correct:

    ```bash
    python -m pytest ./tests/test_tokenizer.py
    python -m pytest ./tests/test_model.py
    ```

