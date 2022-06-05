# infoxlm_paddle
Implementing InfoXLM's code base and training process with [飞桨PaddlePaddle-源于产业实践的开源深度学习平台](https://www.paddlepaddle.org.cn/)

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
    python -m pytest ./tests/
    ```

    You should be able to pass the 3 included tests successfully. This means that:
    - The tokenizer, after a dirty fix, is able to generated the same tokenized ids as the huggingface implementation
    - The model weights are correct after conversion

## 2. Train the model for downstream tasks

Following the [original InfoXLM paper](https://arxiv.org/abs/2007.07834), we provided the script to finetune the model for XNLI downstream task. We utilized the convenient API provided by `paddlenlp.datasets` to train the model. Unfortunately the Trainer API would still drive into `loss=nan` issue, as mentioned by some peer programmers, so we implemented our manual training process in `trainer_manual.py`. Please use `trainer_api_script.py` with caution as it probably won't run

**To train the model**

```bash
python trainer_manual.py \
    --train_batch_size=8 \
    --max_length=128 \
    --learning_rate=1e-5 \
    --fp16 \
    --train_lang=en \
    --eval_lang=zh
```

The finetuned weights are provided for your convenience, the download instruction is in "model_checkpoints\finetuned_paddle\download.txt".



**To eval the model**

There is a cleaner script to evaluate the model on all 15 languages. Please download the finetuned weights as instructed above, then run the command below. 

```bash
python eval.py
```

The results are shown here, comparing to the original paper report:

| Languages     | Paddle Version           | Original Paper |
| ------------- | ------------------------ | -------------- |
| AR            | 0.731                    | 0.742          |
| BG            | 0.787                    | 0.793          |
| DE            | 0.781                    | 0.793          |
| EL            | 0.771                    | 0.778          |
| EN            | 0.854                    | 0.864          |
| ES            | 0.805                    | 0.809          |
| FR            | 0.799                    | 0.803          |
| HI            | 0.711                    | 0.722          |
| RU            | 0.768                    | 0.776          |
| SW            | 0.674                    | 0.675          |
| TH            | 0.741                    | 0.746          |
| TR            | 0.750                    | 0.756          |
| UR            | 0.668                    | 0.673          |
| VI            | 0.758                    | 0.771          |
| ZH            | 0.760                    | 0.770          |
| **-average-** | **0.760 **( within 0.5%) | **0.765**      |

**Note**

There are a few key differences between our finetuning process:

- The original paper used a much bigger batch size (256), while ours was trained with an effective batch size of 32.
- The original paper evaluated the models for every 5000 steps, while ours was only evaluated at some epoch ends.

The smaller batch size does hurt the performance by a bit, we noticed that the performance could vary from epoch to epoch wildly after the warm-up period. For example we obtained a checkpoint that scored 0.770 in ZH (almost 1.0% higher than that of the uploaded checkpoint), yet performs not as ideal in other languages.
