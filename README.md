# infoxlm_paddle


使用[飞桨PaddlePaddle-源于产业实践的开源深度学习平台](https://www.paddlepaddle.org.cn/) 深度学习框架复现InfoXLM 论文的实验结果

## 0. 环境配置

我们推荐在一个新的虚拟环境中使用飞桨PaddlePaddle。

```bash
conda create -n infoxlm_paddle python=3.9.6
source activate infoxlm_paddle
conda install paddlepaddle-gpu==2.3.0 cudatoolkit=11.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge -n infoxlm_paddle
```

## 1. 准备原始权重

微软团队给出的原始模型权重可以使用如下的命令导出为飞桨PaddlePaddle的权重文件。

1. 从 [Huggingface/InfoXLM](https://huggingface.co/microsoft/infoxlm-base/tree/main)下载所有的文件，并放置在目录`model_checkpoints/original_pytorch_huggingface`下，完成后，目录应该如下：

    ```
    model_checkpoints/
        original_pytorch_huggingface/
            config.json
            pytorch_model.bin
            sentencepiece.bpe.model
            tokenizer.json
    ```
2. **确认已经进入刚才创建的infoxlm_paddle虚拟环境**，执行如下操作:

    ```bash
    cd model_checkpoints
    python converter.py --torch original_pytorch_huggingface/pytorch_model.bin --paddle converted_paddle/model_state.pdparams
    ```

    转换后的权重保存在 `converted_paddle/model_state.pdparams`, 命令行的输出应该与 `model_checkpoints/conversion_log.txt`一致.

3. 执行如下命令，运行自动化测试脚本，确保模型权重已经转换成功：

    ```bash
    python -m pytest ./tests/
    ```

    您应该能看到大多数的测试都能通过，这意味着您的模型权重已经转换成功。（有部分测试函数针对的是微调后的模型）
    
## 2. 针对下游任务微调模型

按照[原始InfoXLM论文](https://arxiv.org/abs/2007.07834)，我们提供了针对下游任务微调InfoXLM的程序，其中，在调用数据方面，我们使用了方便的`paddlenlp.datasets`API。不过使用`paddlenlp.trainer`相关的API时有时候会出现loss=nan的情况，所以我们手动实现了相关的训练过程，如`trainer_manual.py`所示。

**训练模型**

```bash
python trainer_manual.py \
    --train_batch_size=8 \
    --max_length=128 \
    --learning_rate=1e-5 \
    --fp16 \
    --train_lang=en \
    --eval_lang=zh
```

微调过的权重通过网盘提供，下载信息在"model_checkpoints\finetuned_paddle\download.txt".



## 3. 评估模型

我们提供了在15个语言的数据集上进行评估的程序`eval.py`，评估结果如下：

```bash
python eval.py
```

The results are shown here, comparing to the original paper report:

| 语言     | Paddle Version           | Original Paper |
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
| **平均** | **0.760 **( 误差 0.5%) | **0.765**      |

**讨论**

我们认为两个关键的差异在于：

- 原始论文微调使用的batch_size = 256，而我们复现的版本因为硬件限制，实际有效的batch_size是32
- 原始论文每5000步即评估一次val set精度，我们只在每个epoch结束的时候评估了。

更小的batch_size 影响了一些性能，我们注意到有些epoch中，中文的精度可以达到77.0%，比起实际提交的微调版本高出一个点，但是这个epoch在其他语言上的表现平均不如提交的版本。可见更小的batch_size 导致的性能波动还是可观的。

## 4. 模型推理部署

### 4.1 导出模型

我们提供了模型的导出和部署程序，如果您下载了微调后的权重（finetuned_paddle），您可以直接使用这个权重部署模型，如下：

```bash
python TIPC/export_paddle/export_model.py --model_path=model_checkpoints/finetuned_paddle --save_inference_dir ./xnli_exported_model
```

导出成功后，目标文件夹`xnli_exported_model`会保存如下文件：

```
xnli_exported_models/
    added_tokens.json
    inference.pdiparams
    inference.pdiparams.info
    inference.pdmodel
    special_tokens_map.json
    spiece.model
    tokenizer_config.json
```

### 4.2 使用模型推理

使用这个模型进行推理的操作如下：

```bash
python TIPC/inference_paddle/infer.py --model_dir ./xnli_exported_models --use_gpu True --warmup 1 --text "You don't have to stay here.<sep>You can not leave."
```

其中--text参数提供了本次推理的输入数据，格式为`"前提<sep>假设"`，语言无需额外指定，所有语言输入都能通过提供的sentencepiece模型进行分词。推理输出如下：

```
[2022-06-05 23:07:24,056] [    INFO] - Adding <pad> to the vocabulary
[2022-06-05 23:07:24,057] [    INFO] - Adding <mask> to the vocabulary
[2022-06-05 23:07:24,057] [    INFO] - Adding <pad> to the vocabulary
[2022-06-05 23:07:24,058] [    INFO] - Adding <mask> to the vocabulary
[2022-06-05 23:07:24,058] [    INFO] - Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
text: You don't have to stay here. <sep> You can not leave., label_id: 0, prob: 0.8804811835289001, label: contradiction
```

推理成功，结果为contradiction（概率0.880），与模型本身一致。

