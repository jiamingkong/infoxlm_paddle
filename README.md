# infoxlm_paddle

使用[飞桨PaddlePaddle-源于产业实践的开源深度学习平台](https://www.paddlepaddle.org.cn/) 深度学习框架复现InfoXLM 论文的实验结果。复现概要：

- Sentence Retrieval 结果与原始论文一致；
- XNLI 推理与原始论文结果接近。



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

### 3.1 XNLI 任务

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

Non-English 平均分是0.751，English的分数是0.854，两者相差10.3%，与原文一致。

**讨论**

我们认为两个关键的差异在于：

- 原始论文微调使用的batch_size = 256，而我们复现的版本因为硬件限制，实际有效的batch_size是32
- 原始论文每5000步即评估一次val set精度，我们只在每个epoch结束的时候评估了。

更小的batch_size 影响了一些性能，我们注意到有些epoch中，中文的精度可以达到77.0%，比起实际提交的微调版本高出一个点，但是这个epoch在其他语言上的表现平均不如提交的版本。可见更小的batch_size 导致的性能波动还是可观的。

### 3.2 Tatoeba Sentence Retrieval任务

该任务不需要训练，只需要使用导出的原始预训练权重即可完成。原文中给出了参考的Tatoeba数据集使用的论文：[[1812.10464\] Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond (arxiv.org)](https://arxiv.org/abs/1812.10464)，我们阅读论文后找到了论文开源的数据集合和评测方法，来源：[facebookresearch/LASER: Language-Agnostic SEntence Representations (github.com)](https://github.com/facebookresearch/LASER)，并且将评测过程使用paddle重现了；

Sentence Retrieval任务是评测跨语言的同样含义的句子（例如一对互译的句子），经过预训练模型编码后，其表征是否有足够的相似度。InfoXLM论文使用了Tatoeba里与XNLI的14个语言 -- 英文互译的数据集；每个语言和英语的互译句子有1000句。针对一个语言评测时，我们执行以下操作：

- 将这1000句话的英文和外语版本（共2000句）encode，取出InfoXLM第7层transformer输出的向量，并沿token的方向取平均，得到1 \* 768的向量，并作L2标准化；
- 对En->X 计算句子取回的准确率：
  - 针对每个英语的句子，计算该句子的向量与X语言1000句的向量的余弦相似度，取出余弦相似度最高的句子；
  - 计算取出的句子确实是对应译文的平均概率，为准确率。
- 对X-> En 计算句子取回的准确率：
  - 与上述过程类似，针对每个X语言句子计算英语1000句中的相似度

我们已经在git中附带了要用到的Tatoeba数据，位于`experiments/sentence_retrieval/datasets`。如果您完成了原始权重的转换，您可以通过以下命令行复现上述实验的过程：

```bash
python experiments/sentence_retrieval/run_experiments.py
```

执行后会输出log.txt 和 sentence_retrieval_results.csv，结果如下，表格中所有指标与论文报告（table-2）完全一致。

| X           | en->x     | x->en     |
| ----------- | --------- | --------- |
| ar          | 0.686     | 0.591     |
| bg          | 0.787     | 0.787     |
| de          | 0.951     | 0.939     |
| el          | 0.726     | 0.622     |
| es          | 0.872     | 0.882     |
| fr          | 0.84      | 0.794     |
| hi          | 0.883     | 0.871     |
| ru          | 0.857     | 0.838     |
| sw          | 0.408     | 0.395     |
| th          | 0.912     | 0.850     |
| tr          | 0.847     | 0.832     |
| ur          | 0.733     | 0.73      |
| vi          | 0.92      | 0.896     |
| zh          | 0.864     | 0.864     |
| **Average** | **0.806** | **0.778** |



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

## 5. Serving 部署

### 5.1 安装Serving依赖

详情请参考[Serving 安装](https://github.com/PaddlePaddle/Serving/blob/v0.9.0/doc/Install_CN.md)。在这里我们推荐使用linux docker的方式来安装，原生版本无论在windows还是unix环境中不可控因素较难排查。我们提供的Git代码经测试可以在windows + linux docker cpu only 的版本中成功部署。

**为了方便，启动的时候可以挂载上当前的目录**
```
# 启动 CPU Docker
docker pull registry.baidubce.com/paddlepaddle/serving:0.9.0-devel
docker run -p 9292:9292 --name test_cpu -v ${PWD}:/home/infoxlm_paddle -dit registry.baidubce.com/paddlepaddle/serving:0.9.0-devel bash
docker exec -it test_cpu bash
git clone https://github.com/PaddlePaddle/Serving
```
**并完成页面内其他的步骤，安装依赖项目**

### 5.2 部署推理服务

使用4.1章节中导出的模型，部署推理服务，如下：

```bash