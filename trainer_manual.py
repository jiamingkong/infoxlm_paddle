# this script performs a manual training process on the XNLI dataset to enable more control

from args import parse_args
import logging
import os
import paddle
from experiments.xnli.xnli_utils import XNLI_Dataset, convert_examples, convert_example
from infoxlm_paddle import (
    InfoXLMTokenizer,
    InfoXLMModel,
    InfoXLMForSequenceClassification,
)
from utils import set_seed, get_writer, save_json, try_remove_old_ckpt, XNLI_LANGS
from argparse import Namespace
from paddle.amp import GradScaler, auto_cast
from paddle.metric import Accuracy
from paddle.optimizer import Adam, AdamW
import math
from tqdm.auto import tqdm


HERE = os.path.dirname(os.path.abspath(__file__))
PADDLE_WEIGHT = os.path.join(HERE, "model_checkpoints/converted_paddle")
SPM_MODEL = os.path.join(PADDLE_WEIGHT, "sentencepiece.bpe.model")

# print(model.roberta.embeddings.word_embeddings.weight.std())
# import pdb; pdb.set_trace()
logger = logging.getLogger(__name__)
args = parse_args()


tokenizer = InfoXLMTokenizer(
    sentencepiece_model_file=SPM_MODEL, do_lower_case=False, remove_space=True
)

if not args.quick_verify:
    base_model = InfoXLMModel.from_pretrained(PADDLE_WEIGHT)
    model = InfoXLMForSequenceClassification(base_model, num_classes=3, dropout=0.05)
    print(model.roberta.embeddings.word_embeddings.weight.std())
else:
    model = InfoXLMForSequenceClassification.from_pretrained(
        os.path.join(HERE, "model_checkpoints/finetuned_paddle")
    )


def as_tensor(t):
    return paddle.to_tensor(t)


@paddle.no_grad()
def evaluate(args, model, tokenizer, lang, metric, val=True):
    model.eval()
    metric.reset()
    splt = "dev" if val else "test"
    validation_loader = XNLI_Dataset(lang, split=splt)
    for batch in tqdm(
        validation_loader.get_batch_iterator(args.eval_batch_size), leave=False
    ):
        premises, hypotheses, labels = batch
        encoded_inputs = tokenizer(premises, hypotheses, padding=True)
        input_token_ids = paddle.to_tensor(encoded_inputs["input_ids"])
        logits = model(input_token_ids)
        correct = metric.compute(logits, as_tensor(labels))
        metric.update(correct)
    acc = metric.accumulate()
    model.train()
    return acc


def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "run.log"), mode="w", encoding="utf-8",
            )
        ],
    )
    logger.info("**********  Configuration Arguments **********")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("**************************************************")

    set_seed(args)

    writer = get_writer(args)

    params = Namespace(
        min_count=0,
        tokens_per_batch=-1,
        max_batch_size=0,
        group_by_size=False,
        max_len=args.max_length,
        eval_batch_size=args.eval_batch_size,
        train_batch_size=args.train_batch_size,
    )

    train_dataset = XNLI_Dataset(args.train_lang, "train",)
    eval_dataset = XNLI_Dataset(args.eval_lang, "test",)

    opt_cls = AdamW if args.optimizer.lower() == "adamw" else Adam
    optimizer = opt_cls(
        learning_rate=args.learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.classifier.parameters(),
    )
    model.roberta.stop_gradient=True

    metric = Accuracy()

    if args.fp16:
        scaler = GradScaler(init_loss_scaling=args.scale_loss)

    num_update_steps_per_epoch = args.sentences_per_epoch // args.train_batch_size
    if args.max_train_steps > 0:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )
    else:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

    logger.info("********** Running training **********")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous train batch size = {args.train_batch_size}")
    logger.info(f"  Instantaneous eval batch size = {args.eval_batch_size}")
    logger.info(f"  Total train batch size (w. accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    save_json(vars(args), os.path.join(args.output_dir, "args.json"))
    progress_bar = tqdm(range(args.max_train_steps))
    global_steps = 0
    tr_loss, logging_loss = 0.0, 0.0
    max_val_acc = 0.0

    ce_loss = paddle.nn.CrossEntropyLoss(reduction="mean")

    dls = train_dataset.get_batch_iterator(args.train_batch_size)

    for epoch in range(args.num_train_epochs):
        step = 0
        for batch in dls:
            model.train()
            model.roberta.eval()
            with auto_cast(
                args.fp16, custom_white_list=["layer_norm", "softmax", "gelu"]
            ):
                premises, hypotheses, labels = batch
                # tokenize

                encoded_inputs = tokenizer(premises, hypotheses, padding=True, max_length=params.max_len)
                input_token_ids = paddle.to_tensor(encoded_inputs["input_ids"])
                logits = model(input_token_ids)
                loss = (
                    ce_loss(logits, as_tensor(labels))
                    / args.gradient_accumulation_steps
                )
                tr_loss += loss.item()
            if args.fp16:
                scaled = scaler.scale(loss)
                scaled.backward()
            else:
                loss.backward()

            if step % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.clear_grad(set_to_zero=False)
                progress_bar.update(1)
                global_steps += 1

                if (
                    args.logging_steps > 0 and global_steps % args.logging_steps == 0
                ) or global_steps == args.max_train_steps:
                    writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_steps,
                    )
                    logger.info(
                        "global_steps {} loss: {:.8f}".format(
                            global_steps, (tr_loss - logging_loss) / args.logging_steps,
                        )
                    )
                    logging_loss = tr_loss

                if (
                    args.save_steps > 0 and global_steps % args.save_steps == 0
                ) or global_steps == args.max_train_steps:
                    logger.info("********** Running evaluating **********")
                    results_dict = {}
                    # val
                    val_avg_acc = 0
                    for lang in XNLI_LANGS:
                        val_acc = evaluate(
                            args, model, tokenizer, lang, metric, val=True
                        )
                        results_dict[f"val_{lang}_acc"] = val_acc
                        val_avg_acc += val_acc
                        logger.info(
                            f"##########  val_{lang}_acc {val_acc:.4f} ##########"
                        )

                    results_dict["val_avg_acc"] = val_avg_acc / 15
                    results_dict["val_other_acc"] = (
                        val_avg_acc - results_dict["val_en_acc"]
                    ) / 14
                    results_dict["val_gap_score"] = (
                        results_dict["val_en_acc"] - results_dict["val_other_acc"]
                    )

                    # test
                    test_avg_acc = 0
                    for lang in XNLI_LANGS:
                        # ds = XNLI_Dataset(lang, "test")
                        test_acc = evaluate(
                            args, model, tokenizer, lang, metric, val=False
                        )
                        results_dict[f"test_{lang}_acc"] = test_acc
                        test_avg_acc += test_acc
                        logger.info(
                            f"##########  test_{lang}_acc {test_acc:.4f} ##########"
                        )
                        # print(f"##########  test_{lang}_acc {test_acc} ##########")
                    results_dict["test_avg_acc"] = test_avg_acc / 15
                    results_dict["test_other_acc"] = (
                        test_avg_acc - results_dict["test_en_acc"]
                    ) / 14
                    results_dict["test_gap_score"] = (
                        results_dict["test_en_acc"] - results_dict["test_other_acc"]
                    )

                    for k, v in results_dict.items():
                        writer.add_scalar(f"eval/{k}", v, global_steps)
                        logger.info(f"  {k} = {v}")
                        print(f"  {k} = {v}")

                    val_avg_acc = results_dict["val_avg_acc"]
                    test_avg_acc = results_dict["test_avg_acc"]
                    val_gap_score = results_dict["val_gap_score"]
                    test_gap_score = results_dict["test_gap_score"]

                    if val_avg_acc >= max_val_acc:
                        max_val_acc = val_avg_acc
                        logger.info(
                            f"########## Step {global_steps} val_avg_acc {max_val_acc:.4f} test_avg_acc {test_avg_acc:.4f} ##########"
                        )
                        logger.info(
                            f"########## Step {global_steps} val_gap_score {-val_gap_score:.4f} test_gap_score {-test_gap_score:.4f}, the lower the better"
                        )

                    output_dir = os.path.join(
                        args.output_dir,
                        "ckpt",
                        f"step-{global_steps}-test_avg_acc-{test_avg_acc}-val_avg_acc-{val_avg_acc}",
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    model.save_pretrained(output_dir)
                    try_remove_old_ckpt(args.output_dir, topk=args.topk)

                    logger.info("********** Evaluating Done **********")

            if global_steps >= args.max_train_steps:
                return

            step += 1
            # if step >= args.sentences_per_epoch // args.train_batch_size:
            #     print(f"Epoch {epoch} resample!")
            #     dls = train_dataset.get_batch_iterator
            #     break


if __name__ == "__main__":
    args = parse_args()
    main(args)
