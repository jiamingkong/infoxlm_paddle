import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune an XLM model on xnli")
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--sentences_per_epoch", type=int, default=10000, help="sentences_per_epoch.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=0,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="xnli_outputs",
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="optimizer.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--writer_type",
        type=str,
        default="visualdl",
        choices=["tensorboard", "visualdl"],
        help="writer_type",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=50, help="logging_steps.",
    )
    parser.add_argument(
        "--save_steps", type=int, default=2500, help="save_steps.",
    )
    parser.add_argument(
        "--topk", type=int, default=3, help="save_topk.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="num_workers",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )

    parser.add_argument(
        "--fp16", action="store_true", help="Enable mixed precision training."
    )
    parser.add_argument(
        "--scale_loss",
        type=float,
        default=2 ** 10,
        help="The value of scale_loss for fp16.",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="data_caches", help="cache_dir.",
    )
    parser.add_argument(
        "--train_lang", type=str, default="en", help="The training language.",
    )
    parser.add_argument(
        "--eval_lang", type=str, default="zh", help="The eval language",
    )
    parser.add_argument(
        "--quick_verify",
        action="store_true",
        help="Load the finetuned checkpoint, train 10 steps and eval",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        args.log_dir = os.path.join(args.output_dir, "logs")
        os.makedirs(args.log_dir, exist_ok=True)

    return args
