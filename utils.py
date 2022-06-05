import json
import logging
import pickle
import random
import shutil
from pathlib import Path

import numpy as np
import paddle

logger = logging.getLogger(__name__)

XNLI_LANGS = [
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def try_remove_old_ckpt(output_dir, topk=5):
    if topk <= 0:
        return
    p = Path(output_dir) / "ckpt"
    ckpts = sorted(
        p.glob("step-*"), key=lambda x: float(x.name.split("-")[-1]), reverse=True
    )
    if len(ckpts) > topk:
        shutil.rmtree(ckpts[-1])
        logger.info(f"remove old ckpt: {ckpts[-1]}")


def get_writer(args):
    if args.writer_type == "visualdl":
        from visualdl import LogWriter

        writer = LogWriter(logdir=args.log_dir)
    elif args.writer_type == "tensorboard":
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(logdir=args.log_dir)
    else:
        raise ValueError("writer_type must be in ['visualdl', 'tensorboard']")
    return writer


def save_json(data, file_name):
    with open(file_name, "w", encoding="utf-8") as w:
        w.write(json.dumps(data, ensure_ascii=False, indent=4) + "\n")


def save_pickle(data, file_path):
    with open(str(file_path), "wb") as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    with open(str(input_file), "rb") as f:
        data = pickle.load(f)
    return data
