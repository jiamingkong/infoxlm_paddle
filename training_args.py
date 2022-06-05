from dataclasses import dataclass, field
from paddlenlp.trainer import Trainer, TrainingArguments, PdArgumentParser


@dataclass
class DataArguments:
    dataset: str = field(
        default=None, metadata={"help": "The name of the dataset to use."}
    )

    max_seq_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization."
        },
    )


parser = PdArgumentParser([TrainingArguments, DataArguments])
(training_args, data_args) = parser.parse_args_into_dataclasses()
