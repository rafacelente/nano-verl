from typing import Any
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

from vllm.tokenizers import TokenizerLike


def format_example(inp: dict[str, Any], tokenizer: TokenizerLike):
    input_ids = tokenizer.apply_chat_template(
        inp["messages"], tokenize=True, add_generation_prompt=True, return_dict=False
    )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "ground_truth": inp["ground_truth"]
    }


def get_tokenized_rlvr_math_dataset(tokenizer: TokenizerLike):
    dataset = load_dataset("allenai/RLVR-MATH", split="train")
    tokenized_format_example = lambda inp: format_example(inp, tokenizer)
    return dataset.map(tokenized_format_example, remove_columns=dataset.column_names)

def get_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn= lambda batch: {
            "input_ids": [x["input_ids"] for x in batch],
            "ground_truth": [x["ground_truth"] for x in batch],
            "attention_mask": [x["attention_mask"] for x in batch]
        }
    )