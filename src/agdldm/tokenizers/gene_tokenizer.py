import os
import os.path as osp

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import WordLevelTrainer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from ..utils.hgnc import (
    UNKNOWN_GENE,
)


def train_gene_tokenizer(data):
    tokenizer = Tokenizer(WordLevel(unk_token=UNKNOWN_GENE))  # type: ignore
    tokenizer.pre_tokenizer = Split(pattern=",", behavior="removed")  # type: ignore
    trainer = WordLevelTrainer(
        special_tokens=[UNKNOWN_GENE, "[CLS]", "[SEP]", "[PAD]", "[MASK]"]  # type: ignore
    )

    tokenizer.train_from_iterator(data, trainer=trainer)
    return tokenizer


def get_gene_tokenizer(processed_dir, tokenizer_file="gene_tokenizer.json", data=None):
    tokenizer_path = osp.join(processed_dir, tokenizer_file)
    if not osp.isfile(tokenizer_path):
        if not data:
            raise ValueError("data is required to train gene tokenizer")
        tokenizer = train_gene_tokenizer(data)
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer


def get_gene_tokenizer_fast(tokenizer_path: str) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, truncation=False)
    tokenizer.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "cls_token": "[CLS]",  # reserved
            "sep_token": "[SEP]",  # reserved
            "mask_token": "[MASK]",
        }
    )
    return tokenizer
