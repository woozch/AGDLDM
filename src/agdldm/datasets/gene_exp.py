import json
import pandas as pd
import numpy as np
import pyarrow
from typing import List

from datasets import Dataset, DatasetDict
from ..utils.parquet import find_parquet_files, get_parquet_file_columns


def gene_exp_generator(
    parquet_file_paths: List[str],
    gene_list_file: str | None = None,
    with_genes_string: bool = False,
    with_gene_exp_mask: bool = False,
):
    if gene_list_file is not None:
        with open(gene_list_file, "r") as file:
            columns = json.load(file)
        # check columns is a list of strings
        if not isinstance(columns, list) or not all(
            isinstance(col, str) for col in columns
        ):
            raise ValueError("columns must be a list of strings")
    else:
        columns = None

    for parquet_file_path in parquet_file_paths:
        try:
            df = pd.read_parquet(parquet_file_path, columns=columns)
        except pyarrow.lib.ArrowInvalid as e:
            valid_columns = get_parquet_file_columns(parquet_file_path)
            raise ValueError(
                f"Error with Invalid gene_list_file: {gene_list_file}:\n Invalid columns: {sorted(set(columns) - set(valid_columns))}"
            )

        for idx, row in df.iterrows():
            gene_exp_values = np.array(row.values)

            result = {
                "instance_id": str(idx),
                "gene_exp_values": gene_exp_values.tolist(),
            }
            if with_genes_string:
                result["genes_string"] = ",".join(df.columns.tolist())
            if with_gene_exp_mask:
                gene_exp_mask = np.where(
                    (
                        np.isnan(gene_exp_values)
                        | np.isinf(gene_exp_values)
                        | (gene_exp_values < 0)
                    ),
                    0,  # invalid value
                    1,  # valid value
                )
                result["gene_exp_mask"] = gene_exp_mask.tolist()
            yield result


def load_gene_exp_dataset(
    data_files: List[str],
    gene_list_file: str | None = None,
    split_ratio: float | None = 0.1,
    split_seed: int = 42,
    num_proc: int = 1,
    keep_in_memory: bool = False,
    cache_dir: str | None = None,
    with_genes_string: bool = False,
    with_gene_exp_mask: bool = False,
) -> DatasetDict:
    parquet_file_paths = find_parquet_files(data_files)
    gen_kwargs = {
        "parquet_file_paths": parquet_file_paths,
        "gene_list_file": gene_list_file,
        "with_genes_string": with_genes_string,
        "with_gene_exp_mask": with_gene_exp_mask,
    }
    dataset = Dataset.from_generator(
        gene_exp_generator,
        cache_dir=cache_dir,
        gen_kwargs=gen_kwargs,
        num_proc=num_proc,
        keep_in_memory=keep_in_memory,
    )

    if split_ratio is not None:
        dataset_dict = dataset.train_test_split(test_size=split_ratio, seed=split_seed)
    else:
        dataset_dict = DatasetDict({"train": dataset})
    return dataset_dict
