"""
Utility functions for handling parquet files and fingerprint data processing.
This module provides functions for reading, processing, and transforming fingerprint data stored in parquet format.
"""

import os
from typing import List, Optional, Union, Callable, Tuple, Generator
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from .parallel import process_with_progress_bar

__all__ = [
    "find_parquet_files",
    "find_chunked_parquet_directories",
    "get_index_columns",
    "get_parquet_file_columns",
    "get_parquet_file_row_count",
    "get_parquet_files_row_counts",
    "get_parquet_file_index",
    "get_parquet_files_index",
    "read_parquet_file",
    "read_parquet_files",
    "read_subset_parquet_file",
    "read_subset_parquet_files",
]


def find_parquet_files(
    sources: List[str], generator: bool = False, split_path: bool = False
) -> Union[
    List[str],
    List[Tuple[str, str, str]],
    Generator[str, None, None],
    Generator[Tuple[str, str, str], None, None],
]:
    """
    Recursively find all parquet files from given source paths.

    Args:
        sources (List[str]): List of file or directory paths to search for parquet files
        generator (bool, optional): If True, returns a generator instead of a list. Defaults to False.

    Returns:
        Union[List[str], Generator[str, None, None]]: Either a list or generator of absolute paths to found parquet files

    Raises:
        ValueError: If sources is not a list of strings or if any source path doesn't exist
    """
    # Validate input
    if not isinstance(sources, list) or not all(isinstance(p, str) for p in sources):
        raise ValueError("sources must be a list of strings")

    missing_paths = [p for p in sources if not os.path.exists(p)]
    if missing_paths:
        raise ValueError(f"The following paths do not exist: {missing_paths}")

    def _find_parquet_files():
        for source in sources:
            if os.path.isfile(source) and source.endswith(".parquet"):
                yield source
            elif os.path.isdir(source):
                for root, dirs, files in os.walk(source):
                    # Sort files for consistent file ordering within each root
                    files.sort()
                    for file in files:
                        if file.endswith(".parquet"):
                            if split_path:
                                yield (source, os.path.relpath(root, source), file)
                            else:
                                yield os.path.join(root, file)

    if generator:
        return _find_parquet_files()
    else:
        return list(_find_parquet_files())


def find_chunked_parquet_directories(
    output_dirs: List[str], generator: bool = False
) -> Union[List[str], Generator[str, None, None]]:
    """
    Get the list of parquet chunk directories from the output directories.
    """

    def _find_chunked_parquet_directories():
        for output_dir in output_dirs:
            for root, dirs, _ in os.walk(output_dir):
                for sub_dir in dirs:
                    paruet_dir = os.path.join(root, sub_dir)
                    if not len(
                        [f for f in os.listdir(paruet_dir) if f.endswith(".parquet")]
                    ):
                        continue
                    yield paruet_dir

    if generator:
        return _find_chunked_parquet_directories()
    else:
        return list(_find_chunked_parquet_directories())


def get_index_columns(file_path: str) -> List[str]:
    """
    Get the index columns from a parquet file.
    """
    with pq.ParquetFile(file_path) as pf:
        schema = pf.schema_arrow
        return schema.pandas_metadata.get("index_columns", [])


def get_parquet_file_columns(file_path: str) -> List[str]:
    """
    Get the columns names from a parquet file.
    """
    columns = pq.read_schema(file_path).names
    # remove index columns from columns if exists
    index_columns = get_index_columns(file_path)
    return [col for col in columns if col not in index_columns]


def get_parquet_file_row_count(file_path: str) -> int:
    """
    Get the number of rows in a parquet file without loading the entire data.
    Uses pyarrow which can read parquet files created by any parquet library.

    Args:
        file_path (str): Path to the parquet file

    Returns:
        int: Number of rows in the parquet file

    Raises:
        ImportError: If pyarrow is not installed
    """
    with pq.ParquetFile(file_path) as pf:
        num_rows = pf.metadata.num_rows
    return num_rows


def get_parquet_files_row_counts(
    parquet_files: List[str], progress_bar: bool = False, nproc: int = 1
) -> List[int]:
    """
    Calculate the number of rows in each parquet file.

    Args:
        parquet_files (List[str]): List of parquet file paths

    Returns:
        List[int]: List of row counts for each parquet file
    """
    if progress_bar:
        row_counts = process_with_progress_bar(
            get_parquet_file_row_count,
            parquet_files,
            nproc=nproc,
            description="Counting samples...",
            unit="files",
        )
        return row_counts

    else:
        return [get_parquet_file_row_count(file) for file in parquet_files]


def get_parquet_file_index(file_path: str) -> List[str | int]:
    """
    Get the index values from a parquet file using pyarrow.

    Args:
        file_path (str): Path to the parquet file

    Returns:
        List[str]: List of index values as strings
    """
    # Use pyarrow to read the index column if it exists
    with pq.ParquetFile(file_path) as pf:
        schema = pf.schema_arrow
        # get the index column from metadata pandas
        if getattr(schema, "pandas_metadata", None) is not None:
            index_columns = schema.pandas_metadata.get("index_columns", [])
        else:
            index_columns = []

        if index_columns:
            # Read only the index column
            table = pf.read(columns=index_columns)
            # Convert to list of strings
            index_values = table.column(0).to_pylist()
            return [str(val) for val in index_values]
        else:
            # Fallback to sequential indices if no index column found
            num_rows = pf.metadata.num_rows
            return list(range(num_rows))


def get_parquet_files_index(file_paths: List[str]) -> List[str]:
    """
    Get the index values from multiple parquet files.

    Args:
        file_paths (List[str]): List of paths to parquet files

    Returns:
        List[str]: List of index values from all files combined
    """
    all_indices = []
    for file_path in file_paths:
        file_indices = get_parquet_file_index(file_path)
        if isinstance(file_indices[0], int):
            file_indices = [len(all_indices) + i for i in file_indices]
        all_indices.extend(file_indices)
    return all_indices


def read_parquet_file(
    file_path: str,
    columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    engine: Optional[str] = "pyarrow",
) -> pd.DataFrame:
    """
    Read a parquet file into a pandas DataFrame.

    Args:
        file_path (str): Path to the parquet file
        columns (Optional[List[str]]): List of columns to read. If None, reads all columns
        nrows (Optional[int]): Number of rows to read. If None, reads all rows

    Returns:
        pd.DataFrame: DataFrame containing the parquet data
    """
    df = pd.read_parquet(file_path, columns=columns, engine=engine)
    if nrows is not None:
        df = df.head(nrows)
    return df


def read_parquet_files(
    file_paths: List[str],
    columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    engine: Optional[str] = "pyarrow",
) -> pd.DataFrame:
    """
    Read multiple parquet files into a single pandas DataFrame.

    Args:
        file_paths (List[str]): List of paths to parquet files
        columns (Optional[List[str]]): List of columns to read. If None, reads all columns
        nrows (Optional[int]): Number of rows to read from the combined data. If None, reads all rows.
            The rows are read in order from the first file to the last file.
        engine (Optional[str]): Engine to use for reading parquet files. Defaults to "pyarrow"

    Returns:
        pd.DataFrame: DataFrame containing the combined data from all parquet files

    Raises:
        ValueError: If file_paths is empty
    """
    if not file_paths:
        raise ValueError("file_paths cannot be empty")

    if nrows is None:
        # If nrows is None, read all files
        dfs = []
        for file_path in file_paths:
            df = read_parquet_file(
                file_path=file_path,
                columns=columns,
                nrows=None,
                engine=engine,
            )
            dfs.append(df)
        return pd.concat(dfs, axis=0, ignore_index=True)

    # Get row counts for each file
    row_counts = get_parquet_files_row_counts(file_paths)
    cumulative_rows = 0
    dfs = []

    # Read files until we have enough rows
    for file_path, row_count in zip(file_paths, row_counts):
        if cumulative_rows >= nrows:
            break

        # Calculate how many rows to read from this file
        remaining_rows = nrows - cumulative_rows
        rows_to_read = min(row_count, remaining_rows)

        df = read_parquet_file(
            file_path=file_path,
            columns=columns,
            nrows=rows_to_read,
            engine=engine,
        )
        dfs.append(df)
        cumulative_rows += row_count

    return pd.concat(dfs, axis=0, ignore_index=True)


def _get_parquet_index_mapping(row_counts: List[int]) -> List[tuple]:
    """
    Calculate the index range for each parquet file based on total row counts.

    Args:
        row_counts (List[int]): List of row counts for each parquet file

    Returns:
        List[tuple]: List of (start_index, end_index) tuples for each file
    """
    index_mapping = []
    start_idx = 0
    for count in row_counts:
        end_idx = start_idx + count
        index_mapping.append((start_idx, end_idx))
        start_idx = end_idx
    return index_mapping


def read_subset_parquet_file(
    parquet_file_path: str,
    columns: Optional[List[str]] = None,
    row_indices: Optional[List[int]] = None,
    engine: Optional[str] = "pyarrow",
) -> np.ndarray:
    """Load and transform specified columns from a parquet file into a numpy array.

    This function reads selected columns from a parquet file, optionally filters rows,
    and applies transformations to specified columns before returning a concatenated
    numpy array.

    Args:
        parquet_file_path (str): Path to the parquet file to read
        columns (List[str]): List of column names to load from the parquet file
        row_indices (Optional[List[int]]): Optional list of row indices to select. If None, all rows are selected
        transform_columns (Optional[List[str]]): List of column names that should be transformed.
            Defaults to LOG1P_TRANSFORM_FPS
        transform_fn (Optional[Callable]): Function to apply for transformation.
            Defaults to log1p_transformer

    Returns:
        np.ndarray: A contiguous numpy array containing the concatenated and transformed columns

    Note:
        - If transform_fn is None and transform_columns is empty, log1p_transformer will be used
        - The returned array is guaranteed to be contiguous in memory
    """

    Xs = []
    columns = columns if columns is not None else get_parquet_file_columns(parquet_file_path)
    for column_name in columns:
        data_df = read_parquet_file(
            parquet_file_path, columns=[column_name], engine=engine
        )
        if row_indices is not None:
            data_df = data_df.iloc[row_indices, :]
        Xs.append(data_df.to_numpy())
    return np.ascontiguousarray(np.concatenate(Xs, axis=0))


def read_subset_parquet_files(
    parquet_file_paths: List[str],
    columns: Optional[List[str]] = None,
    row_indices: Optional[Union[np.ndarray, List[int], Tuple[int, ...], slice]] = None,
    row_counts: Optional[List[int]] = None,
) -> np.ndarray:
    """
    read specified columns from multiple parquet files into a numpy array.

    If row_indices is None, reads all data from all files.
    If row_indices is provided, extracts only the specified indices while maintaining their order.

    Args:
        parquet_file_paths (List[str]): List of parquet file paths
        columns (List[str]): List of column names to read
        row_indices (Optional[Union[np.ndarray, List[int], Tuple[int, ...], slice]]):
            Indices to extract. Can be one of:
            - None: extracts all data
            - np.ndarray: array of indices
            - List[int]: list of indices
            - Tuple[int, ...]: tuple of indices
            - slice: slice object (e.g., slice(0, 10, 2))
            The indices should be within the range of total rows across all files.
        row_counts (Optional[List[int]]): List of row counts for each file. If None, will be calculated

    Returns:
        np.ndarray: Extracted data array with shape (n_samples, n_features)

    Raises:
        ValueError: If row_indices type is not supported or indices are out of range
    """
    # Pre-calculate total size to avoid reallocations
    n_features = (
        len(columns)
        if columns is not None
        else len(get_parquet_file_columns(parquet_file_paths[0]))
    )

    if row_counts is None:
        row_counts = get_parquet_files_row_counts(parquet_file_paths)

    total_rows = sum(row_counts)
    # Read all data if row_indices is None
    if row_indices is None:
        # Pre-allocate result array
        result = np.zeros((total_rows, n_features), dtype=np.float32)
        # Track current position in result array
        current_pos = 0

        # Process each file
        for file_path in parquet_file_paths:
            data = read_subset_parquet_file(
                file_path,
                columns=columns,
            )
            # Copy data to pre-allocated array
            result[current_pos : current_pos + len(data)] = data
            current_pos += len(data)

        return result

    # if row_indices is slice
    if isinstance(row_indices, (list, tuple, np.ndarray)):
        row_indices = np.array(row_indices)
    elif isinstance(row_indices, (slice)):
        # retrieve np.arrange
        start, stop, step = row_indices.indices(total_rows)
        row_indices = np.arange(start, stop, step)
    else:
        raise ValueError(f"Invalid row_indices type: {type(row_indices)}")

    # Validate row_indices range
    if np.any(row_indices < 0):
        # Convert negative indices to positive
        row_indices[row_indices < 0] = row_indices[row_indices < 0] + total_rows

    if np.any(row_indices < 0) or np.any(row_indices >= total_rows):
        raise IndexError(f"Indices are out of range: {row_indices}")

    # Calculate index ranges for each parquet file
    index_mapping = _get_parquet_index_mapping(row_counts)
    starts = np.array([start for start, _ in index_mapping])
    ends = np.array([end for _, end in index_mapping])

    # Pre-allocate result array with the correct shape
    result = np.zeros((len(row_indices), n_features), dtype=np.float32)

    if len(row_indices) > len(index_mapping):
        # Process each file
        for file_idx, (start_idx, end_idx) in enumerate(index_mapping):
            # Find indices that belong to current parquet file
            mask = (row_indices >= start_idx) & (row_indices < end_idx)
            file_indices = row_indices[mask]
            if len(file_indices) == 0:
                continue

            # Convert to relative indices within the file
            relative_indices = file_indices - start_idx

            # Extract data from current file
            data = read_subset_parquet_file(
                parquet_file_paths[file_idx],
                columns=columns,
                row_indices=relative_indices.tolist(),
            )

            # Insert data at original positions using boolean indexing
            result[mask] = data
    else:
        # Process only the files that contain requested indices
        # Group indices by file to minimize file reads
        file_indices_map = {}
        original_indices_map = {}  # Store original indices for each file

        # Vectorized binary search for all indices at once
        file_indices = np.searchsorted(starts, row_indices, side="right") - 1

        # Create masks for valid indices
        valid_mask = (file_indices >= 0) & (row_indices < ends[file_indices])

        # Group indices by file
        for i, (file_idx, is_valid) in enumerate(zip(file_indices, valid_mask)):
            if is_valid:
                if file_idx not in file_indices_map:
                    file_indices_map[file_idx] = []
                    original_indices_map[file_idx] = []
                file_indices_map[file_idx].append(row_indices[i] - starts[file_idx])
                original_indices_map[file_idx].append(i)

        # Process each file that contains requested indices
        for file_idx, relative_indices in file_indices_map.items():
            # Extract data from current file
            data = read_subset_parquet_file(
                parquet_file_paths[file_idx],
                columns=columns,
                row_indices=relative_indices,
            )

            # Assign data to result array maintaining original order
            result[original_indices_map[file_idx]] = data

    return result
