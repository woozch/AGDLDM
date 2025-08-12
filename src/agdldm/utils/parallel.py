"""Parallel processing utilities for handling collections of tasks.

This module provides utilities for processing collections of tasks either sequentially
or in parallel using joblib. It includes support for progress bars and flexible argument
handling.

Key Features:
- Sequential and parallel processing of task collections
- Progress bar integration using rich
- Flexible argument handling (positional, keyword, mixed)
- Memory-efficient processing of generators
- Automatic fallback to sequential processing when appropriate

Example:
    >>> def process_item(x, factor=1):
    ...     return x * factor
    >>> items = [1, 2, 3, 4, 5]
    >>> results = process_with_progress_bar(
    ...     process_item,
    ...     items,
    ...     nproc=2,
    ...     process_func_kwargs={'factor': 2},
    ...     description="Processing items"
    ... )
"""

import os
from typing import (
    List,
    Any,
    Callable,
    Tuple,
    Dict,
    Optional,
    Iterable,
    Sequence,
    Union,
    TypeVar,
)
from rich.progress import Progress
from joblib import Parallel, delayed
from .progressbar import create_rich_progress


__all__ = [
    "process_sequential",
    "process_parallel",
    "process_parallel_or_sequential",
    "process_with_progress_bar",
]


T = TypeVar("T")
ArgType = Union[
    Tuple[Any, ...],  # Tuple of positional arguments
    Tuple[Any, Dict[str, Any]],  # Single value with keyword arguments
    Tuple[
        Tuple[Any, ...], Dict[str, Any]
    ],  # Tuple of positional arguments with keyword arguments
    Dict[str, Any],  # Dictionary of keyword arguments
    Any,  # Single value
]


def process_args(
    process_func: Callable[..., T],
    args: ArgType,
    process_func_kwargs: Dict[str, Any],
) -> T:
    """Process arguments with a function, handling different argument formats.

    This function provides a unified interface for processing arguments regardless of their format.
    It supports multiple ways of passing arguments to the process function:
    - Tuple of positional arguments: (arg1, arg2, ...)
    - Single value with keyword arguments: (value, {'kwarg1': val1, ...})
    - Tuple of positional arguments with keyword arguments: ((arg1, arg2, ...), {'kwarg1': val1, ...})
    - Dictionary of keyword arguments: {'arg1': val1, 'arg2': val2, ...}
    - Single value: value

    The function merges any additional keyword arguments (process_func_kwargs) with the provided arguments.

    Parameters
    ----------
    process_func : Callable[..., T]
        Function to process the arguments
    args : ArgType
        Arguments to process, can be in various formats as described above
    process_func_kwargs : Dict[str, Any]
        Additional keyword arguments to pass to process_func. These will be merged with
        any keyword arguments provided in args.

    Returns
    -------
    T
        Result of processing the arguments

    Examples
    --------
    >>> def add(a, b, c=0): return a + b + c
    >>> # Positional arguments
    >>> process_args(add, (1, 2), {'c': 3})
    6
    >>> # Single value with keyword arguments
    >>> process_args(add, (1, {'c': 3}), {'b': 2})
    6
    >>> # Tuple with keyword arguments
    >>> process_args(add, ((1, 2), {'c': 3}), {})
    6
    >>> # Dictionary of keyword arguments
    >>> process_args(add, {'a': 1, 'b': 2, 'c': 3}, {})
    6
    >>> # Single value
    >>> process_args(add, 1, {'b': 2, 'c': 3})
    6
    """
    if isinstance(args, tuple):
        if len(args) == 2:
            if isinstance(args[1], dict):
                if isinstance(args[0], tuple):
                    return process_func(*args[0], **(process_func_kwargs | args[1]))
                else:
                    return process_func(args[0], **(process_func_kwargs | args[1]))

        return process_func(*args, **process_func_kwargs)

    elif isinstance(args, dict):
        return process_func(**(process_func_kwargs | args))
    return process_func(args, **process_func_kwargs)


def process_sequential(
    process_func: Callable[..., T],
    args_collection: Iterable[ArgType],
    process_func_kwargs: Dict[str, Any] = {},
    progress: Optional[Progress] = None,
    task: Optional[int] = None,
) -> List[T]:
    """Process a collection of arguments sequentially.

    Parameters
    ----------
    process_func : Callable[..., T]
        Function to process each set of arguments
    args_collection : Iterable[ArgType]
        Collection of arguments to process
    process_func_kwargs : Dict[str, Any], optional
        Additional keyword arguments to pass to process_func, by default {}
    progress : Optional[Progress], optional
        Progress bar to update, by default None
    task : Optional[int], optional
        Task ID for progress bar, by default None

    Returns
    -------
    List[T]
        List of results in the same order as args_collection

    Examples
    --------
    >>> def square(x): return x * x
    >>> results = process_sequential(square, [1, 2, 3])
    >>> print(results)
    [1, 4, 9]
    """
    results = []
    for args in args_collection:
        result = process_args(process_func, args, process_func_kwargs)
        if progress is not None and task is not None:
            progress.update(task, advance=1)
        results.append(result)
    return results


def process_parallel(
    process_func: Callable[..., T],
    args_collection: Iterable[ArgType],
    nproc: Optional[int] = None,
    process_func_kwargs: Optional[Dict[str, Any]] = None,
    backend: str = "loky",
    pre_dispatch: str = "1*n_jobs",
    buffer_size_factor: int = 4,
    progress: Optional[Progress] = None,
    task: Optional[int] = None,
) -> List[T]:
    """Process a collection of arguments in parallel using joblib.

    This function processes a collection of arguments in parallel using joblib's Parallel
    and delayed functions. It handles both sequences and generators efficiently, using
    buffering for generators to prevent memory issues.

    Parameters
    ----------
    process_func : Callable[..., T]
        Function to process each set of arguments
    args_collection : Iterable[ArgType]
        Collection of arguments to process. Can be a sequence (list, tuple) or a generator.
    nproc : Optional[int], optional
        Number of processes to use:
        - None or Negative: Uses all available CPUs
        - 0 or 1: Falls back to sequential processing
        - Positive integer: Uses specified number of processes
    process_func_kwargs : Optional[Dict[str, Any]], optional
        Additional keyword arguments to pass to process_func
    backend : str, optional
        Backend to use for parallel processing. Options include:
        - "loky": Process-based backend (default)
        - "threading": Thread-based backend
        - "multiprocessing": Legacy process-based backend
    pre_dispatch : str, optional
        Pre-dispatch parameter for joblib.Parallel. Controls how many tasks are
        dispatched at once. Default is "1*n_jobs" for memory-efficient processing.
    buffer_size_factor : int, optional
        For generator inputs, controls the size of the processing buffer.
        Buffer size = nproc * buffer_size_factor. Default is 4.
    progress : Optional[Progress], optional
        Rich progress bar to update during processing
    task : Optional[int], optional
        Task ID for the progress bar

    Returns
    -------
    List[T]
        List of results in the same order as args_collection

    Notes
    -----
    - For generator inputs, the function uses buffering to prevent memory issues
    - Progress updates are handled efficiently to minimize overhead
    - The function automatically falls back to sequential processing if nproc is 0 or 1

    Examples
    --------
    >>> def square(x): return x * x
    >>> # Process a list in parallel
    >>> results = process_parallel(square, [1, 2, 3], nproc=2)
    >>> print(results)
    [1, 4, 9]
    >>> # Process a generator with progress bar
    >>> from rich.progress import Progress
    >>> with Progress() as progress:
    ...     task = progress.add_task("Processing...", total=3)
    ...     results = process_parallel(
    ...         square,
    ...         (x for x in [1, 2, 3]),
    ...         nproc=2,
    ...         progress=progress,
    ...         task=task
    ...     )
    """
    if process_func_kwargs is None:
        process_func_kwargs = {}

    if nproc is None or nproc < 0:
        nproc = os.cpu_count()
    if nproc == 0 or nproc == 1:
        return process_sequential(
            process_func, args_collection, process_func_kwargs, progress, task
        )

    # Create a parallel executor
    results = []
    with Parallel(n_jobs=nproc, pre_dispatch=pre_dispatch, backend=backend) as parallel:
        if isinstance(args_collection, Sequence):
            # args_collection is fetched already, no memory issues
            results.extend(
                parallel(
                    [
                        delayed(process_args)(process_func, args, process_func_kwargs),
                        (
                            progress.update(task, advance=1)
                            if progress is not None and task is not None
                            else None
                        ),
                    ][0]
                    for args in args_collection
                )
            )
        else:
            # args_collection is a generator, need to process items in batches to avoid memory issues
            buffer_size = nproc * buffer_size_factor
            buffer = []
            for args in args_collection:
                buffer.append(
                    delayed(process_args)(process_func, args, process_func_kwargs)
                )

                if len(buffer) >= buffer_size:
                    results.extend(
                        parallel(
                            [
                                [
                                    delay_obj,
                                    (
                                        progress.update(task, advance=1)
                                        if progress is not None and task is not None
                                        else None
                                    ),
                                ][0]
                                for delay_obj in buffer
                            ]
                        )
                    )
                    buffer.clear()

            # Process remaining items
            if buffer:
                results.extend(
                    parallel(
                        [
                            [
                                delay_obj,
                                (
                                    progress.update(task, advance=1)
                                    if progress is not None and task is not None
                                    else None
                                ),
                            ][0]
                            for delay_obj in buffer
                        ]
                    )
                )
                buffer.clear()

    return results


def process_parallel_or_sequential(
    process_func: Callable[..., T],
    args_collection: Iterable[ArgType],
    nproc: int,
    backend: str = "loky",
    process_func_kwargs: Dict[str, Any] = {},
    pre_dispatch: str = "1*n_jobs",
    progress: Optional[Progress] = None,
    task: Optional[int] = None,
) -> List[T]:
    """Process a collection of arguments either sequentially or in parallel.

    Parameters
    ----------
    process_func : Callable[..., T]
        Function to process each set of arguments
    args_collection : Iterable[ArgType]
        Collection of arguments to process
    nproc : int
        Number of processes to use (1 for sequential)
    process_func_kwargs : Dict[str, Any], optional
        Additional keyword arguments to pass to process_func, by default {}
    pre_dispatch : str, optional
        Pre-dispatch parameter for joblib.Parallel, by default "1*n_jobs"
    progress : Optional[Progress], optional
        Progress bar to update, by default None
    task : Optional[int], optional
        Task ID for progress bar, by default None

    Returns
    -------
    List[T]
        List of results in the same order as args_collection

    Examples
    --------
    >>> def square(x): return x * x
    >>> # Sequential processing
    >>> results = process_parallel_or_sequential(square, [1, 2, 3], nproc=1)
    >>> print(results)
    [1, 4, 9]
    >>> # Parallel processing
    >>> results = process_parallel_or_sequential(square, [1, 2, 3], nproc=2)
    >>> print(results)
    [1, 4, 9]
    """
    if nproc > 1:
        return process_parallel(
            process_func,
            args_collection,
            nproc,
            process_func_kwargs=process_func_kwargs,
            backend=backend,
            pre_dispatch=pre_dispatch,
            progress=progress,
            task=task,
        )
    return process_sequential(
        process_func,
        args_collection,
        process_func_kwargs=process_func_kwargs,
        progress=progress,
        task=task,
    )


def process_with_progress_bar(
    process_func: Callable[..., T],
    args_collection: Iterable[ArgType],
    nproc: int = 1,
    backend: str = "loky",
    pre_dispatch: str = "1*n_jobs",
    process_func_kwargs: Dict[str, Any] = {},
    description: str = "",
    unit: str = "it",
) -> List[T]:
    """Process a collection of arguments with a progress bar.

    This is a convenience function that combines parallel processing with a progress bar.
    It automatically creates and manages a Rich progress bar while processing the arguments.

    Parameters
    ----------
    process_func : Callable[..., T]
        Function to process each set of arguments
    args_collection : Iterable[ArgType]
        Collection of arguments to process
    nproc : int, optional
        Number of processes to use. Default is 1 (sequential processing).
    backend : str, optional
        Backend to use for parallel processing. Default is "loky".
    pre_dispatch : str, optional
        Pre-dispatch parameter for joblib.Parallel. Default is "1*n_jobs".
    process_func_kwargs : Dict[str, Any], optional
        Additional keyword arguments to pass to process_func
    description : str, optional
        Description text to show in progress bar
    unit : str, optional
        Unit label to show in progress bar (e.g., "items", "files", "tasks")

    Returns
    -------
    List[T]
        List of results in the same order as args_collection

    Raises
    ------
    ValueError
        If args_collection is not iterable

    Notes
    -----
    - The progress bar automatically determines the total number of items if possible
    - For generators or iterables without length, the progress bar will show indeterminate progress
    - The function handles both sequential and parallel processing transparently

    Examples
    --------
    >>> def process_file(filename):
    ...     return f"Processed {filename}"
    >>> # Process a list of files with progress bar
    >>> files = ["file1.txt", "file2.txt", "file3.txt"]
    >>> results = process_with_progress_bar(
    ...     process_file,
    ...     files,
    ...     nproc=2,
    ...     description="Processing files",
    ...     unit="files"
    ... )
    >>> # Process a generator with indeterminate progress
    >>> def generate_items():
    ...     for i in range(100):
    ...         yield i
    >>> results = process_with_progress_bar(
    ...     lambda x: x * 2,
    ...     generate_items(),
    ...     nproc=4,
    ...     description="Processing items",
    ...     unit="items"
    ... )
    """
    if not isinstance(args_collection, Iterable):
        raise ValueError("args_collection must be an iterable")

    with create_rich_progress() as progress:
        try:
            total = len(args_collection)
        except TypeError:
            total = None

        task = progress.add_task(description, total=total, unit=unit)

        return process_parallel_or_sequential(
            process_func,
            args_collection,
            nproc,
            process_func_kwargs=process_func_kwargs,
            backend=backend,
            pre_dispatch=pre_dispatch,
            progress=progress,
            task=task,
        )
