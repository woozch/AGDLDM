import os
import sys
import gc
import numpy as np
import pytz
import datetime
import logging
import hashlib
from contextlib import contextmanager
from typing import Any, List, Generator
from time import perf_counter
from typing import Optional


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    file_level: Optional[int] = None,
) -> logging.Logger:
    """Get a logger with console and file handlers.

    Parameters
    ----------
    name : str
        Name of the logger
    log_file : Optional[str], optional
        Path to log file, by default None
    level : int, optional
        Console logging level, by default logging.INFO
    file_level : Optional[int], optional
        File logging level, by default same as console level

    Returns
    -------
    logging.Logger
        Logger with console and optional file handlers

    Examples
    --------
    >>> logger = get_logger("my_logger", "output.log")
    >>> logger.info("This will be logged to console and file")
    >>> logger = get_logger("console_only")
    >>> logger.info("This will only be logged to console")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level if file_level is not None else level)
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def add_logger_file_handler(
    logger: logging.Logger, log_file: str, level: int = logging.INFO
):
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    return logger


def to_list(obj: Any) -> List[Any]:
    """
    Convert various types of objects to a list.

    Args:
        obj: Object to convert. Can be a list, numpy array, string, or any iterable.

    Returns:
        List[Any]: Converted list object.

    Examples:
        >>> to_list([1, 2, 3])
        [1, 2, 3]
        >>> to_list(np.array([1, 2, 3]))
        [1, 2, 3]
        >>> to_list("hello")
        ['hello']
        >>> to_list(42)
        [42]
    """
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, str):
        return [obj]
    elif not hasattr(obj, "__iter__"):
        return [obj]
    else:
        return list(obj)


def split_into_chunks(
    lst: List[Any], chunk_size: int
) -> Generator[List[Any], None, None]:
    """
    Split a list into chunks of specified size.

    Args:
        lst (List[Any]): List to be split into chunks.
        chunk_size (int): Size of each chunk.

    Yields:
        List[Any]: Chunks of the original list.

    Examples:
        >>> list(split_into_chunks([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
        >>> list(split_into_chunks(['a', 'b', 'c'], 1))
        [['a'], ['b'], ['c']]
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def get_formatted_timestamp(
    timezone: str = "Asia/Seoul", format_str: str = "%y%m%d_%H%M"
) -> str:
    """
    Get current timestamp in the specified timezone with a custom format.

    Args:
        timezone (str, optional): Timezone string (e.g., 'Asia/Seoul', 'UTC', 'America/New_York').
            Defaults to 'Asia/Seoul'.
        format_str (str, optional): Datetime format string. Defaults to "%y%m%d_%H%M".
            Common format codes:
            - %y: Year without century (00-99)
            - %m: Month (01-12)
            - %d: Day (01-31)
            - %H: Hour (00-23)
            - %M: Minute (00-59)

    Returns:
        str: A formatted timestamp string according to the specified format.

    Examples:
        >>> get_formatted_timestamp()  # Default Seoul timezone
        '240315_1430'  # Example output for March 15, 2024, 14:30 KST
        >>> get_formatted_timestamp('UTC')
        '240315_0530'  # Same time in UTC
        >>> get_formatted_timestamp(format_str='%Y-%m-%d %H:%M:%S')
        '2024-03-15 14:30:00'

    Raises:
        pytz.exceptions.UnknownTimeZoneError: If the specified timezone is invalid.
    """
    tz = pytz.timezone(timezone)
    current_date = datetime.datetime.now(tz)
    return current_date.strftime(format_str)


@contextmanager
def memory_management() -> Generator[None, None, None]:
    """
    Context manager for memory management that forces garbage collection after the context.

    This context manager ensures that garbage collection is performed after the
    execution of the code block, which can be useful for memory-intensive operations.

    Yields:
        None

    Examples:
        >>> with memory_management():
        ...     # Perform memory-intensive operations
        ...     large_data = [i for i in range(1000000)]
        ...     # Garbage collection will be performed automatically after this block
    """
    try:
        yield
    finally:
        gc.collect()


@contextmanager
def performance_monitor(operation_name: str, logger: Optional[logging.Logger] = None):
    start = perf_counter()
    try:
        yield
    finally:
        duration = perf_counter() - start
        if logger is not None:
            logger.info(f"{operation_name} took {duration:.2f} seconds")
        else:
            print(f"{operation_name} took {duration:.2f} seconds")


def save_execution_command(output_dir: str, logger: Optional[logging.Logger] = None):
    # get execution command as a string
    execution_command = " ".join(sys.orig_argv)
    if logger is not None:
        logger.info(f"Execution command: {execution_command}")
    else:
        print(f"Execution command: {execution_command}")
    # save execution command to a file
    with open(os.path.join(output_dir, "execution_command.sh"), "w") as f:
        f.write(execution_command)


def get_file_hash(file_path: str, hash_length: int = 8) -> str:
    """
    Generate a short hash from a file path.

    Args:
        file_path (str): Path to the file
        hash_length (int, optional): Length of the hash to return. Defaults to 8.

    Returns:
        str: A short hash string of the file path

    Examples:
        >>> get_file_hash("/path/to/file.parquet")
        'a1b2c3d4'
        >>> get_file_hash("/path/to/file.parquet", hash_length=4)
        'a1b2'
    """
    return hashlib.md5(file_path.encode()).hexdigest()[:hash_length]
