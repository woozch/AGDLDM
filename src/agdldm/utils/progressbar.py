from typing import Optional
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    Task,
)
from rich.console import RenderableType


__all__ = [
    "create_rich_progress",
    "TasksPerSecondColumn",
]


class TasksPerSecondColumn(TextColumn):
    """Custom progress column that displays the number of tasks processed per second.

    This column shows the processing speed in tasks per second, with a configurable
    unit label. The speed is calculated based on the task's progress and elapsed time.

    Parameters
    ----------
    unit : Optional[str], optional
        Unit label to display after the speed value, by default "it"
    color : str, optional
        Color to use for the speed display, by default "red"
    """

    def __init__(self, *args, unit: Optional[str] = "it", color: str = "red", **kwargs):
        super().__init__("N/A", *args, **kwargs)
        self.unit = unit
        self.text_format = "N/A"
        self.color = color

    def render(self, task: Task) -> RenderableType:
        """Render the tasks per second column.

        Parameters
        ----------
        task : Task
            The task to render the column for

        Returns
        -------
        RenderableType
            The rendered column content
        """
        if task.remaining is None or task.remaining == 0:
            return (
                f"[{self.color}]"
                + "N/A"
                + f" {task.fields.get('unit', self.unit)}/s"
                + f"[/{self.color}]"
            )

        tasks_per_second = 0 if task.speed is None else task.speed
        self.tasks_per_second = tasks_per_second
        self.text_format = (
            f"[{self.color}] {tasks_per_second:.2f}"
            + f" {task.fields.get('unit', self.unit)}/s"
            + f"[/{self.color}]"
        )
        return super().render(task)


def create_rich_progress(transient: bool = True) -> Progress:
    """Create a custom rich progress bar with multiple columns.

    Returns
    -------
    Progress
        A rich Progress object with the following columns:
        - SpinnerColumn: Animated spinner
        - TextColumn: Task description
        - TaskProgressColumn: Percentage complete (hidden if no total)
        - BarColumn: Progress bar (hidden if no total)
        - TextColumn: Completed count
        - TextColumn: Total count (hidden if no total)
        - TimeElapsedColumn: Time elapsed
        - TimeRemainingColumn: Estimated time remaining (hidden if no total)
        - TasksPerSecondColumn: Processing speed (hidden if no total)

    Notes
    -----
    The progress bar is configured with transient=True, meaning it will be removed
    when complete. The TasksPerSecondColumn is a custom column that displays the
    processing speed in tasks per second.

    Examples
    --------
    >>> with create_rich_progress() as progress:
    ...     task = progress.add_task("Processing...", total=100)
    ...     for i in range(100):
    ...         progress.update(task, advance=1)
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TaskProgressColumn(),
        BarColumn(bar_width=None),
        TextColumn("[green]{task.completed}[/green]"),
        TextColumn("[green]/{task.total}[/green]"),
        "[",
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        TextColumn(","),
        TasksPerSecondColumn(),
        "]",
        transient=transient,
    )

    return progress
