from __future__ import annotations

from typing import Any, TYPE_CHECKING

import rich.progress

if TYPE_CHECKING:
    from xtl.tui.console import ConsoleIO


class ProgressTask:

    def __init__(
            self,
            name: str,
            task_id: rich.progress.TaskID,
            progress_bar: ProgressBar | rich.progress.Progress,
    ):
        self._name = name
        self._task_id = task_id
        if isinstance(progress_bar, ProgressBar):
            self._progress: rich.progress.Progress = progress_bar._progress
        elif isinstance(progress_bar, rich.progress.Progress):
            self._progress: rich.progress.Progress = progress_bar
        else:
            raise TypeError(f'`progress_bar` type {type(progress_bar)} not supported.')

    def _is_alive(self) -> bool:
        return self._task_id in self._progress._tasks

    @property
    def _task(self) -> rich.progress.Task:
        if not self._is_alive or not (task := self._progress._tasks.get(self._task_id, None)):
            raise RuntimeError(f'Task {self._name!r} has been terminated.')
        return task

    def update(self, **kwargs) -> None:
        self._progress.update(
            task_id=self._task_id,
            **kwargs
        )

    def advance(self, value: int | float) -> None:
        self._progress.advance(
            task_id=self._task_id,
            advance=value
        )

    def __iadd__(self, value: int | float) -> ProgressTask:
        if not isinstance(value, (int, float)):
            raise TypeError(f'`value` type {type(value)} not supported for in-place addition.')
        self.advance(value)
        return self

    def remove(self) -> None:
        self._progress.remove_task(self._task_id)

    def start(self) -> None:
        self._progress.start_task(self._task_id)

    def stop(self) -> None:
        self._progress.stop_task(self._task_id)

    def reset(self) -> None:
        self._progress.reset(self._task_id)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    @property
    def task_id(self) -> rich.progress.TaskID:
        return self._task_id

    @property
    def description(self) -> str:
        return self._task.description

    @description.setter
    def description(self, value: str) -> None:
        self.update(description=value)

    @property
    def total(self) -> float | None:
        return self._task.total

    @total.setter
    def total(self, value: int | float | None) -> None:
        self.update(total=value)

    @property
    def completed(self) -> float:
        return self._task.completed

    @completed.setter
    def completed(self, value: int | float) -> None:
        self.update(completed=value)

    @property
    def started(self) -> bool:
        return self._task.started

    @property
    def remaining(self) -> float | None:
        return self._task.remaining

    @property
    def elapsed(self) -> float | None:
        return self._task.elapsed

    @property
    def finished(self) -> bool:
        return self._task.finished

    @property
    def percentage(self) -> float:
        return self._task.percentage

    @property
    def speed(self) -> float | None:
        return self._task.speed

    @property
    def time_remaining(self) -> float | None:
        return self._task.time_remaining



class ProgressBar:

    def __init__(
            self,
            *names,
            console: ConsoleIO,
            no_tasks: int | None = None,
            columns: tuple[rich.progress.ProgressColumn, ...] | None = None,
            **kwargs
    ) ->None:
        if no_tasks is not None and len(names) != 0 and no_tasks != len(names):
            raise ValueError(f'`no_tasks` is different from the number of `names` provided')
        elif len(names) == 0 and no_tasks is not None:
            names = (f'Task {i}' for i in range(no_tasks))

        self._console = console
        self._columns: tuple[rich.progress.ProgressColumn, ...] = columns if columns else self.get_default_columns()

        self._progress = rich.progress.Progress(
            *self._columns,
            **kwargs
        )

        self._tasks: dict[str, ProgressTask] = {}
        for name in names:
            self.add_task(name, description='')

    @classmethod
    def get_default_columns(cls) -> tuple[rich.progress.ProgressColumn, ...]:
        return rich.progress.Progress.get_default_columns()

    def add_task(
            self,
            name: str,
            *,
            description: str,
            total: int | float | None = None
    ) -> ProgressTask:
        task_id = self._progress.add_task(
            description=description,
            total=total,
        )
        task = ProgressTask(
            name=name,
            task_id=task_id,
            progress_bar=self,
        )
        self._tasks[name] = task
        return task

    def remove_task(self, name: str) -> None:
        task = self._tasks.pop(name, None)
        if task is not None:
            self._progress.remove_task(task.task_id)

    @property
    def tasks(self) -> dict[str, ProgressTask]:
        return self._tasks

    @property
    def columns(self) -> tuple[rich.progress.ProgressColumn, ...]:
        return self._columns

    def start_all(self):
        for task in self.tasks.values():
            task.start()

    def stop_all(self):
        for task in self.tasks.values():
            task.stop()

    def reset_all(self):
        for task in self.tasks.values():
            task.reset()

    def __rich__(self):
        return self._progress
