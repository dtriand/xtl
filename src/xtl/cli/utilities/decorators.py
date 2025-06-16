import asyncio
from functools import wraps
import inspect
import typing
from typing import Any, Callable, TypeVar
from typing_extensions import Concatenate, ParamSpec

from xtl import settings, Logger
import xtl.cli.utilities.common

logger = Logger(__name__)


def typer_async(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


RHook = TypeVar('RHook')
ParamsHook = ParamSpec('ParamsHook')
RSource = TypeVar('RSource')
ParamsSource = ParamSpec('ParamsSource')


class BadHookError(TypeError): ...


# Implementation by: https://github.com/fastapi/typer/discussions/742
def attach_hook(func: Callable[ParamsHook, RHook], hook_output: str = None) -> \
        Callable[..., Any]:
    """
    Decorates a source function to be executed with a pre-execution hook function. The
    hook function's output is passed to the source function as a specified keyword
    argument. This decorator filters keyword arguments for the hook function according
    to its signature, and the rest of the arguments are passed to the source function.
    It updates the wrapper function's signature to include the combined list of
    arguments, excluding the internally managed ``hook_output``.

    This utility allows combining groups of shared options for Typer CLI scripts.

    Example:

    common.py
    ```
    def logging_options(
        log_level: Annotated[int, typer.Option(help="Log level. Must be between 0 and 9."),
        log_to_file: Annotated[Optional[pathlib.Path], typer.Option(help="A file to stream logs to.") = None
        ):
        if log_level < 0 or log_level > 9:
          raise ValueError("log_level must be between 0 and 9.")
        ...
        # create logger
        ...
        return logger
    ```

    main1.py
    ```
    @attach_hook(common.logging_options, hook_output="logger")
    def foo(size: int, logger: Logger):
        ....

    if __name__=="__main__":
        typer.run(foo)
    ```

    main2.py
    ```
    @attach_hook(common.logging_options, hook_output="logger")
    def bar(color: str, logger: Logger):
        ....

    if __name__=="__main__":
        typer.run(bar)
    ```

    in the example above both main1 and main2 cli's enable to specify shared logging
    arguments from the command line, in addition to the specific argument of each
    script.

    :param func: The hook function to execute before the source function. All required
        arguments must be allowed to be passed as keyword arguments.
    :param hook_output: The keyword argument name for the hook's output passed to the
        source function. If None, defaults to the hook function's name.
    :raises BadHookError: If the hook function has an argument with no default value
        that collides with the source function.
    :return: A decorator that chains the hook function with the source function,
        excluding the `hook_output` from the wrapper's external signature.
    """
    if hook_output is None:
        hook_output = func.__name__

    def decorator(source_func: Callable[Concatenate[RHook, ParamsSource], RSource]) -> \
            Callable[Concatenate[ParamsSource, ParamsHook], RSource]:
        source_params = inspect.signature(source_func).parameters

        # Raise BadHookError if the hook has non-default argument that collides with the
        # `source_func`.
        dup_params = [
            k
            for k, v in inspect.signature(func).parameters.items()
            if k in source_params and v.default == inspect.Parameter.empty
        ]
        if dup_params:
            raise BadHookError(f'The following non-default arguments of the hook '
                               f'function (`{func.__name__}`) collide with the source '
                               f'function (`{source_func.__name__}`): {dup_params}')
        hook_params = {
            k: v.replace(kind=inspect.Parameter.KEYWORD_ONLY)
            for k, v in inspect.signature(func).parameters.items()
            if k not in source_params
        }

        @wraps(source_func)
        def wrapper(*args, **kwargs):
            # Filter kwargs for those accepted by the hook function
            hook_kwargs = {k: v for k, v in kwargs.items() if k in hook_params}

            # Execute hook function with its specific kwargs
            hook_result = func(**hook_kwargs)

            # Filter in the remaining kwargs for the source function.
            source_kwargs = {k: v for k, v in kwargs.items() if k not in hook_kwargs}

            # Execute the source function with original args and pass the hook's output
            # to the source function as the specified keyword argument
            return source_func(*args, **source_kwargs, **{hook_output: hook_result})

        # Combine signatures, but remove the hook_output
        combined_params = [param for param in source_params.values()
                           if param.name != hook_output] + list(hook_params.values())
        wrapper.__signature__ = inspect.signature(source_func).replace(
            parameters=combined_params
        )

        # Combine annotations, but remove the hook_output
        wrapper.__annotations__ = {
            **typing.get_type_hints(source_func),
            **typing.get_type_hints(func),
        }
        wrapper.__annotations__.pop(hook_output)

        return wrapper

    return decorator


def job_options(dependencies: str | list[str] = None):
    if isinstance(dependencies, str):
        dependencies = [dependencies]
    if dependencies:
        for dep in dependencies:
            if dep in settings.dependencies.to_dict():
                if dep not in xtl.cli.utilities.common.REQUIRED_DEPENDENCIES:
                    xtl.cli.utilities.common.REQUIRED_DEPENDENCIES.append(dep)
                modules = getattr(settings.dependencies, dep).modules
                for module in modules:
                    if module not in xtl.cli.utilities.common.REQUIRED_MODULES:
                        xtl.cli.utilities.common.REQUIRED_MODULES.append(module)
            else:
                logger.warning('Dependency %s is not defined in '
                               'xtl.settings.dependencies', dep)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator
