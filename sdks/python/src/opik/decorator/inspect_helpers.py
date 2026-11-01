import inspect
import functools

from typing import Callable, Tuple, Any, Dict, Optional


def extract_inputs(
    func: Callable, args: Tuple, kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    return extract_inputs_and_var_keyword_key(func, args, kwargs)[0]


def extract_inputs_and_var_keyword_key(
    func: Callable, args: Tuple, kwargs: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Same as extract_inputs, but also returns the key under which the arguments
    passed through **kwargs were captured, or None if there is no such key.
    """
    sig = inspect.signature(func)

    try:
        bound_args = sig.bind(*args, **kwargs)  # type: ignore
        bound_args.apply_defaults()
        arg_dict = dict(bound_args.arguments)
        var_keyword_key = next(
            (
                parameter.name
                for parameter in sig.parameters.values()
                if parameter.kind is inspect.Parameter.VAR_KEYWORD
            ),
            None,
        )
    except TypeError:
        arg_dict = {
            "args": args,
            "kwargs": kwargs,
        }
        var_keyword_key = "kwargs"

    if "self" in arg_dict:
        arg_dict.pop("self")
    elif "cls" in arg_dict:
        arg_dict.pop("cls")

    return arg_dict, var_keyword_key


def is_generator(obj: Any) -> bool:
    return inspect.isgenerator(obj)


def _wrapped_chain_contains(obj: Any, predicate: Callable[[Any], bool]) -> bool:
    # Walk the full __wrapped__ chain: the inspect predicates never follow it
    # (beyond functools.partial), so a function decorated with one or more
    # functools.wraps-based decorators is not recognized as a coroutine,
    # generator, or async generator function on its own.
    seen = set()
    current: Any = obj
    while current is not None:
        if id(current) in seen:
            break
        seen.add(id(current))
        if predicate(current):
            return True
        current = getattr(current, "__wrapped__", None)

    return False


def is_generator_function(obj: Any) -> bool:
    return _wrapped_chain_contains(obj, inspect.isgeneratorfunction)


def is_async_generator_function(obj: Any) -> bool:
    return _wrapped_chain_contains(obj, inspect.isasyncgenfunction)


def is_async(func: Callable) -> bool:
    return _wrapped_chain_contains(func, inspect.iscoroutinefunction)


def get_function_name(func: Callable) -> str:
    """Safely get the name of a function, handling functools.partial objects."""
    if isinstance(func, functools.partial):
        # For partial objects, get the name from the underlying function
        return getattr(func.func, "__name__", "<unknown>")

    # For regular functions and other callables
    return getattr(func, "__name__", "<unknown>")
