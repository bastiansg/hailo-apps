from threading import Thread
from typing import TypeVar, Callable, Any


F = TypeVar("F", bound=Callable[..., None])


def threaded(func: Callable[..., None]) -> Callable[..., None]:
    def wrapper(*args: Any, **kwargs: Any) -> None:
        thread = Thread(target=func, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()

    return wrapper
