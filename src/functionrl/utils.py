from functools import wraps
from itertools import count, islice


def linear_decay(start, end, decay_steps):
    delta = (end - start) / decay_steps
    return lambda step: start + delta * step if step < decay_steps else end


def decay_generator(decay_fn):
    for i in count():  # pragma: no branch
        yield decay_fn(i)


def limitable(func):
    @wraps(func)
    def wrapper(*args, n=None, **kwargs):
        yield from islice(func(*args, **kwargs), n)

    return wrapper
