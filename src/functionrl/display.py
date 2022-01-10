import html


def print_grid(values, value_formatter=None):
    if value_formatter is None:
        value_formatter = lambda x: x
    n_states = len(values)
    n_cols = int(n_states ** 0.5)
    for i, value in enumerate(values, start=1):
        sep = "\n" if i % n_cols == 0 else " "
        print(value_formatter(value), end=sep)
    print()


def print_pi(pi):
    arrows = [html.unescape(f"&{c}arr;") for c in ["l", "d", "r", "u"]]
    print_grid(pi, lambda a: arrows[a])


def print_v(v):
    print_grid(v, "{:.4f}".format)
