from functionrl.display import print_grid, print_pi, print_v


def test_print_grid(capfd):
    print_grid([1, 2, 3, 4])
    out = capfd.readouterr()[0]
    assert out == "1 2\n3 4\n\n"


def test_v(capfd):
    v = [1, 2, 3, 4]
    print_v(v)
    out = capfd.readouterr()[0]
    assert out == "1.0000 2.0000\n3.0000 4.0000\n\n"


def test_print_pi(capfd):
    pi = [0, 1, 3, 2]
    print_pi(pi)
    out = capfd.readouterr()[0]
    assert out == "← ↓\n↑ →\n\n"
