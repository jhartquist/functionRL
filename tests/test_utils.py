from functionrl.utils import linear_decay, limitable


def test_linear_decay():
    decay = linear_decay(1, 0.5, 10)
    assert decay(0) == 1
    assert decay(1) == 0.95
    assert decay(5) == 0.75
    assert decay(10) == 0.5
    assert decay(11) == 0.5
    assert decay(100000) == 0.5


def test_limitable():
    @limitable
    def generator():
        while True:
            yield True

    assert len(list(generator(n=10))) == 10
