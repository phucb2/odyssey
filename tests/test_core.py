from odyssey import __version__, hello


def test_hello_default() -> None:
    assert hello() == "Hello from odyssey!"


def test_hello_custom_name() -> None:
    assert hello("world") == "Hello from world!"


def test_version() -> None:
    assert __version__ == "0.1.0"
