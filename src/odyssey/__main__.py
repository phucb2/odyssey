"""Allow `python -m odyssey` / `make run`."""

from odyssey import __version__, hello


def main() -> None:
    print(f"odyssey {__version__}")
    print(hello())


if __name__ == "__main__":
    main()
