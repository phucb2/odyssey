"""Entry point for the `mltrain` command."""
from odyssey.training.train import main as train_main


def main() -> None:
    train_main()


if __name__ == "__main__":
    main()
