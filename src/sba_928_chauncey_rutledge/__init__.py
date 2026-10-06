"""SBA 928 market-research prompt engineering and fine-tuning project."""


def main() -> None:
    from sba_928_chauncey_rutledge.training import main as training_main

    training_main()

__all__ = ["main"]
