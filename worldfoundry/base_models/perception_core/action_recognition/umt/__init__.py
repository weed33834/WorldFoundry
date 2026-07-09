from pathlib import Path


def kinetics_categories_path() -> Path:
    return Path(__file__).with_name("kinetics_400_categories.txt")
