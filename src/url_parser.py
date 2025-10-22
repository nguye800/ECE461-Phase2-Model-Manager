import csv
from pathlib import Path

from config import ModelURLs


def read_url_csv(url_file_path: Path) -> list[ModelURLs]:
    model_urls: list[ModelURLs] = []
    with open(url_file_path.resolve(), mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, fieldnames=['codebase', 'dataset', 'model'], skipinitialspace=True)
        for row in csv_reader:
            model_urls.append(ModelURLs(**row))

    return model_urls