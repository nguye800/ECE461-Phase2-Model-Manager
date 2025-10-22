from pathlib import Path
from database import SQLiteAccessor, FloatMetric, DictMetric
import re
import typing


def get_linked_dataset_metrics(
    readme_file: Path,
    database_accessor: SQLiteAccessor,
    schema: list,
) -> typing.Optional[tuple]:
    """
    Attempts to retrieve metrics tied to the dataset
    Args:
        readme_file: location of the readme
        database_accessor: database to check in
        schema: schema of metrics tied exclusively to the dataset
    """
    text = readme_file.read_text(encoding="utf-8")
    matches = re.findall(
        r"\[.*\]\((.+)\)",
        text,
    )
    for match in matches:
        metrics = database_accessor.get_database_metrics_if_exists(match, schema)
        if metrics is not None:
            return (match, metrics)
