from typing import Protocol, Generic, TypeVar
import sqlite3
from pathlib import Path
from typing_extensions import override

PROD_DATABASE_PATH: Path = Path("models.db")

T = TypeVar("T")


class MetricStats(Generic[T]):
    def __init__(self, name: str, data: T, latency: int):
        self.name = name
        self.data = data
        self.latency = latency

    def to_sql_schema(self) -> str:
        raise NotImplementedError(
            "Metric does not have defined SQL type"
        )  # pragma: no cover


class FloatMetric(MetricStats[float]):
    @override
    def to_sql_schema(self) -> str:
        return f'"{self.name}" REAL, "{self.name}_latency" INTEGER'


class DictMetric(MetricStats[dict[str, float]]):
    @override
    def to_sql_schema(self) -> str:
        schema: str = ""
        for key, _ in self.data.items():
            schema += f'"{self.name}_{key}" REAL, '
        schema += f'"{self.name}_latency" INTEGER'
        return schema


class ModelStats:
    def __init__(
        self,
        model_url: str,
        name: str,
        database_url: str,
        code_url: str,
        net_score: float,
        net_score_latency: int,
        metrics: list,
    ):
        self.model_url = model_url
        self.name = name
        self.database_url = database_url
        self.code_url = code_url
        self.net_score = net_score
        self.net_score_latency = net_score_latency
        self.metrics = metrics


class DatabaseAccessor(Protocol):
    def init_database(self): ...

    def db_exists(self) -> bool: ...

    # checks whether or not a given model is in the database
    def check_entry_in_db(self) -> bool: ...

    def add_to_db(self, model: ModelStats): ...

    def get_model_statistics(self, model_url: str) -> ModelStats: ...


class SQLiteAccessor:
    def __init__(
        self,
        db_location,
        metric_schema: list,
        create_if_missing: bool = True,
    ):
        if db_location is None:
            self.connection: sqlite3.Connection = sqlite3.connect(":memory:")
        else:
            self.connection: sqlite3.Connection = sqlite3.connect(db_location)
        self.cursor: sqlite3.Cursor = self.connection.cursor()
        self.metric_schema = metric_schema
        if not self.db_exists() and create_if_missing:
            self.init_database()

    def __del__(self):
        self.connection.close()

    def db_exists(self) -> bool:
        try:
            self.cursor.execute("PRAGMA table_info(models)")
            columns = {row[1]: row[2] for row in self.cursor.fetchall()}
        except sqlite3.OperationalError:  # pragma: no cover
            return False

        # Normalize column names (remove quotes)
        normalized_columns = {col.replace('"', ""): typ for col, typ in columns.items()}

        # Required base columns
        required_columns = {
            "model_url": "TEXT",
            "name": "TEXT",
            "database_url": "TEXT",
            "code_url": "TEXT",
            "net_score": "REAL",
            "net_score_latency": "INTEGER",
        }

        # Add metric columns from metric_schema
        for metric in self.metric_schema:
            items = [
                item.strip()
                for item in metric.to_sql_schema().split(",")
                if item.strip()
            ]
            for item in items:
                parts = item.split()
                if len(parts) == 2:
                    col, typ = parts
                    col = col.replace('"', "")
                    required_columns[col] = typ

        for col, typ in required_columns.items():
            if col not in normalized_columns.keys() or normalized_columns[col] != typ:
                return False
        return True

    def init_database(self):
        # create the table with schema matching ModelStats, url as PRIMARY KEY
        metric_schema_str = ", ".join([m.to_sql_schema() for m in self.metric_schema])
        if metric_schema_str:
            sql = f"""
                CREATE TABLE IF NOT EXISTS models (
                    model_url TEXT PRIMARY KEY,
                    name TEXT,
                    database_url TEXT,
                    code_url TEXT,
                    net_score REAL,
                    net_score_latency INTEGER,
                    {metric_schema_str}
                )
            """
        else:
            sql = f"""
                CREATE TABLE IF NOT EXISTS models (
                    model_url TEXT PRIMARY KEY,
                    name TEXT,
                    database_url TEXT,
                    code_url TEXT,
                    net_score REAL,
                    net_score_latency INTEGER
                )
            """
        self.cursor.execute(sql)
        self.connection.commit()

    def get_database_metrics_if_exists(
        self, url: str, schema: list
    ):
        self.cursor.execute("SELECT * FROM models WHERE database_url = ?", (url,))
        row = self.cursor.fetchone()
        if row is None:
            return None
        col_names = [desc[0] for desc in self.cursor.description]
        scores: list = []
        for metric in schema:
            if isinstance(metric, FloatMetric):
                value = row[col_names.index(metric.name)]
                latency = row[col_names.index(f"{metric.name}_latency")]
                scores.append(FloatMetric(metric.name, value, latency))
            else:
                dict_data: dict[str, float] = {}
                for key in metric.data.keys():
                    col = f"{metric.name}_{key}"
                    if col in col_names:
                        dict_data[key] = row[col_names.index(col)]
                latency = row[col_names.index(f"{metric.name}_latency")]
                scores.append(DictMetric(metric.name, dict_data, latency))
        return scores

    def check_entry_in_db(self, url: str) -> bool:
        self.cursor.execute("SELECT model_url from models WHERE model_url = ?", (url,))
        return self.cursor.fetchone() is not None

    def add_to_db(self, model_stats: ModelStats):
        # Build columns and values for base fields
        columns = [
            "model_url",
            "name",
            "database_url",
            "code_url",
            "net_score",
            "net_score_latency",
        ]
        values = [
            model_stats.model_url,
            model_stats.name,
            model_stats.database_url,
            model_stats.code_url,
            model_stats.net_score,
            model_stats.net_score_latency,
        ]

        # Add metric columns and values
        for metric in model_stats.metrics:
            if isinstance(metric, FloatMetric):
                columns.append(metric.name)
                columns.append(f"{metric.name}_latency")
                values.append(metric.data)
                values.append(metric.latency)
            else:
                for key, val in metric.data.items():
                    columns.append(f"{metric.name}_{key}")
                    values.append(val)
                columns.append(f"{metric.name}_latency")
                values.append(metric.latency)

        # Validate columns against database schema
        self.cursor.execute("PRAGMA table_info(models)")
        db_columns = set(row[1] for row in self.cursor.fetchall())
        for col in columns:
            if col not in db_columns:
                raise ValueError(f"Column '{col}' not found in database schema.")

        # Build SQL statement
        col_str = ", ".join([f'"{col}"' for col in columns])
        placeholders = ", ".join(["?" for _ in values])
        sql = f"INSERT OR REPLACE INTO models ({col_str}) VALUES ({placeholders})"
        self.cursor.execute(sql, values)
        self.connection.commit()

    def get_model_statistics(self, model_url: str) -> ModelStats:
        self.cursor.execute("SELECT * FROM models WHERE model_url = ?", (model_url,))
        row = self.cursor.fetchone()
        if not row:
            raise ValueError(f"No entry found in database for URL: {model_url}")

        # Get column names for mapping
        col_names = [desc[0] for desc in self.cursor.description]

        # Extract base fields
        model_url = row[col_names.index("model_url")]
        name = row[col_names.index("name")]
        database_url = row[col_names.index("database_url")]
        code_url = row[col_names.index("code_url")]
        net_score = row[col_names.index("net_score")]
        net_score_latency = row[col_names.index("net_score_latency")]

        metrics = []
        for metric in self.metric_schema:
            if isinstance(metric, FloatMetric):
                value = row[col_names.index(metric.name)]
                latency = row[col_names.index(f"{metric.name}_latency")]
                metrics.append(FloatMetric(metric.name, value, latency))
            else:
                dict_data: dict[str, float] = {}
                for key in metric.data.keys():
                    col = f"{metric.name}_{key}"
                    if col in col_names:
                        dict_data[key] = row[col_names.index(col)]
                latency = row[col_names.index(f"{metric.name}_latency")]
                metrics.append(DictMetric(metric.name, dict_data, latency))

        return ModelStats(
            model_url,
            name,
            database_url,
            code_url,
            net_score,
            net_score_latency,
            metrics,
        )
