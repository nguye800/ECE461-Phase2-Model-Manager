import unittest
import os
import typing
from src.metric import *  # pyright: ignore[reportWildcardImportFromLibrary, reportMissingTypeStubs]
from src.workflow import *  # pyright: ignore[reportWildcardImportFromLibrary, reportMissingTypeStubs]
from src.database import *  # pyright: ignore[reportWildcardImportFromLibrary, reportMissingTypeStubs]


class TestDatabaseAccess(unittest.TestCase):
    def setUp(self):
        self.test_db = Path("test.db")
        self.maxDiff = None
        # normal schema
        self.schema1 = [
            FloatMetric("size", 0.3, 10),
            FloatMetric("setup", 0.2, 30),
            DictMetric("compatibility", {"windows": 0.5, "mac": 0.2, "linux": 0.8}, 29),
        ]
        self.schema2 = [
            FloatMetric("size", 0.3, 10),
            FloatMetric("setup", 0.2, 30),
            FloatMetric("speed", 0.7, 49),
            FloatMetric("accuracy", 0.1, 10),
            DictMetric("compatibility", {"windows": 0.5, "mac": 0.2, "linux": 0.8}, 29),
            FloatMetric("team_size", 0.9, 9),
            FloatMetric("team_balance", 0.6, 41),
            FloatMetric("funding", 0.8, 1000),
            FloatMetric("compliance", 0.1, 943),
            FloatMetric("license", 0.5, 234),
        ]
        self.schema3 = [
            DictMetric(
                "size", {"raspberry_pi": 0.5, "desktop_pc": 0.7, "aws_server": 1.0}, 54
            ),
            DictMetric("compatibility", {"windows": 0.5, "mac": 0.2, "linux": 0.8}, 29),
            DictMetric(
                "documentation",
                {"english": 0.9, "spanish": 0.5, "french": 0.2, "mandarin": 0.0},
                1094,
            ),
            DictMetric("license", {"distribution": 1.0, "modification": 0.3}, 297),
            FloatMetric("setup", 0.2, 30),
            FloatMetric("speed", 0.7, 49),
            FloatMetric("accuracy", 0.1, 10),
            DictMetric("database", {"size": 0.2, "quality": 0.9}, 964),
            FloatMetric("team_size", 0.9, 9),
        ]

        # edge cases

        self.schema0: list = []
        self.schema_weird_names: list = [
            FloatMetric("valid sql header, actually", 0.3, 11),
            DictMetric(
                "this one too",
                {
                    "and this": 0.0,
                    "and that": 1.0,
                    "and this again": 0.5,
                    "fourth one": 0.1,
                },
                3939,
            ),
        ]

    # database initialization
    def testCreateNewDatabase(self):

        accessor = SQLiteAccessor(None, self.schema1)
        accessor.cursor.execute("PRAGMA table_info(models)")
        columns = {row[1]: row[2] for row in accessor.cursor.fetchall()}

        self.assertDictEqual(
            columns,
            {
                "model_url": "TEXT",
                "name": "TEXT",
                "database_url": "TEXT",
                "code_url": "TEXT",
                "net_score": "REAL",
                "net_score_latency": "INTEGER",
                "size": "REAL",
                "size_latency": "INTEGER",
                "setup": "REAL",
                "setup_latency": "INTEGER",
                "compatibility_windows": "REAL",
                "compatibility_mac": "REAL",
                "compatibility_linux": "REAL",
                "compatibility_latency": "INTEGER",
            },
        )

    def test_add_and_get_model(self):
        accessor = SQLiteAccessor(None, self.schema1)
        model = ModelStats(
            model_url="example.com/test_url",
            name="Test Model",
            database_url="example.com/database_url",
            code_url="github.com/me/too",
            net_score=0.99,
            net_score_latency=5,
            metrics=[
                FloatMetric("size", 0.3, 10),
                FloatMetric("setup", 0.2, 30),
                DictMetric(
                    "compatibility", {"windows": 0.5, "mac": 0.2, "linux": 0.8}, 29
                ),
            ],
        )
        accessor.add_to_db(model)
        fetched = accessor.get_model_statistics("example.com/test_url")
        self.assertEqual(fetched.model_url, model.model_url)
        self.assertEqual(fetched.name, model.name)
        self.assertEqual(fetched.database_url, model.database_url)
        self.assertEqual(fetched.code_url, model.code_url)
        self.assertEqual(fetched.net_score, model.net_score)
        self.assertEqual(fetched.net_score_latency, model.net_score_latency)
        # Check metrics
        for m1, m2 in zip(model.metrics, fetched.metrics):
            self.assertEqual(m1.name, m2.name)
            self.assertEqual(m1.latency, m2.latency)
            self.assertEqual(m1.data, m2.data)

    def test_add_and_get_model_schema2(self):
        accessor = SQLiteAccessor(None, self.schema2)
        model2 = ModelStats(
            model_url="example.com/test_url",
            name="Test Model",
            database_url="example.com/test_url_2",
            code_url="github.com/second/url",
            net_score=0.99,
            net_score_latency=5,
            metrics=self.schema2,
        )
        accessor.add_to_db(model2)
        fetched = accessor.get_model_statistics("example.com/test_url")
        self.assertEqual(fetched.model_url, model2.model_url)
        self.assertEqual(fetched.name, model2.name)
        self.assertEqual(fetched.database_url, model2.database_url)
        self.assertEqual(fetched.code_url, model2.code_url)
        self.assertEqual(fetched.net_score, model2.net_score)
        self.assertEqual(fetched.net_score_latency, model2.net_score_latency)
        # Check metrics
        for m1, m2 in zip(model2.metrics, fetched.metrics):
            self.assertEqual(m1.name, m2.name)
            self.assertEqual(m1.latency, m2.latency)
            self.assertEqual(m1.data, m2.data)

    def test_add_and_get_model_schema3(self):
        accessor = SQLiteAccessor(None, self.schema3)
        model3 = ModelStats(
            model_url="example.com/test_url",
            name="Test Model",
            database_url="example.com/test_url",
            code_url="example.com/test_url",
            net_score=0.99,
            net_score_latency=5,
            metrics=self.schema3,
        )
        accessor.add_to_db(model3)
        fetched = accessor.get_model_statistics("example.com/test_url")
        self.assertEqual(fetched.model_url, model3.model_url)
        self.assertEqual(fetched.name, model3.name)
        self.assertEqual(fetched.database_url, model3.database_url)
        self.assertEqual(fetched.code_url, model3.code_url)
        self.assertEqual(fetched.net_score, model3.net_score)
        self.assertEqual(fetched.net_score_latency, model3.net_score_latency)
        # Check metrics
        for m1, m2 in zip(model3.metrics, fetched.metrics):
            self.assertEqual(m1.name, m2.name)
            self.assertEqual(m1.latency, m2.latency)
            self.assertEqual(m1.data, m2.data)

    def test_check_entry_in_db(self):
        accessor = SQLiteAccessor(None, self.schema1)
        model = ModelStats(
            model_url="example.com/test_url",
            name="Test Model 2",
            database_url="example.com/test_url",
            code_url="example.com/test_url",
            net_score=0.88,
            net_score_latency=7,
            metrics=[
                FloatMetric("size", 0.4, 12),
                FloatMetric("setup", 0.3, 32),
                DictMetric(
                    "compatibility", {"windows": 0.6, "mac": 0.3, "linux": 0.9}, 31
                ),
            ],
        )
        accessor.add_to_db(model)
        self.assertTrue(accessor.check_entry_in_db("example.com/test_url"))
        self.assertFalse(accessor.check_entry_in_db("nonexistent_url"))

    def test_db_exists_schema_match(self):
        accessor = SQLiteAccessor(None, self.schema1)
        self.assertTrue(accessor.db_exists())

    def test_empty_schema(self):
        accessor = SQLiteAccessor(None, self.schema0)
        model = ModelStats(
            model_url="example.com/test_url",
            name="Empty Model",
            database_url="example.com/test_ur2l",
            code_url="example.com/test_ur2l",
            net_score=0.0,
            net_score_latency=0,
            metrics=[],
        )
        accessor.add_to_db(model)
        fetched = accessor.get_model_statistics("example.com/test_url")
        self.assertEqual(fetched.model_url, model.model_url)
        self.assertEqual(fetched.name, model.name)
        self.assertEqual(fetched.database_url, model.database_url)
        self.assertEqual(fetched.code_url, model.code_url)
        self.assertEqual(fetched.metrics, [])

    def test_weird_names_schema(self):
        accessor = SQLiteAccessor(None, self.schema_weird_names)
        model = ModelStats(
            model_url="example.com/weird_url",
            name="Weird Model",
            database_url="example.com/weird_url",
            code_url="example.com/weird_url",
            net_score=0.5,
            net_score_latency=1,
            metrics=self.schema_weird_names,
        )
        accessor.add_to_db(model)
        fetched = accessor.get_model_statistics("example.com/weird_url")
        self.assertEqual(fetched.model_url, model.model_url)
        self.assertEqual(fetched.name, model.name)
        self.assertEqual(fetched.database_url, model.database_url)
        self.assertEqual(fetched.code_url, model.code_url)
        self.assertEqual(fetched.name, model.name)
        self.assertEqual(fetched.net_score, model.net_score)
        self.assertEqual(fetched.net_score_latency, model.net_score_latency)
        for m1, m2 in zip(model.metrics, fetched.metrics):
            self.assertEqual(m1.name, m2.name)
            self.assertEqual(m1.latency, m2.latency)
            self.assertEqual(m1.data, m2.data)

    def test_get_nonexistent_model(self):
        accessor = SQLiteAccessor(None, self.schema1)
        model = ModelStats(
            model_url="example.com/test_url2",
            name="Test Model 2",
            database_url="example.com/test_url2",
            code_url="example.com/test_url2",
            net_score=0.88,
            net_score_latency=7,
            metrics=[
                FloatMetric("size", 0.4, 12),
                FloatMetric("setup", 0.3, 32),
                DictMetric(
                    "compatibility", {"windows": 0.6, "mac": 0.3, "linux": 0.9}, 31
                ),
            ],
        )
        accessor.add_to_db(model)
        # test for panics
        accessor.get_model_statistics("example.com/test_url2")
        with self.assertRaises(ValueError):
            accessor.get_model_statistics("nonexistent_url")

    def test_wrong_schema(self):

        accessor = SQLiteAccessor(self.test_db, self.schema1)
        del accessor

        accessor_2 = SQLiteAccessor(self.test_db, self.schema3, create_if_missing=False)
        self.assertFalse(accessor_2.db_exists())
        del accessor_2
        os.remove(self.test_db)

    def test_add_with_wrong_schema(self):
        accessor = SQLiteAccessor(None, self.schema1)
        model1 = ModelStats(
            "example.com/correct",
            "Correct",
            "example.com/correct",
            "example.com/correct",
            0.2,
            1225,
            self.schema1,
        )
        model2 = ModelStats(
            "example.com/incorrect",
            "Incorrect",
            "example.com/incorrect",
            "example.com/incorrect",
            0.2,
            1225,
            self.schema2,
        )
        accessor.add_to_db(model1)
        with self.assertRaises(ValueError):
            accessor.add_to_db(model2)

    def test_uninitialized_db(self):
        accessor = SQLiteAccessor(None, self.schema1, create_if_missing=False)
        self.assertFalse(accessor.db_exists())

    def test_get_database_metrics(self):
        accessor = SQLiteAccessor(None, self.schema3)
        model3 = ModelStats(
            model_url="example.com/test_url",
            name="Test Model",
            database_url="example.com/test_url",
            code_url="example.com/test_url",
            net_score=0.99,
            net_score_latency=5,
            metrics=self.schema3,
        )
        database_metrics = self.schema3[0::2]
        accessor.add_to_db(model3)
        fetched = accessor.get_database_metrics_if_exists(
            model3.database_url, database_metrics
        )
        self.assertIsNotNone(fetched)
        if fetched is None:
            # purely to make the linter happy
            raise ValueError("fetched is None")
        # Check metrics
        for m1, m2 in zip(database_metrics, fetched):
            self.assertEqual(m1.name, m2.name)
            self.assertEqual(m1.latency, m2.latency)
            self.assertEqual(m1.data, m2.data)

    def test_get_unknown_database(self):
        accessor = SQLiteAccessor(None, self.schema3)
        model3 = ModelStats(
            model_url="example.com/test_url",
            name="Test Model",
            database_url="example.com/test_url",
            code_url="example.com/test_url",
            net_score=0.99,
            net_score_latency=5,
            metrics=self.schema3,
        )
        database_metrics = self.schema3[0::2]
        accessor.add_to_db(model3)
        fetched = accessor.get_database_metrics_if_exists(
            "example.com/new_url", database_metrics
        )
        self.assertIsNone(fetched)
