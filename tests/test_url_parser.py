import unittest
import tempfile
from pathlib import Path
from config import ModelURLs
from url_parser import read_url_csv


class TestURLParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a shared temporary workspace for this test class
        cls._class_tmp = tempfile.TemporaryDirectory()
        cls.workspace = Path(cls._class_tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._class_tmp.cleanup()

    def generate_test_file(self, name: str, content: str) -> Path:
        target_path = self.workspace / name
        target_path.write_text(content, encoding="utf-8")
        return target_path

    def test_proper_format(self):
        test_content = """a, b, c
e,f,g
j,i,h"""

        path: Path = self.generate_test_file("test1.txt", test_content)
        result: list[ModelURLs] = read_url_csv(path)
        self.assertEqual(result[0], ModelURLs(model="c", codebase="a", dataset="b"))
        self.assertEqual(result[1], ModelURLs(model="g", codebase="e", dataset="f"))
        self.assertEqual(result[2], ModelURLs(model="h", codebase="j", dataset="i"))

    def test_empty_fields(self):
        test_content = """a, b, c
    ,,g
    ,,h"""

        path: Path = self.generate_test_file("test1.txt", test_content)
        result: list[ModelURLs] = read_url_csv(path)

        self.assertEqual(result[0], ModelURLs(model="c", codebase="a", dataset="b"))
        self.assertEqual(result[1], ModelURLs(model="g", codebase=None, dataset=None))
        self.assertEqual(result[2], ModelURLs(model="h", codebase=None, dataset=None))

    def test_erroring(self):
        test_content = """a, b, 
    ,,g
    ,,h"""

        path: Path = self.generate_test_file("test1.txt", test_content)

        with self.assertRaises(Exception):
            result: list[ModelURLs] = read_url_csv(path)
