import unittest
from metrics.license import *  # pyright: ignore[reportWildcardImportFromLibrary]
from pathlib import Path
import os, shutil
from metric import ModelPaths


class LicenseMetricTest(unittest.TestCase):
    TEST_DIR = Path("tests/license_tests")
    in_readme_dir = TEST_DIR / "typical_model"
    linked_license_dir = TEST_DIR / "typical_proprietary_model"
    linked_license_md_dir = TEST_DIR / "typical_proprietary_model_2"
    no_readme_dir = TEST_DIR / "extremely_weird_model"
    text_in_readme_dir = TEST_DIR / "theoretically_possible_model"
    no_license_dir = TEST_DIR / "fake_model"
    no_license_but_readme_dir = TEST_DIR / "not_a_model"
    unknown_noncommercial_license_dir = TEST_DIR / "research_model"

    mit_license = Path("tests/sample_licenses/MIT.txt")
    lgpl_v3_license = Path("tests/sample_licenses/lgpl v3.txt")

    license_instance = LicenseMetric()

    def setUp(self) -> None:
        # set up testing directories
        if self.TEST_DIR.exists():
            shutil.rmtree(self.TEST_DIR)
        os.makedirs(self.TEST_DIR)
        os.mkdir(self.in_readme_dir)
        os.mkdir(self.linked_license_dir)
        os.mkdir(self.linked_license_md_dir)
        os.mkdir(self.no_readme_dir)
        os.mkdir(self.text_in_readme_dir)
        os.mkdir(self.no_license_dir)
        os.mkdir(self.no_license_but_readme_dir)
        os.mkdir(self.unknown_noncommercial_license_dir)

        # license included in metadata
        readme_file = self.in_readme_dir / "README.md"
        with open(readme_file, "wt") as file:
            file.writelines(
                [
                    "---\n",
                    "license: lgpl-2.1\n",
                    "---\n",
                    "# Typical Model\n",
                    "This is a normal model :)\n",
                ]
            )

        # license included as a link to a LICENSE file
        readme_file = self.linked_license_dir / "README.md"
        with open(readme_file, "wt") as file:
            file.writelines(
                [
                    "---\n",
                    "license: other\n",
                    "license-name: idk\n",
                    "---\n",
                    "# Proprietary Model\n",
                    "This is a big model with weird permission :/\n",
                    "# License\n",
                    "Lol no actually it's just [LGPL v3](LICENSE)\n",
                ]
            )
        shutil.copy(self.lgpl_v3_license, self.linked_license_dir / "LICENSE")

        # license included as a link to a LICENSE.md file
        readme_file = self.linked_license_md_dir / "README.md"
        with open(readme_file, "wt") as file:
            file.writelines(
                [
                    "---\n",
                    "license: other\n",
                    "license-name: idk\n",
                    "---\n",
                    "# Proprietary Model\n",
                    "This is a big model with weird permission :/\n",
                    "# License\n",
                    "Lol no actually it's just [MIT](LICENSE.md)\n",
                ]
            )
        shutil.copy(self.mit_license, self.linked_license_md_dir / "LICENSE.md")

        # license included as LICENSE but no readme present
        readme_file = self.no_readme_dir / "README.md"
        shutil.copy(self.mit_license, self.no_readme_dir / "LICENSE.md")

        # license text included in full in the readme
        mit_text: list[str]
        with open(self.mit_license, "rt") as file:
            mit_text = file.readlines()
        readme_file = self.text_in_readme_dir / "README.md"
        with open(readme_file, "wt") as file:
            file.writelines(
                [
                    "---\n",
                    "license: other\n",
                    "license-name: idk\n",
                    "---\n",
                    "# Proprietary Model\n",
                    "Collect my pages.\n",
                    "# License\n",
                    "",
                ]
            )
            file.writelines(mit_text)
            file.writelines(
                [
                    "\n# Credits\n",
                    "ChatGPT made this entire thing, which is why I can't use a more restrictive license.\n",
                    "I don't know if I technically own this code, honestly?\n",
                ]
            )

        # no license
        with open(self.no_license_dir / "todo.txt", "wt") as file:
            file.write("TODO: create model")

        # no license, but with a readme
        with open(self.no_license_but_readme_dir / "README.md", "wt") as file:
            file.writelines(
                ["# Not A Model\n", "This repository is not for a model.\n"]
            )

        # unknown non-commercial license
        with open(self.unknown_noncommercial_license_dir / "LICENSE.md", "wt") as file:
            file.writelines(
                ["This model must be used for non-commercial, research purposes only."]
            )

        return super().setUp()

    def tearDown(self) -> None:
        shutil.rmtree(self.TEST_DIR)
        return super().tearDown()

    def testReadmeInMetadata(self):
        dirs = ModelPaths()
        dirs.model = self.in_readme_dir
        self.license_instance.set_local_directory(dirs)
        self.license_instance.run()
        self.assertAlmostEqual(self.license_instance.parse_license_file(), 0.0)
        self.assertIsInstance(self.license_instance.score, float)
        if isinstance(self.license_instance.score, dict):
            return
        self.assertAlmostEqual(self.license_instance.score, license_score["lgpl-2.1"])

    def testReadmeLinked(self):
        dirs = ModelPaths()
        dirs.model = self.linked_license_dir
        self.license_instance.set_local_directory(dirs)
        self.license_instance.run()
        self.assertIsInstance(self.license_instance.score, float)
        if isinstance(self.license_instance.score, dict):
            return
        self.assertAlmostEqual(self.license_instance.score, license_score["lgpl-3.0+"])

    def testReadmeLinked_md(self):
        dirs = ModelPaths()
        dirs.model = self.linked_license_md_dir
        self.license_instance.set_local_directory(dirs)
        self.license_instance.run()
        self.assertIsInstance(self.license_instance.score, float)
        if isinstance(self.license_instance.score, dict):
            return
        self.assertAlmostEqual(self.license_instance.score, license_score["mit"])

    def testNoReadme(self):
        dirs = ModelPaths()
        dirs.model = self.no_readme_dir
        self.license_instance.set_local_directory(dirs)
        self.license_instance.run()
        self.assertIsInstance(self.license_instance.score, float)
        if isinstance(self.license_instance.score, dict):
            return
        self.assertAlmostEqual(self.license_instance.score, license_score["mit"])

    def testLicenseTextInReadme(self):
        dirs = ModelPaths()
        dirs.model = self.in_readme_dir
        self.license_instance.set_local_directory(dirs)
        self.license_instance.run()
        self.assertIsInstance(self.license_instance.score, float)
        if isinstance(self.license_instance.score, dict):
            return
        self.assertAlmostEqual(self.license_instance.score, license_score["mit"])

    def testNoLicense(self):
        dirs = ModelPaths()
        dirs.model = self.no_license_dir
        self.license_instance.set_local_directory(dirs)
        self.license_instance.run()
        self.assertIsInstance(self.license_instance.score, float)
        if isinstance(self.license_instance.score, dict):
            return
        self.assertAlmostEqual(self.license_instance.score, 0.0)

    def testNoLicenseButReadme(self):
        dirs = ModelPaths()
        dirs.model = self.no_license_but_readme_dir
        self.license_instance.set_local_directory(dirs)
        self.license_instance.run()
        self.assertIsInstance(self.license_instance.score, float)
        if isinstance(self.license_instance.score, dict):
            return
        self.assertAlmostEqual(self.license_instance.score, 0.0)

    def testUnknownNoncommercialLicense(self):
        dirs = ModelPaths()
        dirs.model = self.unknown_noncommercial_license_dir
        self.license_instance.set_local_directory(dirs)
        self.license_instance.run()
        self.assertIsInstance(self.license_instance.score, float)
        if isinstance(self.license_instance.score, dict):
            return
        self.assertAlmostEqual(self.license_instance.score, 0.0)

    def testNoLocalDirectory(self):
        incomplete_instance = LicenseMetric()
        with self.assertRaises(ValueError):
            incomplete_instance.setup_resources()
            incomplete_instance.calculate_score()
