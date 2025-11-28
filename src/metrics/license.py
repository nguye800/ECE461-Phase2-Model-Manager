from metric import BaseMetric  # pyright: ignore[reportMissingTypeStubs]
from pathlib import Path
import re
import spdx_matcher

metadata_pattern = re.compile(r"^license: (.*)$")
heading_pattern = re.compile(r"^#+ *(.*)$")
license_link_pattern = re.compile(r"\[.+\]\(LICENSE.*\)")

# full huggingface license list
license_score: dict[str, float] = {
    # 0.0 means either non-commercial or incompatible with LGPL v2.1 (see https://www.gnu.org/licenses/license-list.html)
    "apache-2.0": 1.0,  # only compatible with gpl v3
    "mit": 1.0,
    # all incompatible because they impose restrictions on reuse not present in the LGPL
    "openrail": 0.0,
    "creativeml-openrail-m": 0.0,
    "bigscience-openrail-m": 0.0,
    "bigscience-bloom-rail-1.0": 0.0,
    "bigcode-openrail-m": 0.0,
    "afl-3.0": 0.0,
    "artistic-2.0": 0.9,
    "bsl-1.0": 1.0,
    "bsd": 1.0,
    "bsd-2-clause": 1.0,
    "bsd-3-clause": 1.0,
    "bsd-3-clause-clear": 1.0,
    "c-uda": 0.0,
    # cc is too broad since it could include sharealike and non-commercial licenses; use fallback
    "cc0-1.0": 1.0,
    "cc-by-2.0": 1.0,
    "cc-by-2.5": 1.0,
    "cc-by-3.0": 1.0,
    "cc-by-4.0": 1.0,
    "cc-by-sa-3.0": 0.0,
    "cc-by-sa-4.0": 0.0,
    "cc-by-nc-2.0": 0.0,
    "cc-by-nc-3.0": 0.0,
    "cc-by-nc-4.0": 0.0,
    "cc-by-nc-nd-3.0": 0.0,
    "cc-by-nc-nd-4.0": 0.0,
    "cc-by-nc-sa-2.0": 0.0,
    "cc-by-nc-sa-3.0": 0.0,
    "cc-by-nc-sa-4.0": 0.0,
    "cdla-sharing-1.0": 0.0,
    "cdla-permissive-1.0": 1.0,
    "cdla-permissive-2.0": 1.0,
    "wtfpl": 1.0,
    "ecl-2.0": 0.0,  # only compatible with gpl v3.0
    "epl-1.0": 0.0,
    "epl-2.0": 0.3,  # potentially usable depending on the secondary license allowances
    "etalab-2.0": 0.0,
    # technically compatible through crazy relicensing shenanigans
    "eupl-1.1": 0.3,
    "eupl-1.2": 0.3,
    "agpl-3.0": 0.0,
    "gfdl": 0.0,  # for documentation, so likely weird compatibility-wise
    "gpl": 1.0,  # lgpl says you can choose whatever gpl license you want if it isn't specified by the distributor
    # remember all code is distributed under LGPL v2.1
    # see https://www.gnu.org/licenses/gpl-faq.html#AllCompatibility
    "gpl-2.0": 0,
    "gpl-2.0+": 0,
    "gpl-2.0-or-later": 0,
    "gpl-3.0": 0.0,
    "gpl-3.0+": 0.0,
    "gpl-3.0-or-later": 0,
    "lgpl": 1.0,  #  lgpl says you can choose whatever lgpl license you want if it isn't specified by the distributor
    "lgpl-2.1": 1.0,
    "lgpl-2.1+": 1.0,
    "lgpl-2.1-or-later": 1.0,
    "lgpl-3.0": 0.7,  # re-use is allowed, but modification will require the code to be relicensed.
    "lgpl-3.0+": 0.7,  # re-use is allowed, but modification will require the code to be relicensed.
    "lgpl-3.0-or-later": 0.7,
    "isc": 1.0,
    "h-research": 0.0,  # non commercial
    "intel-research": 0.0,  # restrictions on redistribution.
    "lppl-1.3c": 0.0,  # incompatible with gpl v2/3, unsure with lgpl, also no real models use it.
    "ms-pl": 0.0,  # copyleft and incompatible with gpl copyleft
    "apple-ascl": 0.5,  # can redistribute as long as no modifications are made
    "apple-amlr": 0.0,  # can only use for research purposes
    "mpl-2.0": 1.0,
    "odc-by": 1.0,
    "odbl": 0.0,
    "openmdw-1.0": 1.0,
    "openrail++": 0.0,
    "osl-3.0": 0.0,
    "postgresql": 1.0,
    "ofl-1.1": 0.0,  # copyleft
    "ncsa": 1.0,
    "unlicense": 1.0,
    "zlib": 1.0,
    "pddl": 1.0,
    "lgpl-lr": 0.0,
    "deepfloyd-if-license": 0.0,  # non-commercial
    "fair-noncommercial-research-license": 0.0,
    # all the llama licenses are not free software licenses, so they can't be
    # redistributed under the LGPL
    "llama2": 0.0,
    "llama3": 0.0,
    "llama3.1": 0.0,
    "llama3.2": 0.0,
    "llama3.3": 0.0,
    "llama4": 0.0,
    # can't redistribute under LGPL, not an open source license
    "gemma": 0.0,
}


# License metric
# Assumes it is being run in the base directory of the model repository
class LicenseMetric(BaseMetric):
    metric_name: str = "license"
    license_file: Path
    model_dir: Path
    readme_file: Path

    def __init__(self):
        super().__init__()

    def parse_readme(self) -> float:
        readme_text = self.readme_file.read_text(encoding='utf-8')
        metadata_score = None
        readme_score = None
        license_section: str = ''
        current_heading = None
        for line in readme_text.splitlines():
            if current_heading is not None and current_heading.lower() == 'license':
                license_section += f"{line}\n"
            capture = heading_pattern.match(line)
            if capture is not None:
                new_heading = capture.group(1)
                if isinstance(new_heading, str):
                    current_heading = new_heading
            capture = metadata_pattern.match(line)
            if capture is not None:
                license_name = capture.group(1)
                if isinstance(license_name, str):
                    metadata_score = license_score.get(license_name)

        if license_section:
            matches = license_link_pattern.findall(license_section)
            if matches:
                readme_score = self.parse_license_file()
                existing = self.explain_score()
                if existing:
                    self._set_debug_details(f"README references LICENSE; {existing}")
                else:
                    self._set_debug_details('README references LICENSE but parsing failed')
            else:
                try:
                    licenses_detected, _ = spdx_matcher.analyse_license_text(license_section)
                except Exception:
                    licenses_detected = {}
                if isinstance(licenses_detected, dict) and licenses_detected.get('license'):
                    spdx_ids = list(licenses_detected['license'].keys())
                    if spdx_ids:
                        spdx_id = spdx_ids[0].lower()
                        readme_score = license_score.get(spdx_id, 0.0)
                        self._set_debug_details(f"README license text matched {spdx_id}")

        if readme_score is not None:
            return readme_score
        if metadata_score is not None:
            self._set_debug_details(f"README metadata license detected -> score {metadata_score}")
            return metadata_score
        self._set_debug_details('No license details found in README')
        return 0.0

    def parse_license_file(self) -> float:
        if not self.license_file.exists():
            self._set_debug_details('LICENSE file missing')
            return 0.0

        license_text = self.license_file.read_text(encoding='utf-8', errors='ignore')
        lt = license_text.lower()

        if not heuristics_check(lt):
            self._set_debug_details('LICENSE rejected by heuristics (likely non-commercial)')
            return 0.0

        spdx_ids: list[str] = []
        try:
            detected, _percent = spdx_matcher.analyse_license_text(license_text)
            if isinstance(detected, dict):
                if isinstance(detected.get('license'), dict):
                    spdx_ids = list(detected['license'].keys())
                elif isinstance(detected.get('licenses'), dict):
                    spdx_ids = list(detected['licenses'].keys())
                elif isinstance(detected.get('matches'), list):
                    spdx_ids = [m.get('spdx_id') for m in detected['matches'] if m.get('spdx_id')]
        except Exception:
            pass

        if not spdx_ids:
            if 'mit license' in lt or 'permission is hereby granted, free of charge' in lt:
                self._set_debug_details('LICENSE heuristically matched MIT text')
                return license_score['mit']
            if 'gnu lesser general public license' in lt or ' lgpl' in lt:
                if 'version 2.1' in lt or 'version 2.1-only' in lt or 'version 2.1 or later' in lt:
                    self._set_debug_details('LICENSE heuristically matched LGPL-2.1 text')
                    return license_score['lgpl-2.1+']
                self._set_debug_details('LICENSE heuristically matched LGPL-3.0 text')
                return license_score['lgpl-3.0+']
            if 'apache license' in lt and 'version 2.0' in lt:
                self._set_debug_details('LICENSE heuristically matched Apache-2.0')
                return license_score['apache-2.0']

        if not spdx_ids:
            self._set_debug_details('Unable to determine license from LICENSE file')
            return 0.0

        spdx_id = spdx_ids[0].lower()
        self._set_debug_details(f"LICENSE matched SPDX id {spdx_id}")
        return license_score.get(spdx_id, 0.0)

    def calculate_score(self) -> float:
        if self.local_directory is None or self.local_directory.model is None:
            raise ValueError("Local model directory not specified")
        self.model_dir = Path(self.local_directory.model)
        self.license_file = self.model_dir / "LICENSE"
        if not self.license_file.exists():
            self.license_file = self.model_dir / "LICENSE.md"
        self.readme_file = self.model_dir / "README.md"
        if self.readme_file.exists():
            return self.parse_readme()
        elif self.license_file.exists():
            return self.parse_license_file()
        else:
            return 0.0

    def setup_resources(self):
        pass


# simple heuristic to catch non-commercial and copyleft licenses
def heuristics_check(text: str) -> bool:
    t = text.lower()
    # Block non-commercial style restrictions; DO NOT block “copyleft”
    flagged_phrases = [
        "non-commercial", "noncommercial",
        "no commercial use", "not for commercial use",
        "research purposes only", "research use only",
        "non commercial", "noncommercial use"
    ]
    return not any(p in t for p in flagged_phrases)
