import coverage
import unittest
import io
import sys
import logging
import contextlib
import os
from pathlib import Path

project_root = Path(__file__).parent.resolve()
src_path = project_root / "src"
tests_path = project_root / "tests"

buffer = io.StringIO() # careful or the string will bust  D:
with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
# Start coverage before importing project code
    cov = coverage.Coverage(
        source=[str(project_root)],  # measure everything under project root
        omit=[
            "*/__pycache__/*",
            "*/venv/*",
            "*/.venv/*",
            "*/env/*",
            "*/site-packages/*",
            "*__main__.py"
        ],
    )
    cov.start()

    # Ensure paths are importable
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(tests_path) not in sys.path:
        sys.path.insert(0, str(tests_path))

    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(tests_path), pattern="test_*.py")
    total_tests = suite.countTestCases()

    logging.debug(f"Test discovery starting from: {tests_path}")
    logging.debug(f"Tests discovered: {total_tests}")

    runner = unittest.TextTestRunner(verbosity=2)  # show output for debugging
    result = runner.run(suite)

    # Stop coverage
    cov.stop()
    cov.save()

    # Report
    report_output = io.StringIO()
    coverage_percent = cov.report(file=report_output, show_missing=False)
    report_content = report_output.getvalue()
    logging.debug("Coverage report:")
    logging.debug(report_content)

    coverage_data = cov.get_data()
    measured_files = coverage_data.measured_files()
    logging.debug(f"Files measured for coverage: {len(measured_files)}")

    per_file_rows = []
    for filename in measured_files:
        path_obj = Path(filename)
        if src_path not in path_obj.parents and path_obj != src_path:
            continue  # only report files under src/
        try:
            filename, statements, excluded, missing, _ = cov.analysis2(filename)
        except coverage.CoverageException:
            continue
        total = len(statements)
        if total == 0:
            continue
        missing_count = len(missing)
        covered = total - missing_count
        percent = (covered / total) * 100 if total else 100.0
        rel_path = path_obj.relative_to(project_root)
        per_file_rows.append((str(rel_path), percent, covered, total))
        logging.debug(
            f"Measured file: {rel_path} ({covered}/{total} lines, {percent:.1f}%)"
        )

# Summary
passed_tests = total_tests - len(result.failures) - len(result.errors)
print(f"{passed_tests}/{total_tests} test cases passed. {coverage_percent:.0f}% line coverage achieved.")

if per_file_rows:
    print("\nPer-file coverage (src/):")
    max_name = max(len(row[0]) for row in per_file_rows)
    for name, percent, covered, total in sorted(per_file_rows):
        print(f"{name.ljust(max_name)}  {percent:6.1f}% ({covered}/{total})")


sys.exit(0)
