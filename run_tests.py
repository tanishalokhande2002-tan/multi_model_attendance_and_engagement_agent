"""
Standalone Test Runner for Multimodal Attendance & Engagement Agent.
Executes pytest across the test suite and displays formatted results.
Usage:
    python run_tests.py
"""

import sys
from pathlib import Path
import pytest

if __name__ == "__main__":
    print("=" * 70)
    print("Running Multimodal Attendance & Engagement Agent Test Suite")
    print("=" * 70)
    project_root = Path(__file__).resolve().parent
    tests_dir = project_root / "tests"

    exit_code = pytest.main([
        str(tests_dir),
        "-v",
        "--tb=short",
        "-W", "ignore::DeprecationWarning",
    ])
    print("=" * 70)
    if exit_code == 0:
        print("[SUCCESS] All unit and integration tests passed cleanly!")
    else:
        print(f"[FAIL] Some tests failed with exit code: {exit_code}")
    print("=" * 70)
    sys.exit(exit_code)
