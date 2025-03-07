import unittest
import os

# List of test modules to include

test_modules = [
    "test_basic_tables",
    "test_dataset",
    "test_download",
    "test_execution",
    "test_upload",
    "test_features",
]

os.environ["DERIVA_PY_TEST_HOSTNAME"] = "dev.eye-ai.org"


def load_tests():
    """Loads test cases from specified test modules."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for module in test_modules:
        # Load test cases from each module
        tests = loader.loadTestsFromName(module)
        suite.addTests(tests)

    return suite


if __name__ == "__main__":
    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = load_tests()
    runner.run(test_suite)
