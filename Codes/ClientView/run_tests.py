from tests.test_models import *
from tests.test_ap_controller import *
from tests.test_logger import *

def main():
    print("\nRunning all model tests...")
    test_all_models()

    print("\nRunning all Ap Control tests...")
    test_all_ap_control()

    print("\nRunning the tests for Logger")
    test_all_logger()

    print("All tests completed")

if __name__ == "__main__":
    main()