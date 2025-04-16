# tests/conftest.py
import time
import pytest

@pytest.fixture(autouse=True)
def delay_between_tests():
    yield
    delay_seconds = 30
    print(f"\nWaiting {delay_seconds} seconds before the next test to avoid rate limit issues...")
    time.sleep(delay_seconds)
