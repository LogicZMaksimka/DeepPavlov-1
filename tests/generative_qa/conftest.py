import pytest

def pytest_addoption(parser):
    parser.addoption("--coqa_path", action="store")

@pytest.fixture
def coqa_path(request):
    return request.config.getoption("--coqa_path")