import pytest

import pymit.data


@pytest.fixture(scope='session')
def madelon():
    return pymit.data.madelon()
