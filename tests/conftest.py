import os
import shutil
import zipfile

import numpy as np
import pytest
import requests

import pymit.data


@pytest.fixture(scope='session')
def madelon():
    return pymit.data.madelon()