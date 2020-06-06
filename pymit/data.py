import os
import shutil
import zipfile

import numpy as np
import requests


def madelon():
    ARCHIVE = 'MADELON.zip'
    URL = 'http://clopinet.com/isabelle/Projects/NIPS2003/MADELON.zip'
    FOLDER = 'MADELON'

    if not os.path.exists(ARCHIVE):
        with requests.get(URL, stream=True) as r:
            with open(ARCHIVE, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

    shutil.rmtree(FOLDER, ignore_errors=True)

    with zipfile.ZipFile(ARCHIVE, 'r') as z:
        z.extractall(FOLDER)

    data = np.loadtxt(os.path.join(FOLDER, 'MADELON', 'madelon_train.data'), dtype=np.int)
    labels = np.loadtxt(os.path.join(FOLDER, 'MADELON', 'madelon_train.labels'), dtype=np.int)

    return data, labels
    