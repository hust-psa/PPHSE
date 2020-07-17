# Copyright 2020 Jingyu Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import pickle
from itertools import starmap
from multiprocessing import Pool

from pandas import DataFrame
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from sklearn.metrics import mean_squared_error as mse

PICKLE_PATH = 'estimation/networks/pickled/'
HEADER = os.urandom(6)
CIPHER = Cipher(algorithms.AES(os.urandom(16)), modes.CFB(os.urandom(16)), backend=default_backend())
HMAC_KEY = os.urandom(32)
PARALLEL_POOL_NUM = 32


def pool_map(*args, **kwargs):
    star = kwargs.pop('star', False)
    if PARALLEL_POOL_NUM == 1:
        return list(map(*args, **kwargs) if not star else starmap(*args, **kwargs))
    elif PARALLEL_POOL_NUM > 1:
        pool = Pool(PARALLEL_POOL_NUM)
        res = pool.map(*args, **kwargs) if not star else pool.starmap(*args, **kwargs)
        pool.close()
        pool.join()
        return res
    pass


def reset_index(df: DataFrame):
    df.name = df.index
    return df.reset_index(drop=True)


def update_index(df: DataFrame, col_names, lookups):
    # update <colnames> values with their indices in <lookups>
    if type(col_names) is str:
        col_names = [col_names]
    for col in col_names:
        assert col in df.columns, "The column <%s> must be contained by the provided data frame." % col
        df[col] = [lookups.index(name) for name in list(df[col])]
    return df


def rmse(true, pred):
    if true is None or pred is None:
        return float('nan')
    else:
        return mse(true, pred, squared=False)


def enc_sign(data):
    byte_data = pickle.dumps(data)
    enc = CIPHER.encryptor()
    cipher = enc.update(byte_data) + enc.finalize()
    mac_gen = hmac.HMAC(HMAC_KEY, hashes.SHA256(), backend=default_backend())
    mac_gen.update(HEADER + cipher)
    digest = mac_gen.finalize()
    return cipher, digest


def veri_dec(enc_sign_data):
    cipher, digest = enc_sign_data
    mac_ver = hmac.HMAC(HMAC_KEY, hashes.SHA256(), backend=default_backend())
    mac_ver.update(HEADER + cipher)
    mac_ver.verify(digest)
    dec = CIPHER.decryptor()
    byte_data = dec.update(cipher) + dec.finalize()
    return pickle.loads(byte_data)


def comm_delay(data, bitrate=1e9, latency=0):
    data_amount = len(HEADER)
    for d in data:
        data_amount += len(d)
    return data_amount * 8 / bitrate + latency


class ParaFuncs(object):
    """
    Define the functions used for parallel computation
    """

    @staticmethod
    def bnd_flow_h_H(x):
        return [x[1] * x[2] + x[0] * x[3],
                x[0] * x[2] - x[1] * x[3]]

    @staticmethod
    def enc_Gi_ri(tau, x):
        return tau * x if x != 0 else None

    @staticmethod
    def sum_Gi_ri(x):
        x = list(filter(None, x))
        return sum(x) if len(x) > 0 else None
