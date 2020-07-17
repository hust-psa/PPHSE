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


import sys
import math
import gmpy2
import secrets
import numpy as np
import pandas as pd

from encryption.key import PaillierThresholdPublicKey
from estimation.util import pool_map


class EncodedNumber(object):
    """
    Transform integer and float numbers into integers for encryption
    """
    BASE = 16
    LOG2_BASE = math.log(BASE, 2)
    FLOAT_MANTISSA_BITS = sys.float_info.mant_dig

    def __init__(self, public_key, encoding, exponent):
        self.public_key, self.encoding, self.exponent = public_key, encoding, exponent
        pass

    @classmethod
    def encode(cls, public_key, scalar, exponent=None):
        if exponent is None:
            if isinstance(scalar, int):
                exponent = 0
            elif isinstance(scalar, float):
                bin_flt_exponent = math.frexp(scalar)[1]
                bin_lsb_exponent = bin_flt_exponent - cls.FLOAT_MANTISSA_BITS
                exponent = math.floor(bin_lsb_exponent / cls.LOG2_BASE)
            else:
                raise TypeError("Don't know the precision of type %s." % type(scalar))
        int_rep = gmpy2.mpz(round(scalar * pow(cls.BASE, -exponent)))
        if abs(int_rep) > public_key.max_int:
            raise ValueError('Integer needs to be within +/-%d but got %d' % (public_key.max_int, int_rep))
        return cls(public_key, int_rep % public_key.n, exponent)

    def decode(self):
        if self.encoding >= self.public_key.n:
            raise ValueError('Attempted to decode corrupted number.')
        elif self.encoding <= self.public_key.max_int:
            mantissa = self.encoding
        elif self.encoding >= self.public_key.n - self.public_key.max_int:
            mantissa = self.encoding - self.public_key.n
        else:
            raise OverflowError('Overflow detected in decrypted number.')
        return float(mantissa * pow(self.BASE, self.exponent))

    def decrease_exponent_to(self, new_exp):
        if new_exp > self.exponent:
            raise ValueError('New exponent %i should be more negative than old exponent %i' % (new_exp, self.exponent))
        factor = pow(self.BASE, self.exponent - new_exp)
        new_enc = self.encoding * factor % self.public_key.n
        self.exponent = new_exp
        self.encoding = new_enc
        pass


class EncryptedNumber(EncodedNumber):
    """
    Original Paillier threshold encrypted number
    """
    def __init__(self, public_key, encrypted, exponent):
        assert type(public_key) is PaillierThresholdPublicKey, "Fatal key provided."
        super().__init__(public_key, encrypted, exponent)
        pass

    def decrease_exponent_to(self, new_exp):
        if new_exp > self.exponent:
            raise ValueError('New exponent %i should be more negative than old exponent %i' % (new_exp, self.exponent))
        factor = pow(self.BASE, self.exponent - new_exp)
        new_enc = gmpy2.powmod(self.encoding, factor, self.public_key.n2)
        self.exponent = new_exp
        self.encoding = new_enc
        pass

    def __add__(self, other):
        assert isinstance(other, (int, float, EncodedNumber)), "Unsupported operant type <%s>" % type(other)
        other = other if isinstance(other, EncryptedNumber) else PaillierThreshold(self.public_key).encrypt(other)
        if self.exponent > other.exponent:
            self.decrease_exponent_to(other.exponent)
        elif self.exponent < other.exponent:
            other.decrease_exponent_to(self.exponent)
        if other.encoding < 0 or other.encoding >= self.public_key.n2:
            raise Exception("Operants should be in Z_{n^2}.")
        return EncryptedNumber(self.public_key, self.encoding * other.encoding % self.public_key.n2, self.exponent)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        assert isinstance(other, (int, float)) or type(other) is EncodedNumber,\
            "Unsupported factor type <%s>" % type(other)
        other = other if type(other) is EncodedNumber else EncodedNumber.encode(self.public_key, other)
        return EncryptedNumber(self.public_key,
                               gmpy2.powmod(self.encoding, other.encoding, self.public_key.n2),
                               self.exponent + other.exponent)

    def __rmul__(self, other):
        return self.__mul__(other)


class PaillierThreshold(object):

    def __init__(self, key):
        assert isinstance(key, PaillierThresholdPublicKey), "Fatal key provided."
        self.key = key
        self.can_decrypt = False if type(key) is PaillierThresholdPublicKey else True
        self._enc_type = EncryptedNumber
        pass

    def encrypt(self, m):
        if isinstance(m, (np.ndarray, pd.DataFrame, list)):
            return self._encrypt_array(m)
        elif isinstance(m, (int, float)) or type(m) is EncodedNumber:
            return self._encrypt_single(m)
        pass

    def _encrypt_single(self, m):
        assert isinstance(self.key, PaillierThresholdPublicKey), "Fatal key provided"
        assert isinstance(m, (int, float)) or type(m) is EncodedNumber, "Unsupported plaintext type <%s>" % type(m)
        m = m if type(m) is EncodedNumber else EncodedNumber.encode(self.key, m)
        if m.encoding < 0 or m.encoding >= self.key.n:
            raise Exception("Plaintext must be in Z_n.")
        while True:
            r = gmpy2.mpz_random(gmpy2.random_state(secrets.randbelow(sys.maxsize)), self.key.n)
            if gmpy2.gcd(r, self.key.n) == 1:
                break
        c = gmpy2.powmod(self.key.n + 1, m.encoding, self.key.n2) * \
            gmpy2.powmod(r, self.key.n, self.key.n2) % self.key.n2
        public_key = self.key if type(self.key) is PaillierThresholdPublicKey else self.key.getPublicKey()
        return EncryptedNumber(public_key, c, m.exponent)

    def _encrypt_array(self, arr):
        res = np.array(arr, dtype=np.object) if not isinstance(arr, pd.DataFrame) else arr.to_numpy().astype(np.object)
        shape = res.shape
        raw = res.flatten()
        enc = pool_map(self._encrypt_single, raw)
        enc = np.array(enc, dtype=np.object).reshape(shape)
        return enc if not isinstance(arr, pd.DataFrame) else pd.DataFrame(enc, index=arr.index, columns=arr.columns)

    def decrypt(self, c):
        if isinstance(c, (np.ndarray, pd.DataFrame, list)):
            return self._decrypt_array(c)
        elif isinstance(c, EncryptedNumber):
            return self._decrypt_single(c)
        pass

    def _decrypt_single(self, c):
        assert c is None or isinstance(c, EncryptedNumber), "Invalid ciphertext provided."
        assert self.can_decrypt, "Cannot conduct decryption because of lack of private key."
        return PartialDecryption(self.key, c) if c else None

    def _decrypt_array(self, arr):
        res = np.array(arr, dtype=np.object) if not isinstance(arr, pd.DataFrame) else arr.to_numpy().astype(np.object)
        shape = res.shape
        raw = res.flatten()
        dec = pool_map(self._decrypt_single, raw)
        dec = np.delete(np.array(dec + [None], dtype=np.object), -1, axis=0).reshape(shape)
        return dec if not isinstance(arr, pd.DataFrame) else pd.DataFrame(dec, index=arr.index, columns=arr.columns)

    def combine_shares(self, c, shares):
        assert len(shares) >= self.key.w, "Can only combine more than w shares to fully decrypt a ciphertext."
        if isinstance(c, (list, np.ndarray, pd.DataFrame)):
            return self._combine_shares_array(c, shares)
        else:
            return self._combine_shares_single(c, shares)
        pass

    def _combine_shares_single(self, c, shares):
        if c is None:
            return 0
        cprime = gmpy2.mpz(1)
        for i in range(self.key.w):
            lambda_ = self.key.delta
            for i1 in range(self.key.w):
                if i1 != i:
                    if shares[i].id != shares[i1].id:
                        lambda_ = lambda_ * (-shares[i1].id) / (shares[i].id - shares[i1].id)
                    else:
                        raise Exception("Cannot have repeated shares.")
            shr = gmpy2.invert(shares[i].decryption, self.key.n2) if lambda_ < 0 else shares[i].decryption
            cprime = cprime * gmpy2.powmod(shr, 2 * gmpy2.mpz(abs(lambda_)), self.key.n2) % self.key.n2
        L = (cprime - 1) // self.key.n
        val = L * self.key.combineShareConstant % self.key.n
        return EncodedNumber(self.key, val, c.exponent).decode()

    def _combine_shares_array(self, c, shares):
        shares = np.array([share.flatten().tolist() + [None]
                           if not isinstance(share, pd.DataFrame)
                           else share.to_numpy().flatten().tolist() + [None] for share in shares]).T
        shares = np.delete(shares, -1, 0)
        c_arr = np.array(c, dtype=np.object) if not isinstance(c, pd.DataFrame) else c.to_numpy().astype(np.object)
        shape = c_arr.shape
        c_raw = c_arr.flatten().tolist()
        if not isinstance(c_raw, list):
            c_raw = [c_raw]
        param = list(zip(c_raw, shares))
        res = pool_map(self._combine_shares_single, param, star=True)
        res = np.array(res, dtype=np.object).reshape(shape)
        return res.astype(np.float) if not isinstance(c, pd.DataFrame) \
            else pd.DataFrame(res, index=c.index, columns=c.columns, dtype=np.float)


class PartialDecryption(object):
    def __init__(self, private_key, c):
        if c.encoding < 0 or c.encoding >= private_key.n2:
            raise Exception("Ciphertext must be in Z_{n^2}.")
        self.decryption = gmpy2.powmod(c.encoding, 2 * private_key.si * private_key.delta, private_key.n2)
        self.id = private_key.id
        pass
