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


import secrets
from encryption.paillierth import EncodedNumber, EncryptedNumber, PaillierThreshold
from encryption.key import PaillierThresholdPublicKey


class Deg2EncryptedNumber(object):
    """
    Degree-2 Paillier threshold encrypted number
    """
    def __init__(self, public_key, a=None, b=None, alpha=None, beta=None, enc_one=None):
        assert type(public_key) is PaillierThresholdPublicKey, "Fatal key provided."
        assert ((isinstance(a, (int, float, EncodedNumber)) and isinstance(b, EncryptedNumber)) or
                (isinstance(alpha, EncryptedNumber) and isinstance(beta, list))), "Invalid ciphertext provided."
        self.public_key = public_key
        self.enc_one = enc_one
        self.a_raw = a
        self.a = a if a is None else EncodedNumber.encode(self.public_key, a)
        self.b, self.alpha, self.beta = b, alpha, beta
        pass

    @property
    def is_deg2(self):
        return self.beta is not None

    def __add__(self, other):
        assert isinstance(other, (int, float)) or type(other) in [EncodedNumber, Deg2EncryptedNumber], \
            "Unsupported operant type <%s>" % type(other)
        other = other if type(other) is Deg2EncryptedNumber else Deg2PaillierThreshold(self.public_key).encrypt(other)
        if not self.is_deg2 and not other.is_deg2:
            a = self.a_raw + other.a_raw
            b = self.b + other.b
            return Deg2EncryptedNumber(self.public_key, a=a, b=b, enc_one=self.enc_one)
        else:
            oprd1 = self if self.is_deg2 else self * self.enc_one
            oprd2 = other if other.is_deg2 else other * self.enc_one
            return Deg2EncryptedNumber(self.public_key, enc_one=self.enc_one,
                                       alpha=oprd1.alpha + oprd2.alpha, beta=oprd1.beta + oprd2.beta)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        assert isinstance(other, (int, float)) or type(other) in [EncodedNumber, Deg2EncryptedNumber], \
            "Unsupported operant type <%s>" % type(other)
        if type(other) is Deg2EncryptedNumber:
            if self.is_deg2 or other.is_deg2:
                raise Exception("Cannot multiply Level-2 ciphertext.")
            alpha = PaillierThreshold(self.public_key).encrypt(self.a_raw * other.a_raw)
            beta = [[self.b, other.b]]
            return Deg2EncryptedNumber(self.public_key, enc_one=self.enc_one,
                                       alpha=(alpha + other.b * self.a + self.b * other.a), beta=beta)
        else:
            other = other.decode() if isinstance(other, EncodedNumber) else other
            if not self.is_deg2:
                a = self.a_raw * other
                b = self.b * other
                return Deg2EncryptedNumber(self.public_key, enc_one=self.enc_one, a=a, b=b)
            else:
                scaled_beta = []
                for beta in self.beta:
                    scaled_beta.append([beta[0] * other, beta[1]])
                return Deg2EncryptedNumber(self.public_key, enc_one=self.enc_one,
                                           alpha=self.alpha * other, beta=scaled_beta)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return other - self

    def __truediv__(self, other):
        assert isinstance(other, (int, float)) or type(other) is EncodedNumber, \
            "Unsupported operant type <%s>" % type(other)
        factor = 1 / other.decode() if type(other) is EncodedNumber else 1 / other
        return self.__mul__(factor)


class Deg2PaillierThreshold(PaillierThreshold):
    def __init__(self, key):
        super().__init__(key)
        self._enc_type = Deg2EncryptedNumber
        self.enc_one = self._encrypt_single(1, init=True)
        pass

    def _encrypt_single(self, m, init=False):
        m = m if type(m) is not EncodedNumber else m.decode()
        b = secrets.SystemRandom().uniform(0, m)
        a = m - b
        b = EncodedNumber.encode(self.key, b)
        public_key = self.key if type(self.key) is PaillierThresholdPublicKey else self.key.getPublicKey()
        return (Deg2EncryptedNumber(public_key, a=a, b=PaillierThreshold(public_key).encrypt(b)) if init
                else Deg2EncryptedNumber(public_key, enc_one=self.enc_one,
                                         a=a, b=PaillierThreshold(public_key).encrypt(b)))

    def decrypt(self, c):
        if isinstance(c, Deg2EncryptedNumber):
            return self._decrypt_single(c)
        else:
            return super().decrypt(c)
        pass

    def _decrypt_single(self, c):
        assert c is None or isinstance(c, Deg2EncryptedNumber), "Invalid ciphertext provided."
        assert self.can_decrypt, "Cannot conduct decryption because of lack of private key."
        if c is None:
            return None
        if not c.is_deg2:
            return super()._decrypt_single(c.b)
        else:
            shares = [super()._decrypt_single(c.alpha)]
            for b in c.beta:
                shares += [super()._decrypt_single(b[0]), super()._decrypt_single(b[1])]
            return shares

    def _combine_shares_single(self, c, shares):
        assert c is None or isinstance(c, Deg2EncryptedNumber), "Invalid ciphertext provided."
        if c is None:
            return 0.
        if not c.is_deg2:
            return c.a_raw + super()._combine_shares_single(c.b, shares)
        else:
            parts = list(map(list, zip(*shares)))
            final = []
            for i in range(len(parts)):
                cipher = c.alpha if i == 0 else c.beta[(i - 1) // 2][(i - 1) % 2]
                final.append(super()._combine_shares_single(cipher, parts[i]))
            result = final[0]
            for i in range(1, len(final), 2):
                result += final[i] * final[i + 1]
            return result
