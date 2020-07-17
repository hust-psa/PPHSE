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
import sys
import secrets
import time
import json
import math
import gmpy2


class PaillierThresholdPublicKey(object):
    def __init__(self, n, n_party, n_threshold, combineShareConstant):
        self.n, self.l, self.w, self.combineShareConstant = n, n_party, n_threshold, combineShareConstant
        self.n2 = n ** 2
        self.max_int = n // 3 - 1
        self.k = n.bit_length()
        self.delta = gmpy2.fac(n_party)
        pass


class PaillierThresholdPrivateKey(PaillierThresholdPublicKey):
    def __init__(self, n, n_party, n_threshold, combineShareConstant, party_id, si):
        super().__init__(n, n_party, n_threshold, combineShareConstant)
        self.id, self.si = party_id, si
        pass

    def getPublicKey(self):
        return PaillierThresholdPublicKey(self.n, self.l, self.w, self.combineShareConstant)


def gen_key(bitlen, n_party, n_threshold, saveToFile=False, verbose=False):
    assert math.log2(bitlen) >= 8, "Bit-length of n should not be less than 256."
    bitlen >>= 1
    if verbose:
        print("Start generating public and private keys for threshold Paillier ...", end=" ")
    res = {}

    def __getSafePrimes(n_len):
        rng = secrets.SystemRandom()
        prime_ = gmpy2.mpz(rng.getrandbits(n_len - 1))
        prime_ = gmpy2.bit_set(prime_, n_len - 2)
        while True:
            prime_ = gmpy2.next_prime(prime_)
            prime = 2 * prime_ + 1
            if gmpy2.is_prime(prime, 25):
                break
        return prime_, prime

    start = time.process_time()
    p1, p = __getSafePrimes(bitlen)
    while True:
        q1, q = __getSafePrimes(bitlen)
        if p1 != q1 and p1 != q and q1 != p:
            break
    n, m = p * q, p1 * q1
    nm = n * m
    d = m * gmpy2.invert(m, n)
    a = [d]
    a = a + [gmpy2.mpz_random(gmpy2.random_state(secrets.randbelow(sys.maxsize)), nm) for i in range(1, n_threshold)]
    delta = gmpy2.fac(n_party)
    combineShareConstant = gmpy2.invert((4 * delta ** 2) % n, n)
    shares = [gmpy2.mpz(0)] * n_party
    for index in range(n_party):
        fx = [a[i] * ((index + 1) ** i) for i in range(n_threshold)]
        shares[index] += sum(fx) % nm
    for i in range(n_party):
        res[i + 1] = PaillierThresholdPrivateKey(n, n_party, n_threshold, combineShareConstant, i + 1, shares[i])
    end = time.process_time()

    if verbose:
        print("finished. Time elapsed: %g seconds." % (end - start))
    if saveToFile:
        if verbose:
            print("Saving keys to files ...")
        keys = {'n': str(n), 'l': n_party, 'w': n_threshold, 'combineShareConstant': str(combineShareConstant)}
        for i in range(n_party):
            keys.update({'si': str(shares[i]), 'id': i + 1})
            if not os.path.isdir("encryption/keys/"):
                os.mkdir("encryption/keys/")
            with open("encryption/keys/" + str(i + 1) + ".key", 'w') as f:
                json.dump(keys, f)
    return res


def load_key(keyid, path="encryption/keys/"):
    if not os.access(path, os.R_OK):
        raise Exception("Cannot read the key from the given file: %s." % path)
    with open(path + str(keyid) + ".key", 'r') as f:
        keys = json.load(f)
        return PaillierThresholdPrivateKey(gmpy2.mpz(keys['n']), keys['l'], keys['w'],
                                           gmpy2.mpz(keys['combineShareConstant']), keys['id'], gmpy2.mpz(keys['si']))
