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


import time
import numpy as np

from estimation.util import ParaFuncs, pool_map, enc_sign, veri_dec, comm_delay


def run_pphse(ips_net, secure=True, key_len=2048, fast_decoupled=False,
              tolerance=1e-4, max_iter=30, verbose=True):
    if verbose:
        prefix = 'PP ' if secure else ''
        prefix += 'FD ' if fast_decoupled else ''
        print("%sHSE%s:" % (prefix, " (%d-bit keys)" % key_len if secure else ''))
    try:
        # initialize HCC and LCCs
        assert ips_net.has_meas, "Please add measurements first."
        hcc, lccs = ips_net.partition_by_zone(secure, key_len, fast_decoupled)
        n_all_bus = 0
        for lcc in lccs.values():
            n_all_bus += len(lcc.net.bus)

        # initialize result containers and convergence criterion
        max_error = [np.inf] * len(lccs)
        global_v, global_delta = np.zeros(n_all_bus), np.zeros(n_all_bus)

        # start iteration
        iter_idx = 0
        total_time_consumption = 0
        while max(max_error) > tolerance and iter_idx < max_iter:
            iter_idx += 1
            max_error = []

            # time check 1: start
            time_consumption = 0

            execution_time = []
            cipher_L2H = []
            for idx, lcc in lccs.items():
                tic = time.perf_counter()
                # LCC computes delta_yi
                lcc.compute_delta_yi()
                # LCC reports intermediate terms for computing bnd_flow_hi and bnd_flow_Hi
                data = lcc.report_bnd_flow_terms(hcc.bnd_flow_terms)
                # encrypt and sign the reported data messages
                enc_data = enc_sign(data)
                toc = time.perf_counter()
                cipher_L2H.append(enc_data)
                execution_time.append(toc - tic + comm_delay(enc_data))

            # time check 2: all LCCs finish reporting data
            time_consumption += max(execution_time)

            tic = time.perf_counter()
            for enc_data in cipher_L2H:
                # decrypt and verify the received data
                veri_dec(enc_data)
            # HCC combines the received intermediate terms
            bnd_flow_h_H = hcc.compute_bnd_flow_h_H()
            cipher_H2L = enc_sign(bnd_flow_h_H)
            toc = time.perf_counter()

            # time check 3: all LCCs receive bnd_flow_hi and bnd_flow_Hi
            time_consumption += (toc - tic + comm_delay(cipher_H2L))

            # LCCs finally compute their corresponding bnd_flow_hi and bnd_flow_Hi
            execution_time = dict()
            if secure:
                # each LCC decrypts the received data and sends the shares to others
                shares, t_receive_share, enc_shares = dict(), dict(), dict()
                t_finish_compute_share = dict()
                for idx, lcc in lccs.items():
                    shares[idx], t_receive_share[idx], enc_shares[idx] = [], [], []
                for idx, lcc in lccs.items():
                    execution_time[idx] = 0
                    tic = time.perf_counter()
                    veri_dec(cipher_H2L)
                    for idx_tmp, lcc_tmp in lccs.items():
                        res = bnd_flow_h_H[idx_tmp].copy()
                        if len(res) > 0:
                            share = lcc.enc.decrypt(res)
                            if idx_tmp != idx:
                                enc_share = enc_sign(share)
                                tmp_toc = time.perf_counter()
                                enc_shares[idx_tmp].append(enc_share)
                                t_receive_share[idx_tmp].append(tmp_toc - tic + comm_delay(enc_share))
                            shares[idx_tmp].append(share)
                    toc = time.perf_counter()
                    t_finish_compute_share[idx] = (toc - tic)

                # wait until all LCCs receive the shares
                for idx, lcc in lccs.items():
                    last_time = max(t_receive_share[idx] + [t_finish_compute_share[idx]])
                    execution_time[idx] += last_time

                # LCCs combine shares to obtain plaintexts
                for idx, lcc in lccs.items():
                    tic = time.perf_counter()
                    for enc_data in enc_shares[idx]:
                        veri_dec(enc_data)
                    res = bnd_flow_h_H[idx].copy()
                    res = lcc.enc.combine_shares(res, shares[idx])
                    bnd_flow_h_H[idx] = res
                    toc = time.perf_counter()
                    execution_time[idx] += (toc - tic)

            # LCCs use the received data to compute their own bnd_flow_hi and bnd_flow_Hi
            bnd_Gi, bnd_ri = dict(), dict()
            for idx, lcc in lccs.items():
                execution_time[idx] = 0
                tic = time.perf_counter()
                veri_dec(cipher_H2L)
                res = bnd_flow_h_H[idx].copy()
                bnd_flow_hi = lcc.final_bnd_flow_hi(res)
                if not fast_decoupled:
                    bnd_flow_Hi = lcc.final_bnd_flow_Hi(res)
                else:
                    bnd_flow_Hi = lcc.bnd_flow_Hi

                # LCCs compute bnd_hi, bnd_Hi, bnd_Gi, and bnd_ri
                lcc.compute_bnd_hi_Hi(bnd_flow_hi, bnd_flow_Hi)
                bGi = lcc.compute_bnd_Gi()
                bri = lcc.compute_bnd_ri()
                toc = time.perf_counter()
                execution_time[idx] += (toc - tic)
                bnd_Gi[idx] = bGi
                bnd_ri[idx] = bri

            # LCCs report tau_i if <secure> is True
            if secure:
                tau, cipher_L2H = [], []
                t_receive_tau = dict()
                for idx, lcc in lccs.items():
                    tic = time.perf_counter()
                    tau_i = lcc.enc.encrypt(np.random.rand())
                    enc_tau_i = enc_sign(tau_i)
                    toc = time.perf_counter()
                    tau.append(tau_i)
                    cipher_L2H.append(enc_tau_i)
                    execution_time[idx] += (toc - tic)
                    t_receive_tau[idx] = execution_time[idx] + comm_delay(enc_tau_i)

                # HCC waits for collecting all tau_i
                t_receive_tau = max(t_receive_tau.values())
                tic = time.perf_counter()
                for enc_data in cipher_L2H:
                    veri_dec(enc_data)
                tau = sum(tau)
                cipher_H2L = enc_sign(tau)
                toc = time.perf_counter()
                t_receive_tau = t_receive_tau + (toc - tic) + comm_delay(cipher_H2L)

                # LCCs securely multiply the sum of tau_i to their corresponding bnd_Gi and bnd_ri
                for idx, lcc in lccs.items():
                    execution_time[idx] = t_receive_tau
                    bGi, bri = bnd_Gi[idx].toarray(), bnd_ri[idx].toarray()
                    tic = time.perf_counter()
                    bGi_raw, bri_raw = bGi.flatten().tolist(), bri.flatten().tolist()
                    res = pool_map(ParaFuncs.enc_Gi_ri, list(zip([tau] * len(bGi_raw), bGi_raw)), star=True)
                    bGi = np.array(res).reshape(bGi.shape)
                    res = pool_map(ParaFuncs.enc_Gi_ri, list(zip([tau] * len(bri_raw), bri_raw)), star=True)
                    bri = np.array(res).reshape(bri.shape)
                    enc_data = enc_sign([bGi, bri])
                    toc = time.perf_counter()
                    cipher_L2H.append(enc_data)
                    bnd_Gi[idx] = bGi
                    bnd_ri[idx] = bri
                    execution_time[idx] += (toc - tic + comm_delay(enc_data))

            # time check 4: HCC receives all reported bnd_Gi and bnd_ri
            time_consumption += max(execution_time.values())

            # HCC combines all bnd_Gi and bnd_ri (equivalent to computing lambda)
            tic = time.perf_counter()
            bnd_G, bnd_r = hcc.compute_lambda(bnd_Gi, bnd_ri)
            cipher_H2L = enc_sign([bnd_G, bnd_r])
            toc = time.perf_counter()

            # time check 5: LCCs receive tau_sum_G and tau_sum_r
            time_consumption += (toc - tic + comm_delay(cipher_H2L))

            # LCCs decrypt the received data and exchange the shares
            if secure:
                execution_time = []
                exchange_share = None
                bnd_G_shares, bnd_r_shares = [], []
                for idx, lcc in lccs.items():
                    tic = time.perf_counter()
                    veri_dec(cipher_H2L)
                    share_bnd_G = lcc.enc.decrypt(bnd_G)
                    share_bnd_r = lcc.enc.decrypt(bnd_r)
                    exchange_share = enc_sign([share_bnd_G, share_bnd_r])
                    toc = time.perf_counter()
                    bnd_G_shares.append(share_bnd_G)
                    bnd_r_shares.append(share_bnd_r)
                    execution_time.append(toc - tic + comm_delay(exchange_share))

                # time check 6: LCCs should basically receive these shares at the same time
                time_consumption += max(execution_time)
                c_bnd_G, c_bnd_r = bnd_G.copy(), bnd_r.copy()
                execution_time = []
                for idx, lcc in lccs.items():
                    tic = time.perf_counter()
                    veri_dec(exchange_share)
                    bnd_G = lcc.enc.combine_shares(c_bnd_G, bnd_G_shares)
                    bnd_r = lcc.enc.combine_shares(c_bnd_r, bnd_r_shares)
                    toc = time.perf_counter()
                    execution_time.append(toc - tic)
                time_consumption += max(execution_time)
            else:
                bnd_G[bnd_G == None] = 0
                bnd_G = bnd_G.astype(np.float)
                bnd_r[bnd_r == None] = 0
                bnd_r = bnd_r.astype(np.float)

            execution_time = []
            for idx, lcc in lccs.items():
                tic = time.perf_counter()
                # step 3: compute ui
                lcc.compute_ui(bnd_G, bnd_r)
                # step 4: update xi
                delta_xi = lcc.compute_delta_xi()
                lcc.E += delta_xi
                lcc.eppci.update_E(lcc.E)
                max_error.append(max(abs(delta_xi)))
                toc = time.perf_counter()
                execution_time.append(toc - tic)

            # time check 7: finish
            time_consumption += max(execution_time)
            total_time_consumption += time_consumption

            if verbose:
                print("Maximum state variable increment after Iteration %d: %g" % (iter_idx, max(max_error)))

        if max(max_error) > tolerance:
            raise RuntimeError("%s fails to converge.")
        else:
            # return final global v, delta, and the measured time consumption
            for idx, lcc in lccs.items():
                lcc.save_result(global_v, global_delta)
            return global_v, global_delta, total_time_consumption
    except (AssertionError, RuntimeError) as e:
        print('An error occurred during computation: %s' % e)
        return None, None, float('inf')
