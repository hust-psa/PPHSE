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
import gzip
import pickle
import copy
import pandas as pd
import warnings

from prettytable import PrettyTable
from estimation.util import rmse, PICKLE_PATH, PARALLEL_POOL_NUM
from estimation.pphse import run_pphse
from estimation.cse import run_cse
from estimation.networks.prepare_networks import InterPowerSystem
from estimation.networks.typical_networks import case118, case300, case2869, get_meas_info

warnings.filterwarnings('ignore')


def accuracy_test(tolerance=1e-4, max_iteration=30):
    # initialize an interconnected power system model without measurements
    case = case300()
    network = InterPowerSystem(case)

    # add measurements to the model
    meas_dfs = dict()
    path = PICKLE_PATH + 'accuracy.pkl'
    if os.access(path, os.R_OK):
        with gzip.open(path, 'rb') as f:
            meas_dfs = pickle.load(f)
    else:
        blank_meas_df = network.net.measurement.copy()
        target = ['boundary', 'internal']
        redundancy = ['high', 'low']
        for (t, r) in [(x, y) for x in target for y in redundancy]:
            meas_info = get_meas_info(target=t, redundancy=r)
            for meas in meas_info:
                side = meas.get('side', '')
                network.add_measurement(meas['meas_type'], meas['element_type'],
                                        meas['element'], side, meas['std_dev'])
            meas_dfs[t + '_' + r] = network.net.measurement.copy()
            network.net.measurement = blank_meas_df.copy()
        with gzip.open(path, 'wb') as f:
            pickle.dump(meas_dfs, f)

    # test accuracy with different measurement allocation schemes
    print("=============================================")
    print("                Accuracy Tests               ")
    print("      (RMSE w.r.t. power flow results)       ")
    print("---------------------------------------------")
    print("Using: %s\n" % case['id'])

    boundary = ['high', 'low']
    internal = ['high', 'low']
    for (b, i) in [(x, y) for x in boundary for y in internal]:
        bnd_meas_df = meas_dfs['boundary_' + b]
        int_meas_df = meas_dfs['internal_' + i]
        meas_df = pd.concat([bnd_meas_df, int_meas_df], ignore_index=True)
        network.net.measurement = meas_df
        network.has_meas = True
        print("Redundancy: boundary - %s, internal - %s" % (b, i))
        table = PrettyTable()
        table.field_names = ['', 'CSE', 'HSE', 'PPHSE']
        table.align[''] = 'l'
        table.align['CSE'] = 'r'
        table.align['HSE'] = 'r'
        table.align['PPHSE'] = 'r'
        rmse_v, rmse_theta = ['RMSE of v'], ['RMSE of theta']
        # set power flow calculation results as benchmark
        pf_result = network.net.res_bus.sort_index()
        pf_v, pf_theta = pf_result.vm_pu.to_numpy().flatten(), pf_result.va_degree.to_numpy().flatten()
        # accuracy of CSE
        cse_v, cse_theta, _ = run_cse(network.net, tolerance=tolerance, max_iter=max_iteration)
        rmse_v.append('%g' % rmse(cse_v, pf_v))
        rmse_theta.append('%g' % rmse(cse_theta, pf_theta))
        # accuracy of HSE
        hse_net = copy.deepcopy(network)
        hse_v, hse_theta, _ = run_pphse(hse_net, secure=False, fast_decoupled=False, tolerance=tolerance,
                                        max_iter=max_iteration, verbose=False)
        rmse_v.append('%g' % rmse(hse_v, pf_v))
        rmse_theta.append('%g' % rmse(hse_theta, pf_theta))
        # accuracy of PPHSE
        pphse_net = copy.deepcopy(network)
        pphse_v, pphse_theta, _ = run_pphse(pphse_net, secure=True, fast_decoupled=False, tolerance=tolerance,
                                            max_iter=max_iteration, verbose=False)
        rmse_v.append('%g' % rmse(pphse_v, pf_v))
        rmse_theta.append('%g' % rmse(pphse_theta, pf_theta))

        # print results
        table.add_row(rmse_v)
        table.add_row(rmse_theta)
        print(table)
    pass


def efficiency_tests(tolerance=1e-4, max_iteration=30):
    # initialize an interconnected power system model without measurements
    case = case300()
    network = InterPowerSystem(case)

    # add measurements to the model
    meas_dfs = dict()
    path = PICKLE_PATH + 'efficiency.pkl'
    if os.access(path, os.R_OK):
        with gzip.open(path, 'rb') as f:
            meas_dfs = pickle.load(f)
    else:
        blank_meas_df = network.net.measurement.copy()
        target = ['boundary', 'internal']
        redundancy = ['high', 'low']
        for (t, r) in [(x, y) for x in target for y in redundancy]:
            meas_info = get_meas_info(target=t, redundancy=r)
            for meas in meas_info:
                side = meas.get('side', '')
                network.add_measurement(meas['meas_type'], meas['element_type'],
                                        meas['element'], side, meas['std_dev'])
            meas_dfs[t + '_' + r] = network.net.measurement.copy()
            network.net.measurement = blank_meas_df.copy()
        with gzip.open(path, 'wb') as f:
            pickle.dump(meas_dfs, f)

    # test efficiency with different measurement allocation schemes and different key lengths
    print("=============================================")
    print("               Efficiency Tests              ")
    print("---------------------------------------------")
    print("Using: %s" % case['id'])
    print("Multiprocessing core(s): %d\n" % PARALLEL_POOL_NUM)

    boundary = ['high', 'low']
    internal = ['high', 'low']
    key_len = [512, 768, 1024, 1536, 2048]
    for (b, i) in [(x, y) for x in boundary for y in internal]:
        bnd_meas_df = meas_dfs['boundary_' + b]
        int_meas_df = meas_dfs['internal_' + i]
        meas_df = pd.concat([bnd_meas_df, int_meas_df], ignore_index=True)
        network.net.measurement = meas_df
        network.has_meas = True
        print("Redundancy: boundary - %s, internal - %s" % (b, i))
        table = PrettyTable()
        table.field_names = ['', '-'] + list(map(str, key_len))
        table.align[''] = 'l'
        table.align['-'] = 'r'
        for k_l in map(str, key_len):
            table.align[k_l] = 'r'
        time = ['Time (s)']
        # efficiency of HSE
        hse_net = copy.deepcopy(network)
        _, _, t = run_pphse(hse_net, secure=False, fast_decoupled=False, tolerance=tolerance,
                            max_iter=max_iteration, verbose=False)
        time.append('%g' % t)
        for k in key_len:
            # efficiency of PPHSE using k-bit keys
            pphse_net = copy.deepcopy(network)
            _, _, t = run_pphse(pphse_net, key_len=k, fast_decoupled=False, tolerance=tolerance,
                                max_iter=max_iteration, verbose=False)
            time.append('%g' % t)

        # print results
        table.add_row(time)
        print(table)
    pass


def scalability_test(tolerance=1e-4, max_iteration=30):
    # initialize interconnected power system models without measurements
    # cases = [case118(), case300(), case2869()]
    cases = [case2869()]
    print("=============================================")
    print("              Scalability Tests              ")
    print("      (RMSE w.r.t. power flow results)       ")
    print("---------------------------------------------")
    print("Multiprocessing core(s): %d\n" % PARALLEL_POOL_NUM)
    # add measurements to the model
    meas_dfs = dict()
    path = PICKLE_PATH + 'scalability.pkl'
    if os.access(path, os.R_OK):
        with gzip.open(path, 'rb') as f:
            meas_dfs = pickle.load(f)
    for case in cases:
        network = InterPowerSystem(case)
        if case['id'] not in meas_dfs:
            for meas_info in case['default_meas']:
                side = meas_info.get('side', '')
                network.add_measurement(meas_info['meas_type'], meas_info['element_type'],
                                        meas_info['element'], side, meas_info['std_dev'])
            meas_dfs[case['id']] = network.net.measurement.copy()
            with gzip.open(path, 'wb') as f:
                pickle.dump(meas_dfs, f)
        else:
            network.net.measurement = meas_dfs[case['id']]
            network.has_meas = True

        # test scalability of different models
        print("Current model: %s" % case['id'])
        network.print_statistics()
        table = PrettyTable()
        table.field_names = ['', 'CSE', 'HSE', 'PPHSE']
        table.align[''] = 'l'
        table.align['CSE'] = 'r'
        table.align['HSE'] = 'r'
        table.align['PPHSE'] = 'r'
        rmse_v, rmse_theta, time = ['RMSE of v'], ['RMSE of theta'], ['Time (s)']

        # set power flow calculation results as benchmark
        pf_result = network.net.res_bus.sort_index()
        pf_v, pf_theta = pf_result.vm_pu.to_numpy().flatten(), pf_result.va_degree.to_numpy().flatten()
        # CSE
        cse_v, cse_theta, t = run_cse(network.net, tolerance=tolerance, max_iter=max_iteration)
        rmse_v.append('%g' % rmse(cse_v, pf_v))
        rmse_theta.append('%g' % rmse(cse_theta, pf_theta))
        time.append('%g' % t)
        # HSE
        hse_v, hse_theta, t = run_pphse(network, secure=False, fast_decoupled=False, tolerance=tolerance,
                                        max_iter=max_iteration, verbose=False)
        rmse_v.append('%g' % rmse(hse_v, pf_v))
        rmse_theta.append('%g' % rmse(hse_theta, pf_theta))
        time.append('%g' % t)
        # PPHSE
        pphse_v, pphse_theta, t = run_pphse(network, secure=True, fast_decoupled=False, tolerance=tolerance,
                                            max_iter=max_iteration, verbose=False)
        rmse_v.append('%g' % rmse(pphse_v, pf_v))
        rmse_theta.append('%g' % rmse(pphse_theta, pf_theta))
        time.append('%g' % t)

        # print results
        table.add_row(rmse_v)
        table.add_row(rmse_theta)
        table.add_row(time)
        print(table)
    pass


if __name__ == '__main__':
    # accuracy_test()
    # efficiency_tests()
    scalability_test()
