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
from pandapower.auxiliary import pandapowerNet
from pandapower.estimation.state_estimation import estimate


def run_cse(cse_net: pandapowerNet, tolerance=1e-06, max_iter=10):
    try:
        tic = time.perf_counter()
        flag = estimate(cse_net, init='flat', tolerance=tolerance, maximum_iterations=max_iter,
                        zero_injection=None)
        toc = time.perf_counter()
        if not flag:
            raise RuntimeError("CSE fails to converge.")
        else:
            result = cse_net.res_bus_est.sort_index()
            return result.vm_pu.to_numpy().flatten(), result.va_degree.to_numpy().flatten(), (toc - tic)
    except RuntimeError as e:
        print('An error occurred during computation: %s' % e)
        return None, None, float('inf')
