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


import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, csc_matrix, issparse, vstack, hstack
from scipy.sparse.linalg import spsolve, inv
from encryption.paillierthd2 import Deg2PaillierThreshold
from pandapower.estimation.ppc_conversion import pp2eppci, ExtendedPPCI
from pandapower.estimation.algorithm.matrix_base import BaseAlgebra
from pandapower.estimation.results import eppci2pp
from estimation.util import ParaFuncs, pool_map


class HighLevelCtrlCenter(object):
    """
    High-level control center
    """

    def __init__(self, zones, bnd_meas_info, bnd_flow_terms, secure, fast_decoupled):
        self.zones = zones
        self.bnd_meas_info = bnd_meas_info
        self.bnd_flow_terms = bnd_flow_terms
        self.secure = secure
        self.fast_decoupled = fast_decoupled
        pass

    def compute_bnd_flow_h_H(self):
        # This function computes bnd_flow_h and bnd_flow_H and then separates them for each LCC.
        bnd_flow_h_H = dict()
        for zone in self.zones:
            bnd_flow_h_H[zone] = dict()

        _ = self.bnd_flow_terms.to_numpy()
        res = np.array(pool_map(ParaFuncs.bnd_flow_h_H, _))
        res = pd.DataFrame(res, index=self.bnd_flow_terms.index)
        for zone in self.zones:
            _flow = self.bnd_meas_info[zone]['flow']
            bnd_flow_h_H[zone] = res.loc[_flow.index]
        return bnd_flow_h_H

    @staticmethod
    def compute_lambda(bnd_Gi, bnd_ri):
        bnd_Gi_raw = [Gi.toarray().flatten().tolist() if issparse(Gi)
                      else Gi.flatten().tolist() for Gi in bnd_Gi.values()]
        bnd_Gi_raw = list(map(list, zip(*bnd_Gi_raw)))
        bnd_ri_raw = [ri.toarray().flatten().tolist() if issparse(ri)
                      else ri.flatten().tolist() for ri in bnd_ri.values()]
        bnd_ri_raw = list(map(list, zip(*bnd_ri_raw)))
        res = pool_map(ParaFuncs.sum_Gi_ri, bnd_Gi_raw)
        bnd_G = np.array(res).reshape(list(bnd_Gi.values())[0].shape)
        res = pool_map(ParaFuncs.sum_Gi_ri, bnd_ri_raw)
        bnd_r = np.array(res).reshape(list(bnd_ri.values())[0].shape)
        return bnd_G, bnd_r


class LowLevelCtrlCenter(object):
    """
    Low-level control center
    """

    def __init__(self, lcc_info, bnd_meas_info, key, fast_decoupled):
        self.net, self.tie_line_param, self.bnd_meas, self.bnd_zi, self.bnd_Ri = lcc_info
        self.bnd_meas_info = bnd_meas_info
        self.key = key
        self.secure = self.key is not None
        if self.secure:
            self.enc = Deg2PaillierThreshold(self.key)
        self.fast_decoupled = fast_decoupled

        self.net, self.ppc, self.eppci = pp2eppci(self.net, 'flat', 'flat', True, None)
        self.sem = ExtendedBaseAlgebra(self.eppci)
        self.Ri_inv = csr_matrix(np.diagflat(1 / self.eppci.r_cov ** 2))
        self.E = self.eppci.E

        # If using the fast-decoupled model, bnd_flow_H can be computed locally
        self.bnd_flow_Hi = self._compute_bnd_flow_H() if self.fast_decoupled else None
        self.Hi, self.Gi, self.delta_yi = None, None, None
        self.bnd_hi, self.bnd_Hi = None, None
        self.ui = None
        pass

    def _compute_bnd_flow_H(self):
        # This function computes bnd_flow_Hi based on the fast-decoupled model.
        # That is, assume that V = 1.0 pu and theta = 0.
        flow_meas_info = self.bnd_meas_info['flow']
        bnd_flow_H = pd.DataFrame(np.zeros((flow_meas_info.shape[0], 2)), index=flow_meas_info.index)
        for (meas_type, side) in [(x, y) for x in ['p', 'q'] for y in [True, False]]:
            r = flow_meas_info.loc[(flow_meas_info.measurement_type == meas_type) &
                                   (flow_meas_info.side == side)]
            l_par = self.tie_line_param.loc[r.line]
            if meas_type == 'p':
                if side:
                    bnd_flow_H.loc[r.index, 0] = -l_par.b.to_numpy()
                    bnd_flow_H.loc[r.index, 1] = (l_par.g + 2 * l_par.gsh).to_numpy()
                else:
                    bnd_flow_H.loc[r.index, 0] = l_par.b.to_numpy()
                    bnd_flow_H.loc[r.index, 1] = -l_par.g.to_numpy()
            elif meas_type == 'q':
                if side:
                    bnd_flow_H.loc[r.index, 0] = -l_par.g.to_numpy()
                    bnd_flow_H.loc[r.index, 1] = -(l_par.b + 2 * l_par.bsh).to_numpy()
                else:
                    bnd_flow_H.loc[r.index, 0] = l_par.g.to_numpy()
                    bnd_flow_H.loc[r.index, 1] = l_par.b.to_numpy()
        return bnd_flow_H

    def compute_delta_yi(self):
        # This function computes delta_yi
        r = csr_matrix(self.sem.create_rx(self.E)).T
        Hi = csr_matrix(self.sem.create_hx_jacobian(self.E))
        Gi = Hi.T * (self.Ri_inv * Hi)
        delta_yi = spsolve(Gi, Hi.T * (self.Ri_inv * r))
        self.Hi, self.Gi, self.delta_yi = Hi, Gi, delta_yi
        return delta_yi

    def report_bnd_flow_terms(self, container):
        # This function computes the intermediate terms for computing bnd_flow_h and bnd_flow_H.
        v, delta = self.sem.e2v(self.E)
        bus_lookups = list(self.net.bus.name)

        flow_meas_info = self.bnd_meas_info['flow']
        reported_data = []
        for side in [True, False]:
            r = flow_meas_info.loc[flow_meas_info.side == side]
            l_par = self.tie_line_param.loc[r.line]
            in_zone_bus_idx = np.array(list(map(bus_lookups.index, r.bus)), dtype=np.int)
            ppc_bus_idx = self.net['_pd2ppc_lookups']['bus'][in_zone_bus_idx]
            v_bus, delta_bus = v[ppc_bus_idx], delta[ppc_bus_idx]
            if side:
                container.loc[r.index, 0] = v_bus * np.sin(delta_bus)
                container.loc[r.index, 1] = v_bus * np.cos(delta_bus)
                if self.secure:
                    container.loc[r.index, 0:1] = self.enc.encrypt(container.loc[r.index, 0:1].to_numpy())
                reported_data.append(container.loc[r.index, 0:1])
            else:
                container.loc[r.index, 2] = v_bus * (np.cos(delta_bus) * l_par.g.to_numpy() -
                                                     np.sin(delta_bus) * l_par.b.to_numpy())
                container.loc[r.index, 3] = v_bus * (np.sin(delta_bus) * l_par.g.to_numpy() +
                                                     np.cos(delta_bus) * l_par.b.to_numpy())
                if self.secure:
                    container.loc[r.index, 2:3] = self.enc.encrypt(container.loc[r.index, 2:3].to_numpy())
                reported_data.append(container.loc[r.index, 2:3])
        return reported_data

    def final_bnd_flow_hi(self, res):
        # This function finally computes bnd_flow_hi.
        v, delta = self.sem.e2v(self.E)
        bus_lookups = list(self.net.bus.name)
        flow_meas_info = self.bnd_meas_info['flow'].loc[self.bnd_meas_info['flow'].side == True]
        bnd_flow_hi = pd.DataFrame(np.zeros(flow_meas_info.shape[0]), index=flow_meas_info.index)
        for meas_type in ['p', 'q']:
            r = flow_meas_info.loc[(flow_meas_info.measurement_type == meas_type)]
            l_par = self.tie_line_param.loc[r.line]
            in_zone_bus_idx = np.array(list(map(bus_lookups.index, r.bus)), dtype=np.int)
            ppc_bus_idx = self.net['_pd2ppc_lookups']['bus'][in_zone_bus_idx]
            v_bus = v[ppc_bus_idx]
            if meas_type == 'p':
                bnd_flow_hi.loc[r.index, 0] = v_bus ** 2 * (l_par.g + l_par.gsh).to_numpy() - \
                                              res.loc[r.index, 0].to_numpy()
            elif meas_type == 'q':
                bnd_flow_hi.loc[r.index, 0] = -v_bus ** 2 * (l_par.b + l_par.bsh).to_numpy() - \
                                              res.loc[r.index, 1].to_numpy()
        return bnd_flow_hi

    def final_bnd_flow_Hi(self, res):
        # This function finally computes bnd_flow_Hi
        v, delta = self.sem.e2v(self.E)
        bus_lookups = list(self.net.bus.name)
        flow_meas_info = self.bnd_meas_info['flow']
        bnd_flow_Hi = pd.DataFrame(np.zeros((flow_meas_info.shape[0], 2)), index=flow_meas_info.index)
        for (meas_type, side) in [(x, y) for x in ['p', 'q'] for y in [True, False]]:
            r = flow_meas_info.loc[(flow_meas_info.measurement_type == meas_type) &
                                   (flow_meas_info.side == side)]
            l_par = self.tie_line_param.loc[r.line]
            in_zone_bus_idx = np.array(list(map(bus_lookups.index, r.bus)), dtype=np.int)
            ppc_bus_idx = self.net['_pd2ppc_lookups']['bus'][in_zone_bus_idx]
            v_bus = v[ppc_bus_idx]
            if meas_type == 'p':
                if side:
                    bnd_flow_Hi.loc[r.index, 0] = res.loc[r.index, 1].to_numpy()
                    bnd_flow_Hi.loc[r.index, 1] = 2 * v_bus * (l_par.g + l_par.gsh).to_numpy() - \
                                                  res.loc[r.index, 0].to_numpy() / v_bus
                else:
                    bnd_flow_Hi.loc[r.index, 0] = -res.loc[r.index, 1].to_numpy()
                    bnd_flow_Hi.loc[r.index, 1] = -res.loc[r.index, 0].to_numpy() / v_bus
            elif meas_type == 'q':
                if side:
                    bnd_flow_Hi.loc[r.index, 0] = -res.loc[r.index, 0].to_numpy()
                    bnd_flow_Hi.loc[r.index, 1] = -2 * v_bus * (l_par.b + l_par.bsh).to_numpy() - \
                                                  res.loc[r.index, 1].to_numpy() / v_bus
                else:
                    bnd_flow_Hi.loc[r.index, 0] = res.loc[r.index, 0].to_numpy()
                    bnd_flow_Hi.loc[r.index, 1] = -res.loc[r.index, 1].to_numpy() / v_bus
        return bnd_flow_Hi

    def compute_bnd_hi_Hi(self, bnd_flow_hi, bnd_flow_Hi):
        return self._bnd_hi(bnd_flow_hi), self._bnd_Hi(bnd_flow_Hi, bnd_flow_hi)

    def _bnd_hi(self, bnd_flow_hi):
        bnd_hi = np.zeros(self.bnd_zi.shape[0])
        bus_lookups = list(self.net.bus.name)
        v, delta = self.sem.e2v(self.E)
        _ = self.bnd_meas_info
        _flow = _['flow'].loc[_['flow'].side == True]
        _i = _['current'].loc[_['current'].side == True]
        _inj = _['injection'].loc[_['injection'].side == True]
        _v = _['voltage']

        # line flow measurements
        _flow_actual = _flow.loc[_flow.type == 'actual']
        bnd_hi[_flow_actual.measurement] = bnd_flow_hi.loc[_flow_actual.index].to_numpy().flatten()
        # line current magnitude measurements
        related_p_meas = [i[0] for i in list(_i.related_p_meas)]
        related_q_meas = [i[0] for i in list(_i.related_q_meas)]
        p = bnd_flow_hi.loc[related_p_meas].to_numpy().flatten()
        q = bnd_flow_hi.loc[related_q_meas].to_numpy().flatten()
        s = np.sqrt(p ** 2 + q ** 2)
        bus_idx_in_zone = list(map(bus_lookups.index, list(_i.bus)))
        ppc_bus_idx = self.net['_pd2ppc_lookups']['bus'][bus_idx_in_zone]
        v_bus = v[ppc_bus_idx]
        bnd_hi[_i.measurement] = s / v_bus
        # bus injection measurements
        bus_idx_in_zone = list(map(bus_lookups.index, list(_inj.bus)))
        ppc_bus_idx = self.net['_pd2ppc_lookups']['bus'][bus_idx_in_zone]
        int_hi_idx = np.where(_inj.measurement_type == 'p',
                              ppc_bus_idx,
                              ppc_bus_idx + len(self.eppci['bus']) + 2 * len(self.eppci['branch']))
        int_hi = self.sem.hx[int_hi_idx]
        related_flow_idx = _inj.related_flow_meas
        added_hi = np.array([np.sum(bnd_flow_hi.loc[idx].to_numpy()) for idx in related_flow_idx])
        bnd_hi[_inj.measurement] = (int_hi + added_hi)
        # bus voltage measurements
        bus_idx_in_zone = list(map(bus_lookups.index, list(_v.bus)))
        ppc_bus_idx = self.net['_pd2ppc_lookups']['bus'][bus_idx_in_zone]
        bnd_hi[_v.measurement] = v[ppc_bus_idx]
        self.bnd_hi = bnd_hi
        return bnd_hi

    def _bnd_Hi(self, bnd_flow_H, bnd_flow_h):
        bnd_Hi = np.zeros((self.bnd_zi.shape[0], self.net.bus.shape[0] * 2))
        bus_lookups = list(self.net.bus.name)
        v, delta = self.sem.e2v(self.E)
        _ = self.bnd_meas_info
        _flow, _i, _inj, _v = _['flow'], _['current'], _['injection'], _['voltage']
        # line flow measurements
        _flow_actual = _flow.loc[_flow.type == 'actual']
        bus_idx_in_zone = list(map(bus_lookups.index, list(_flow_actual.bus)))
        theta_idx = self.net['_pd2ppc_lookups']['bus'][bus_idx_in_zone]
        v_idx = theta_idx + self.net.bus.shape[0]
        bnd_Hi[_flow_actual.measurement, theta_idx] = bnd_flow_H.loc[_flow_actual.index, 0].to_numpy()
        bnd_Hi[_flow_actual.measurement, v_idx] = bnd_flow_H.loc[_flow_actual.index, 1].to_numpy()
        # line current magnitude measurements
        for side in [True, False]:
            r = _i.loc[_i.side == side]
            bus_idx_in_zone = list(map(bus_lookups.index, list(r.bus)))
            ppc_idx = self.net['_pd2ppc_lookups']['bus'][bus_idx_in_zone]
            theta_idx = ppc_idx
            v_idx = theta_idx + self.net.bus.shape[0]
            if side:
                p_idx = [i[0] for i in list(r.related_p_meas)]
                q_idx = [i[0] for i in list(r.related_q_meas)]
                p, q = bnd_flow_h.loc[p_idx].to_numpy().flatten(), bnd_flow_h.loc[q_idx].to_numpy().flatten()
                p, q = np.where(abs(p) < 1e-8, 0, p), np.where(abs(q) < 1e-8, 0, q)
                s = np.sqrt(p ** 2 + q ** 2)
                cos, sin = np.nan_to_num(p / s), np.nan_to_num(q / s)
                v_bus = v[ppc_idx]
                bnd_Hi[r.measurement, theta_idx] = (cos * bnd_flow_H.loc[p_idx, 0].to_numpy() +
                                                    sin * bnd_flow_H.loc[q_idx, 0].to_numpy()) / v_bus
                bnd_Hi[r.measurement, v_idx] = (-s / v_bus + (cos * bnd_flow_H.loc[p_idx, 1].to_numpy() +
                                                              sin * bnd_flow_H.loc[q_idx, 1].to_numpy())) / v_bus
            else:
                p_idx = [i[1] for i in list(r.related_p_meas)]
                q_idx = [i[1] for i in list(r.related_q_meas)]
                p, q = bnd_flow_h.loc[p_idx].to_numpy().flatten(), bnd_flow_h.loc[q_idx].to_numpy().flatten()
                S_to = p + 1j * q
                Y = (self.tie_line_param.loc[_flow.loc[p_idx].line].g.to_numpy() +
                     1j * self.tie_line_param.loc[_flow.loc[p_idx].line].b.to_numpy())
                Ysh = (self.tie_line_param.loc[_flow.loc[p_idx].line].gsh.to_numpy() +
                       1j * self.tie_line_param.loc[_flow.loc[p_idx].line].bsh.to_numpy())
                v_to, delta_to = v[ppc_idx], delta[ppc_idx]
                V_to = v_to * np.exp(1j * delta_to)
                I_to = np.conj(S_to / V_to)
                I_line = I_to - V_to * Ysh
                V_from = V_to - I_line / Y
                I_from = V_from * Ysh - I_line
                S_from = V_from * np.conj(I_from)
                p, q = np.real(S_from), np.imag(S_from)
                p, q = np.where(abs(p) < 1e-8, 0, p), np.where(abs(q) < 1e-8, 0, q)
                s = np.abs(S_from)
                cos, sin = np.nan_to_num(p / s), np.nan_to_num(q / s)
                v_bus = np.abs(V_from)
                p_idx = [i[0] for i in list(r.related_p_meas)]
                q_idx = [i[0] for i in list(r.related_q_meas)]
                bnd_Hi[r.measurement, theta_idx] = (cos * bnd_flow_H.loc[p_idx, 0].to_numpy() +
                                                    sin * bnd_flow_H.loc[q_idx, 0].to_numpy()) / v_bus
                bnd_Hi[r.measurement, v_idx] = (cos * bnd_flow_H.loc[p_idx, 1].to_numpy() +
                                                sin * bnd_flow_H.loc[q_idx, 1].to_numpy()) / v_bus
        # bus injection measurements
        for side in [True, False]:
            r = _inj.loc[_inj.side == side]
            bus_idx_in_zone = list(map(bus_lookups.index, list(r.bus)))
            ppc_idx = self.net['_pd2ppc_lookups']['bus'][bus_idx_in_zone]
            theta_idx = ppc_idx
            v_idx = theta_idx + self.net.bus.shape[0]
            related_flow = r.related_flow_meas
            added_Hi_dS_dtheta = np.array([np.sum(bnd_flow_H.loc[idx, 0].to_numpy()) for idx in related_flow])
            added_Hi_dS_dv = np.array([np.sum(bnd_flow_H.loc[idx, 1].to_numpy()) for idx in related_flow])
            if side:
                int_Hi_idx = np.where(r.measurement_type == 'p',
                                      ppc_idx,
                                      ppc_idx + len(self.eppci['bus']) + 2 * len(self.eppci['branch']))
                bnd_Hi[r.measurement, :] = self.sem.Hx[int_Hi_idx, :]
                bnd_Hi[r.measurement, theta_idx] += added_Hi_dS_dtheta
                bnd_Hi[r.measurement, v_idx] += added_Hi_dS_dv
            else:
                bnd_Hi[r.measurement, theta_idx] = added_Hi_dS_dtheta
                bnd_Hi[r.measurement, v_idx] = added_Hi_dS_dv
        # bus voltage magnitude measurements
        bus_idx_in_zone = list(map(bus_lookups.index, list(_v.bus)))
        ppc_idx = self.net['_pd2ppc_lookups']['bus'][bus_idx_in_zone]
        v_idx = ppc_idx + self.net.bus.shape[0]
        bnd_Hi[_v.measurement, v_idx] = 1
        self.bnd_Hi = bnd_Hi[:, self.sem.delta_v_bus_mask]
        return self.bnd_Hi

    def compute_bnd_Gi(self):
        bnd_Ri = csc_matrix(self.bnd_Ri)
        bnd_Hi = csc_matrix(self.bnd_Hi)
        Gi_inv = inv(self.Gi)
        bnd_Gi = bnd_Ri + bnd_Hi * Gi_inv * bnd_Hi.T
        return bnd_Gi

    def compute_bnd_ri(self):
        delta_yi = csr_matrix(self.delta_yi).T if not issparse(self.delta_yi) else self.delta_yi
        bnd_ri = csr_matrix(self.bnd_zi - self.bnd_hi).T
        bnd_Hi = csr_matrix(self.bnd_Hi)
        bnd_ri = bnd_ri - bnd_Hi * delta_yi
        return bnd_ri

    def compute_ui(self, bnd_G, bnd_r):
        if not issparse(bnd_G):
            bnd_G = csc_matrix(bnd_G)
        if not issparse(bnd_r):
            bnd_r = csc_matrix(bnd_r)
        bnd_Hi = csr_matrix(self.bnd_Hi)
        ui = spsolve(self.Gi, bnd_Hi.T * (inv(bnd_G) * bnd_r))
        self.ui = ui
        return ui

    def compute_delta_xi(self):
        return self.delta_yi + self.ui

    def save_result(self, global_v, global_delta):
        self.net = eppci2pp(self.net, self.ppc, self.eppci)
        global_idx = self.net.bus.loc[self.net.res_bus_est.index].name
        global_v[global_idx] = list(self.net.res_bus_est.vm_pu)
        global_delta[global_idx] = list(self.net.res_bus_est.va_degree)
        pass


class ExtendedBaseAlgebra(BaseAlgebra):
    """
    This class extends the BaseAlgebra class of pandapower
    """

    def __init__(self, eppci: ExtendedPPCI):
        super().__init__(eppci)
        self.v = eppci.v_init.copy()
        self.delta = eppci.delta_init.copy()
        self.hx = None
        self.Hx = None
        pass

    def e2v(self, E):
        self.v = E[self.num_non_slack_bus:]
        self.delta[self.non_slack_buses] = E[:self.num_non_slack_bus]
        return self.v, self.delta

    def create_hx(self, E):
        f_bus, t_bus = self.fb, self.tb
        v, delta = self.e2v(E)
        V = v * np.exp(1j * delta)
        Sfe = V[f_bus] * np.conj(self.Yf * V)
        Ste = V[t_bus] * np.conj(self.Yt * V)
        Sbuse = V * np.conj(self.Ybus * V)
        hx = np.r_[np.real(Sbuse),
                   np.real(Sfe),
                   np.real(Ste),
                   np.imag(Sbuse),
                   np.imag(Sfe),
                   np.imag(Ste),
                   v]
        if self.any_i_meas or self.any_degree_meas:
            va = delta
            Ife = self.Yf * V
            ifem = np.abs(Ife)
            ifea = np.angle(Ife)
            Ite = self.Yt * V
            item = np.abs(Ite)
            itea = np.angle(Ite)
            hx = np.r_[hx,
                       va,
                       ifem,
                       item,
                       ifea,
                       itea]
        self.hx = hx
        return hx[self.non_nan_meas_selector]

    def create_hx_jacobian(self, E):
        # Using sparse matrix in creation sub-jacobian matrix
        v, delta = self.e2v(E)
        V = v * np.exp(1j * delta)

        dSbus_dth, dSbus_dv = self._dSbus_dv(V)
        dSf_dth, dSf_dv, dSt_dth, dSt_dv = self._dSbr_dv(V)
        dvm_dth, dvm_dv = self._dvmbus_dV(V)

        s_jac_th = vstack((dSbus_dth.real,
                           dSf_dth.real,
                           dSt_dth.real,
                           dSbus_dth.imag,
                           dSf_dth.imag,
                           dSt_dth.imag))
        s_jac_v = vstack((dSbus_dv.real,
                          dSf_dv.real,
                          dSt_dv.real,
                          dSbus_dv.imag,
                          dSf_dv.imag,
                          dSt_dv.imag))

        s_jac = hstack((s_jac_th, s_jac_v)).toarray()
        vm_jac = np.c_[dvm_dth, dvm_dv]
        jac = np.r_[s_jac,
                    vm_jac]

        if self.any_i_meas or self.any_degree_meas:
            dva_dth, dva_dv = self._dvabus_dV(V)
            va_jac = np.c_[dva_dth, dva_dv]
            difm_dth, difm_dv, ditm_dth, ditm_dv, difa_dth, difa_dv, dita_dth, dita_dv = self._dimiabr_dV(V)
            im_jac_th = np.r_[difm_dth,
                              ditm_dth]
            im_jac_v = np.r_[difm_dv,
                             ditm_dv]
            ia_jac_th = np.r_[difa_dth,
                              dita_dth]
            ia_jac_v = np.r_[difa_dv,
                             dita_dv]

            im_jac = np.c_[im_jac_th, im_jac_v]
            ia_jac = np.c_[ia_jac_th, ia_jac_v]

            jac = np.r_[jac,
                        va_jac,
                        im_jac,
                        ia_jac]
        self.Hx = jac
        return jac[self.non_nan_meas_selector, :][:, self.delta_v_bus_selector]
