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
from prettytable import PrettyTable

from pandapower import runpp, create_ext_grid, create_measurement, create_empty_network
from encryption.key import gen_key
from estimation.participant import HighLevelCtrlCenter, LowLevelCtrlCenter
from estimation.util import reset_index, update_index


class InterPowerSystem(object):
    """
    Interconnected power system
    """

    def __init__(self, net_param):
        net = net_param['ppnet']
        # check for unsupported power equipment
        unsupported = ['switch', 'trafo3w', 'impedance', 'ward', 'xward', 'dcline', 'storage']
        for key in unsupported:
            if len(net[key]) > 0:
                raise TypeError("Cannot support ""%s"" table in the network." % key)
        self.net = net
        self.has_meas = False

        # run power flow calculation to get ppc arrays
        runpp(self.net)
        # # remove zero impedance lines based on power flow results
        # self._remove_not_in_service_elements()

        # The power flow results calculated by the pandapower package include
        # the energy consumptions of shunt elements.
        # However, bus power injection measurements used in state estimation
        # should only contain the injection of generators and loads.
        # Hence, the energy consumptions of shunt elements need to be subtracted
        # from the power flow results before generating power injection measurements.
        res_bus = self.net.res_bus.copy()
        shunt = self.net.shunt.copy()
        shunt['p_actual'] = self.net.res_shunt.loc[shunt.index, 'p_mw']
        shunt['q_actual'] = self.net.res_shunt.loc[shunt.index, 'q_mvar']
        shunt_bus_idx = res_bus.loc[shunt.bus].index
        p_mw = res_bus.loc[shunt_bus_idx, 'p_mw']
        res_bus.loc[shunt_bus_idx, 'p_mw'] = p_mw.to_numpy() - shunt['p_actual'].to_numpy()
        q_mvar = res_bus.loc[shunt_bus_idx, 'q_mvar']
        res_bus.loc[shunt_bus_idx, 'q_mvar'] = q_mvar.to_numpy() - shunt['q_actual'].to_numpy()

        # initialize lists for storing tie-line and boundary bus indices
        self.tie_line_idx, self.boundary_bus_idx = None, None
        # assign buses to different zones
        self._assign_zones(net_param['zone_info'])
        # set a reference bus for each zone
        self._set_reference_bus(net_param['ref_bus'])

    def _get_bus_zone(self, bus_idx):
        # return the id of the zone that the given bus belongs to
        return self.net.bus.loc[bus_idx, 'zone'].to_numpy()

    def _assign_zones(self, zone_info: dict):
        # The orginal pandapower networks do not contain zone information.
        # This function assigns a zone id to each bus.
        # <zone_info>:
        #   - key: zone id
        #   - value: a set of bus ids belonging to this zone
        net = self.net
        bus, line, trafo = net.bus, net.line, net.trafo
        for zone, idxs in zone_info.items():
            bus.loc[list(set(idxs).intersection(bus.index)), 'zone'] = zone
        if trafo.loc[self._get_bus_zone(trafo.hv_bus) != self._get_bus_zone(trafo.lv_bus)].shape[0] > 0:
            raise TypeError("Transformers cannot be used to connect two control areas.")
        tie_lines = line.loc[self._get_bus_zone(line.from_bus) != self._get_bus_zone(line.to_bus)]
        self.tie_line_idx = tie_lines.index
        self.boundary_bus_idx = list(set(tie_lines.from_bus) | set(tie_lines.to_bus))
        pass

    def _set_reference_bus(self, reference_bus):
        # The orginal pandapower networks often contain only one reference bus.
        # This function designates a reference bus for each zone.
        # <reference_bus>: a list of reference bus ids
        net = self.net
        net.ext_grid.drop(net.ext_grid.index, inplace=True)
        zone_idx = set(net.bus.zone.unique())
        added_zone = set(net.bus.loc[net.bus.index.isin(reference_bus), 'zone'])
        if len(zone_idx - added_zone) > 0:
            raise Exception("Must specify a reference bus for each zone.")
        for ref in reference_bus:
            create_ext_grid(net, ref, net.res_bus.loc[ref, 'vm_pu'], net.res_bus.loc[ref, 'va_degree'])
        pass

    def add_measurement(self, meas_type, element_type, element, side='', std_dev=0.05):
        # The original pandapower networks do not contain measurements.
        # This function adds measurements to the power system.
        # For detailed information about what kind of measurements are support,
        # please check the following conditional logics.

        meas_type = meas_type.lower()
        element_type = element_type.lower()
        if not type(element) is list:
            element = element.lower()
        side = side.lower()

        if element_type not in ['bus', 'line', 'trafo']:
            raise TypeError("Cannot install measurements on ""%s""." % element_type)
        if element_type == 'bus':
            if meas_type not in ['p', 'q', 'v']:
                raise TypeError("Only P, Q, and V measurements are supported for a bus.")
        else:
            if meas_type not in ['p', 'q', 'i']:
                raise TypeError("Only P, Q, and I measurements are supported for a branch.")
        if element_type == 'line':
            if side not in ['from', 'to', 'both']:
                raise TypeError("Cannot install measurements at the ""%s"" side of a line." % side)
        elif element_type == 'trafo':
            if side not in ['hv', 'lv', 'both']:
                raise TypeError("Cannot install measurements at the ""%s"" side of a transformer." % side)
        if not (type(element) is list or element in ['internal', 'boundary', 'half_internal', 'half_boundary', 'all']):
            raise TypeError("Cannot understand where to install the measurements.")
        if element_type in ['line', 'trafo'] and element.startswith('half_'):
            raise TypeError("Cannot install measurements on half of branches.")
        if not isinstance(std_dev, (int, float)):
            raise TypeError("Standard deviation should be a number.")

        net = self.net
        if element_type == 'bus':
            if isinstance(element, list):
                meas_bus_idx = list(set(element).intersection(self.net.bus.index))
            elif element.endswith('boundary'):
                meas_bus_idx = self.boundary_bus_idx
                if element.startswith('half_'):
                    meas_bus_idx = list(np.random.choice(meas_bus_idx, len(meas_bus_idx) // 2, replace=False))
            else:
                meas_bus_idx = net.bus.index.tolist()
                if element.endswith('internal'):
                    meas_bus_idx = list(set(meas_bus_idx) - set(self.boundary_bus_idx))
                    if element.startswith('half_'):
                        meas_bus_idx = list(np.random.choice(meas_bus_idx, len(meas_bus_idx) // 2, replace=False))
            for idx in meas_bus_idx:
                if meas_type == 'p':
                    true_value = -net.res_bus.loc[idx, 'p_mw']
                elif meas_type == 'q':
                    true_value = -net.res_bus.loc[idx, 'q_mvar']
                elif meas_type == 'v':
                    true_value = net.res_bus.loc[idx, 'vm_pu']
                value = np.random.normal(true_value, std_dev * abs(true_value))
                create_measurement(net, meas_type, 'bus', value, std_dev, idx)
                self.has_meas = True
        elif element_type in ['line', 'trafo']:
            if isinstance(element, list):
                if element_type == 'line':
                    meas_dev_idx = list(set(element).intersection(self.net.line.index))
                else:
                    meas_dev_idx = list(set(element).intersection(self.net.trafo.index))
            elif element == 'boundary':
                if element_type == 'trafo':
                    raise TypeError("No boundary transformer is allowed.")
                meas_dev_idx = self.tie_line_idx
            else:
                meas_dev_idx = net[element_type].index.tolist()
                if element == 'internal':
                    if element_type == 'line':
                        meas_dev_idx = list(set(meas_dev_idx) - set(self.tie_line_idx))
            for idx in meas_dev_idx:
                if element_type == 'line':
                    sides = ['from', 'to'] if side == 'both' else [side]
                elif element_type == 'trafo':
                    sides = ['hv', 'lv'] if side == 'both' else [side]
                for s in sides:
                    if meas_type == 'p':
                        true_value = net['res_' + element_type].loc[idx, 'p_' + s + '_mw']
                    elif meas_type == 'q':
                        true_value = net['res_' + element_type].loc[idx, 'q_' + s + '_mvar']
                    elif meas_type == 'i':
                        true_value = net['res_' + element_type].loc[idx, 'i_' + s + '_ka']
                    value = np.random.normal(true_value, std_dev * abs(true_value))
                    create_measurement(net, meas_type, element_type, value, std_dev, idx, s)
                    self.has_meas = True
        pass

    def partition_by_zone(self, secure=True, key_len=2048, fast_decoupled=False):
        # This function partition the interconnected power system into
        # several low-level control centers and a high-level control center.
        # <secure>: whether cryptography is involved in the partition
        # <key_len>: if <secure> is True, <key_len> indicates the length of the keys
        # <fast_decoupled>: whether the fast-decoupled state estimation model is used

        net = self.net
        measurement = net.measurement
        zone_idx = net.bus.zone.unique()

        # extract boundary buses, tie-lines, and boundary measurements
        all_bnd_bus = net.bus.loc[net.bus.index.isin(self.boundary_bus_idx)]
        all_tie_line = net.line.loc[net.line.index.isin(self.tie_line_idx)]
        all_bnd_bus_meas = measurement.loc[(measurement.element_type == 'bus') &
                                           measurement.element.isin(all_bnd_bus.index)]
        all_tie_line_meas = measurement.loc[(measurement.element_type == 'line') &
                                            measurement.element.isin(all_tie_line.index)]
        all_bnd_meas = reset_index(pd.concat([all_bnd_bus_meas, all_tie_line_meas]))
        bnd_meas_lookups = list(all_bnd_meas.name)

        local_info = dict()
        for zone in zone_idx:
            # create local networks
            local_network = create_empty_network("Local Network %d" % zone, net.f_hz, net.sn_mva)

            # add primary equipment
            bus = net.bus.loc[net.bus.zone == zone]
            local_network.bus = reset_index(bus)
            bus_lookups = list(local_network.bus.name)
            line = net.line.loc[net.line.from_bus.isin(bus.index) & net.line.to_bus.isin(bus.index)]
            local_network.line = reset_index(update_index(line, ['from_bus', 'to_bus'], bus_lookups))
            line_lookups = list(local_network.line.name)
            trafo = net.trafo.loc[net.trafo.hv_bus.isin(bus.index) & net.trafo.lv_bus.isin(bus.index)]
            local_network.trafo = reset_index(update_index(trafo, ['hv_bus', 'lv_bus'], bus_lookups))
            trafo_lookups = list(local_network.trafo.name)
            load = net.load.loc[net.load.bus.isin(bus.index)]
            local_network.load = reset_index(update_index(load, ['bus'], bus_lookups))
            sgen = net.sgen.loc[net.sgen.bus.isin(bus.index)]
            local_network.sgen = reset_index(update_index(sgen, ['bus'], bus_lookups))
            gen = net.gen.loc[net.gen.bus.isin(bus.index)]
            local_network.gen = reset_index(update_index(gen, ['bus'], bus_lookups))
            shunt = net.shunt.loc[net.shunt.bus.isin(bus.index)]
            local_network.shunt = reset_index(update_index(shunt, ['bus'], bus_lookups))
            ext_grid = net.ext_grid.loc[net.ext_grid.bus.isin(bus.index)]
            local_network.ext_grid = reset_index(update_index(ext_grid, ['bus'], bus_lookups))

            # add internal measurement
            int_bus = bus.loc[~bus.index.isin(self.boundary_bus_idx)]
            bus_meas = measurement.loc[(measurement.element_type == 'bus') &
                                       measurement.element.isin(int_bus.index)]
            bus_meas = update_index(bus_meas, ['element'], bus_lookups)
            line_meas = measurement.loc[(measurement.element_type == 'line') &
                                        measurement.element.isin(line.index)]
            line_meas = update_index(line_meas, ['element'], line_lookups)
            trafo_meas = measurement.loc[(measurement.element_type == 'trafo') &
                                         measurement.element.isin(trafo.index)]
            trafo_meas = update_index(trafo_meas, ['element'], trafo_lookups)
            local_network.measurement = pd.concat([bus_meas, line_meas, trafo_meas])
            local_network.measurement.element = local_network.measurement.element.astype(np.int)

            # add boundary measurements
            bnd_bus = all_bnd_bus.loc[all_bnd_bus.zone == zone]
            tie_line_from_zone = all_tie_line.loc[all_tie_line.from_bus.isin(bnd_bus.index)]
            tie_line_to_zone = all_tie_line.loc[all_tie_line.to_bus.isin(bnd_bus.index)]
            bnd_bus_meas = all_bnd_meas.loc[(all_bnd_meas.element_type == 'bus') &
                                            all_bnd_meas.element.isin(bnd_bus.index)]
            bnd_line_meas_from = all_bnd_meas.loc[(all_bnd_meas.element_type == 'line') &
                                                  (all_bnd_meas.side == 'from') &
                                                  all_bnd_meas.element.isin(tie_line_from_zone.index)]
            bnd_line_meas_to = all_bnd_meas.loc[(all_bnd_meas.element_type == 'line') &
                                                (all_bnd_meas.side == 'to') &
                                                all_bnd_meas.element.isin(tie_line_to_zone.index)]
            bnd_meas_in_zone = pd.concat([bnd_bus_meas, bnd_line_meas_from, bnd_line_meas_to])
            bnd_meas_in_zone.index = bnd_meas_in_zone.name
            bnd_meas_in_zone = update_index(reset_index(bnd_meas_in_zone), ['name'], bnd_meas_lookups)
            bnd_meas_in_zone.element = bnd_meas_in_zone.element.astype(np.int)

            # store tie_line parameters
            related_tie_line = pd.concat([tie_line_from_zone, tie_line_to_zone])
            r = related_tie_line.r_ohm_per_km.to_numpy()
            x = related_tie_line.x_ohm_per_km.to_numpy()
            c = related_tie_line.c_nf_per_km.to_numpy()
            g = related_tie_line.g_us_per_km.to_numpy()
            length = related_tie_line.length_km.to_numpy()
            parallel = related_tie_line.parallel.to_numpy()
            V_base = all_bnd_bus.loc[related_tie_line.from_bus, 'vn_kv'].to_numpy()
            Y = parallel / (r + 1j * x) / length
            Ysh = ((g * 5e-7) + 1j * np.pi * net.f_hz * c * 1e-9) * length * parallel
            y = Y * V_base ** 2 / net.sn_mva
            ysh = Ysh * V_base ** 2 / net.sn_mva
            tie_line_param = pd.DataFrame({'g': y.real, 'b': y.imag, 'gsh': ysh.real, 'bsh': ysh.imag},
                                          index=related_tie_line.index, dtype=np.float)

            # use the <value> field of <all_bnd_meas> to store
            # the zone a boundary measurement belongs to
            all_bnd_meas.loc[bnd_meas_in_zone.name, 'value'] = zone

            # generate bnd_zi
            bnd_zi = np.zeros(all_bnd_meas.shape[0])
            # convert to pu values
            bnd_p_q_meas = bnd_meas_in_zone.loc[(bnd_meas_in_zone.measurement_type == 'p') |
                                                (bnd_meas_in_zone.measurement_type == 'q')]
            bnd_zi[list(bnd_p_q_meas.name)] = bnd_p_q_meas.value.to_numpy() / net.sn_mva
            bnd_i_meas = bnd_meas_in_zone.loc[bnd_meas_in_zone.measurement_type == 'i']
            for idx, i in bnd_i_meas.iterrows():
                i_meas_bus = all_tie_line.loc[i.element, i.side + '_bus']
                v_bus_kv = all_bnd_bus.loc[i_meas_bus, 'vn_kv']
                bnd_zi[i['name']] = i.value / net.sn_mva * v_bus_kv * np.sqrt(3)
            bnd_v_meas = bnd_meas_in_zone.loc[bnd_meas_in_zone.measurement_type == 'v']
            bnd_zi[list(bnd_v_meas.name)] = bnd_v_meas.value.to_numpy()
            # generate bnd_Ri
            bnd_std_dev = np.zeros(all_bnd_meas.shape[0])
            bnd_std_dev[list(bnd_meas_in_zone.name)] = bnd_meas_in_zone.std_dev.to_numpy()
            bnd_Ri = np.diagflat(bnd_std_dev ** 2)

            # add to <local_info>
            local_info[zone] = (local_network, tie_line_param, bnd_meas_in_zone, bnd_zi, bnd_Ri)

        # reset index for all boundary buses and tie-lines
        all_bnd_bus = reset_index(all_bnd_bus)
        bus_lookups = list(all_bnd_bus.name)
        all_tie_line = reset_index(update_index(all_tie_line, ['from_bus', 'to_bus'], bus_lookups))
        line_lookups = list(all_tie_line.name)
        all_bnd_bus_meas = update_index(all_bnd_meas.iloc[:all_bnd_bus_meas.shape[0]], ['element'], bus_lookups)
        all_tie_line_meas = update_index(all_bnd_meas.iloc[all_bnd_bus_meas.shape[0]:], ['element'], line_lookups)
        all_bnd_meas = pd.concat([all_bnd_bus_meas, all_tie_line_meas])
        all_bnd_meas.value = all_bnd_meas.value.astype(np.int)

        # generate boundary measurement information and the container for intermediate terms
        bnd_meas_info, bnd_flow_terms = self._generate_bnd_meas_info(zone_idx, all_bnd_bus, all_tie_line,
                                                                     all_bnd_meas, secure, fast_decoupled)
        # create HCC
        hcc = HighLevelCtrlCenter(zone_idx, bnd_meas_info, bnd_flow_terms, secure, fast_decoupled)

        # if <secure>, generate keys for LCCs
        if secure:
            keys = gen_key(bitlen=key_len, n_party=len(zone_idx), n_threshold=len(zone_idx))
        # create LCCs
        lccs = dict()
        for zone in zone_idx:
            lccs[zone] = LowLevelCtrlCenter(local_info[zone], bnd_meas_info[zone],
                                            keys[zone] if secure else None, fast_decoupled)
        return hcc, lccs

    @staticmethod
    def _generate_bnd_meas_info(zone_idx, bnd_bus, tie_line, bnd_meas, secure, fast_decoupled):
        # This function generates boundary measurement information for HCC and LCC,
        # including the information <bnd_meas_info> about the boundary measurement allocation,
        # and a container <bnd_flow_terms> used to hold the intermediate terms
        # reported by LCCs for the HCC computing bnd_flow_h and bnd_flow_H
        bnd_meas_info = dict()
        for z in zone_idx:
            bnd_meas_info[z] = dict()

        # add tie-line power flow measurements
        # actual measurements
        flow_meas = bnd_meas.loc[(bnd_meas.element_type == 'line') & (bnd_meas.measurement_type != 'i')]
        meas_idx = list(flow_meas.index)
        meas_type = list(flow_meas.measurement_type)
        line_idx = list(flow_meas.element)
        meas_side = list(flow_meas.side)
        actual_type = ['actual'] * len(meas_idx)

        # Tie-line current magnitude and boundary bus power injection measurements
        # are related to tie-line power flow measurements.
        # For each current magnitude or power injection measurement,
        # check if the corresponding tie-line power flow measurements are requested.
        # If not, create pseudo tie-line power flow measurements.

        # tie-line current magnitude measurements
        i_meas = bnd_meas.loc[(bnd_meas.element_type == 'line') & (bnd_meas.measurement_type == 'i')]
        related_p_meas_idx, related_q_meas_idx = [], []
        for idx, row in i_meas.iterrows():
            related_p_meas, related_q_meas = None, None
            related_p_meas_other_side, related_q_meas_other_side = None, None
            for i, t, l, s in zip(range(len(meas_idx)), meas_type, line_idx, meas_side):
                if l == row.element:
                    if s == row.side:
                        if t == 'p':
                            related_p_meas = i
                        elif t == 'q':
                            related_q_meas = i
                    elif s == 'fromto'.replace(row.side, ''):
                        if t == 'p':
                            related_p_meas_other_side = i
                        elif t == 'q':
                            related_q_meas_other_side = i
                if related_p_meas and related_q_meas and related_p_meas_other_side and related_q_meas_other_side:
                    break
            if related_p_meas is None:
                meas_idx.append(idx)
                line_idx.append(row.element)
                meas_type.append('p')
                meas_side.append(row.side)
                actual_type.append('pseudo')
                related_p_meas = len(meas_idx) - 1
            if related_q_meas is None:
                meas_idx.append(idx)
                line_idx.append(row.element)
                meas_type.append('q')
                meas_side.append(row.side)
                actual_type.append('pseudo')
                related_q_meas = len(meas_idx) - 1
            if related_p_meas_other_side is None:
                meas_idx.append(idx)
                line_idx.append(row.element)
                meas_type.append('p')
                meas_side.append('fromto'.replace(row.side, ''))
                actual_type.append('pseudo')
                related_p_meas_other_side = len(meas_idx) - 1
            if related_q_meas_other_side is None:
                meas_idx.append(idx)
                line_idx.append(row.element)
                meas_type.append('q')
                actual_type.append('pseudo')
                meas_side.append('fromto'.replace(row.side, ''))
                related_q_meas_other_side = len(meas_idx) - 1
            related_p_meas_idx.append([related_p_meas, related_p_meas_other_side])
            related_q_meas_idx.append([related_q_meas, related_q_meas_other_side])
        # boundary bus power injection measurements
        inj_meas = bnd_meas.loc[(bnd_meas.element_type == 'bus') & (bnd_meas.measurement_type != 'v')]
        related_flow_meas_idx, inj_meas_type = [], []
        for idx, row in inj_meas.iterrows():
            this_related_flow_meas_idx = []
            inj_meas_type.append(row.measurement_type)
            related_line = tie_line.loc[(tie_line.from_bus == row.element) | (tie_line.to_bus == row.element)]
            for idx_l, row_l in related_line.iterrows():
                related_flow_meas_side = 'from' if row_l.from_bus == row.element else 'to'
                related_flow_meas = None
                for i, t, l, s in zip(range(len(meas_idx)), meas_type, line_idx, meas_side):
                    if (t == row.measurement_type) and (l == idx_l) and (s == related_flow_meas_side):
                        related_flow_meas = i
                        break
                if related_flow_meas is not None:
                    this_related_flow_meas_idx.append(related_flow_meas)
                else:
                    meas_idx.append(idx)
                    line_idx.append(idx_l)
                    meas_type.append(row.measurement_type)
                    meas_side.append(related_flow_meas_side)
                    actual_type.append('pseudo')
                    this_related_flow_meas_idx.append(len(meas_idx) - 1)
            related_flow_meas_idx.append(this_related_flow_meas_idx)

        # generate information for both actual and pseudo tie-line power flow measurements
        related_line = tie_line.loc[line_idx]
        related_line_from_bus = bnd_bus.loc[related_line.from_bus]
        related_line_to_bus = bnd_bus.loc[related_line.to_bus]
        related_bus = pd.concat([related_line_from_bus, related_line_to_bus])
        bus_side = ['from'] * related_line_from_bus.shape[0] + ['to'] * related_line_to_bus.shape[0]
        this_side = np.array(meas_side * 2) == bus_side
        flow_meas_info = pd.DataFrame({'measurement': meas_idx * 2,
                                       'line': list(related_line.name) * 2,
                                       'measurement_type': meas_type * 2,
                                       'type': actual_type * 2,
                                       'bus': list(related_bus.name),
                                       'side': list(this_side),
                                       'zone': list(related_bus.zone)})
        # re-index <flow_meas_info> to ensure that the new index equals to the index of the
        # corresponding row in the container for holding the reported intermediate terms.
        flow_meas_info.index = flow_meas_info.index.where(flow_meas_info.index < flow_meas_info.shape[0] // 2,
                                                          flow_meas_info.index - flow_meas_info.shape[0] // 2)
        # initialize the container for holding the intermediate terms
        # reported by LCCs to compute bnd_flow_h and bnd_flow_H
        bnd_flow_terms = pd.DataFrame(np.zeros((flow_meas_info.shape[0] // 2, 6)))
        # if <secure> is True, change the type of <bnd_flow_terms>'s entries to hold ciphertext objects
        if secure:
            bnd_flow_terms = bnd_flow_terms.astype(np.object)

        # generate information about other types of boundary measurements
        # tie-line current magnitude measurements
        meas_idx = list(i_meas.index)
        meas_side = list(i_meas.side)
        related_line = tie_line.loc[i_meas.element]
        related_line_from_bus = bnd_bus.loc[related_line.from_bus]
        related_line_to_bus = bnd_bus.loc[related_line.to_bus]
        related_bus = pd.concat([related_line_from_bus, related_line_to_bus])
        bus_side = ['from'] * related_line_from_bus.shape[0] + ['to'] * related_line_to_bus.shape[0]
        this_side = np.array(meas_side * 2) == bus_side
        i_meas_info = pd.DataFrame({'measurement': meas_idx * 2,
                                    'bus': list(related_bus.name),
                                    'side': list(this_side),
                                    'related_p_meas': related_p_meas_idx * 2,
                                    'related_q_meas': related_q_meas_idx * 2,
                                    'zone': list(related_bus.zone)})
        # boundary bus power injection measurements
        meas_idx_ = []
        meas_type_ = []
        bus_idx_ = []
        bus_zone_ = []
        this_side_ = []
        related_flow_meas_idx_ = []
        bus_lookups = list(bnd_bus.name)
        for idx, flow_meas in zip(list(inj_meas.index), related_flow_meas_idx):
            meas_idx_.extend([idx] * (len(flow_meas) + 1))
            meas_type_.extend([inj_meas.measurement_type[idx]] * (len(flow_meas) + 1))
            bus = []
            for f_idx in flow_meas:
                related_bus = list(flow_meas_info.loc[f_idx].bus)
                for b in related_bus:
                    if b == bus_lookups[inj_meas.element[idx]]:
                        if b not in bus:
                            bus.append(b)
                            related_flow_meas_idx_.append(flow_meas)
                    else:
                        bus.append(b)
                        related_flow_meas_idx_.append([f_idx])
            bus_idx_.extend(bus)
            this_side_.extend([b == bus_lookups[inj_meas.element[idx]] for b in bus])
            bus_zone_.extend([bnd_bus.zone[bus_lookups.index(b)] for b in bus])
        inj_meas_info = pd.DataFrame({'measurement': meas_idx_,
                                      'bus': bus_idx_,
                                      'side': this_side_,
                                      'measurement_type': meas_type_,
                                      'related_flow_meas': related_flow_meas_idx_,
                                      'zone': bus_zone_})
        # boundary bus voltage magnitude measurements
        v_meas = bnd_meas.loc[(bnd_meas.element_type == 'bus') & (bnd_meas.measurement_type == 'v')]
        related_bus = bnd_bus.loc[v_meas.element]
        v_meas_info = pd.DataFrame({'measurement': list(v_meas.index),
                                    'bus': list(related_bus.name),
                                    'zone': list(related_bus.zone)})
        # partition <bnd_meas_info> by zone
        for zone in zone_idx:
            bnd_meas_info[zone]['flow'] = flow_meas_info.loc[flow_meas_info.zone == zone]
            bnd_meas_info[zone]['current'] = i_meas_info.loc[i_meas_info.zone == zone]
            bnd_meas_info[zone]['injection'] = inj_meas_info.loc[inj_meas_info.zone == zone]
            bnd_meas_info[zone]['voltage'] = v_meas_info.loc[v_meas_info.zone == zone]
        return bnd_meas_info, bnd_flow_terms

    def print_statistics(self):
        n_all_bus = len(self.net.bus)
        n_bnd_bus = len(self.boundary_bus_idx)
        n_int_bus = n_all_bus - n_bnd_bus
        n_all_branch = len(self.net.line) + len(self.net.trafo)
        n_tie_line = len(self.tie_line_idx)
        n_int_line = n_all_branch - n_tie_line
        n_all_meas = len(self.net.measurement)
        net = self.net
        measurement = self.net.measurement
        all_bnd_bus = net.bus.loc[net.bus.index.isin(self.boundary_bus_idx)]
        all_tie_line = net.line.loc[net.line.index.isin(self.tie_line_idx)]
        all_bnd_bus_meas = measurement.loc[(measurement.element_type == 'bus') &
                                           measurement.element.isin(all_bnd_bus.index)]
        all_tie_line_meas = measurement.loc[(measurement.element_type == 'line') &
                                            measurement.element.isin(all_tie_line.index)]
        n_bnd_meas = len(all_bnd_bus_meas) + len(all_tie_line_meas)
        n_int_meas = n_all_meas - n_bnd_meas
        table = PrettyTable()
        table.field_names = ['', 'Bus', 'Branch', 'Measurement']
        table.align[''] = 'l'
        table.align['Bus'] = 'r'
        table.align['Branch'] = 'r'
        table.align['Measurement'] = 'r'
        table.add_row(['All', n_all_bus, n_all_branch, n_all_meas])
        table.add_row(['Boundary', n_bnd_bus, n_tie_line, n_bnd_meas])
        table.add_row(['Internal', n_int_bus, n_int_line, n_int_meas])
        print(table)
        pass
