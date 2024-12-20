import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import numba as nb
import math
import random
import scipy
import networkx as nx

import itertools
import copy
import os
import gzip
import logging


def shift_ndarray(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

# @nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:]), nopython=True)
@nb.jit(nopython=True)
def extrap(x, xp, yp):
    """np.interp function with linear extrapolation"""
    y = np.interp(x, xp, yp)
    y[x < xp[0]] = yp[0] + (x[x<xp[0]]-xp[0]) * (yp[0]-yp[1]) / (xp[0]-xp[1])
    y[x > xp[-1]]= yp[-1] + (x[x>xp[-1]]-xp[-1])*(yp[-1]-yp[-2])/(xp[-1]-xp[-2])
    return y

# @nb.jit((nb.float64[:], nb.float64), nopython=True)
@nb.jit(nopython=True)
def fillna(x, fill_value):
    '''Numba currently does not support Boolean indexing of multidimensional vectors and needs to operate on each one-dimensional vector in a cyclic manner'''
    x[np.isnan(x)] = fill_value



class DTAParams(object):
    def __init__(self, capacity_factor, early_depart, pt_constant, pt_m_constant, pt_switch, pt_waiting, pt_time, pt_dist, walk_time, car_time, car_dist, price_dist, 
                 u_bound:np.ndarray=None, l_bound:np.ndarray=None):
        '''early_depart: per hour;
           pt_dist:      per meter;
        '''
        self.params = np.array([capacity_factor, early_depart, pt_constant, pt_m_constant, pt_switch, pt_waiting, pt_time, pt_dist, walk_time, car_time, car_dist, price_dist])
        self.params_index = dict(zip(['capacity_factor', 'early_depart', 'pt_constant', 'pt_m_constant', 'pt_switch', 'pt_waiting', 'pt_time', 'pt_dist', 'walk_time', 'car_time', 'car_dist', 'price_dist'],
                                     np.arange(len(self.params))))
        self.u_bound = np.zeros_like(self.params) + np.inf if u_bound is None else np.array(u_bound)
        self.l_bound = np.zeros_like(self.params) - np.inf if l_bound is None else np.array(l_bound)
        # if u_bound is not None:
        #     self.u_bound[:len(u_bound)] = np.array(u_bound)
        # if l_bound is not None:
        #     self.l_bound[:len(l_bound)] = np.array(l_bound)
        self.min_capacity_factor = max(0.005, self.l_bound[self.params_index['capacity_factor']])
        assert (self.u_bound >= self.l_bound).all()
        self.params[self.params < self.l_bound] = self.l_bound[self.params < self.l_bound]
        self.params[self.params > self.u_bound] = self.u_bound[self.params > self.u_bound]

    def __getitem__(self, item:str):
        return self.params[self.params_index[item]]

    def __setitem__(self, key:str, value:float):
        self.params[self.params_index[key]] = value        



class WithinDayLogitUnimodalDTA(object):
    '''Single mode logit-based DTA, including departure time and path choice'''
    def __init__(self, start_min:int, stop_min:int, interval:int, init_network:nx.DiGraph, road_name_map:dict, ordered_eff_links_r, demands, od_set, utility_params:DTAParams, 
                 max_advance_min=90, max_delay_min=90):
        self.name = None
        # configuration parameters
        assert 0 <= start_min <= stop_min <= 24*60 and 0 < interval <= stop_min-start_min and init_network.is_directed()
        self.start_min = int(start_min)
        self.stop_min = int(stop_min)
        self.interval = int(interval)
        self.time_index = np.arange(self.start_min, self.stop_min+self.interval, self.interval, dtype=np.float64)
        self.interval_num = len(self.time_index) - 1
        assert self.interval_num * self.interval == self.stop_min - self.start_min
        # utility parameters
        self.utility_params = utility_params
        self.init_capacity_factor = self.capacity_factor = self.utility_params['capacity_factor']
        self.min_capacity_factor = self.utility_params.min_capacity_factor
        # topological parameters
        self.network = copy.deepcopy(init_network)
        self.init_ordered_eff_links_r = self.ordered_eff_links_r = ordered_eff_links_r # effective links
        self.road_name_map = road_name_map # e.g. {'link_name': (from_node_id, to_node_id)}
        # demand parameters
        self.init_demand_profile, self.demand_profile = demands.copy(), demands.copy()
        self.init_od_set, self.od_set = od_set.copy(), od_set.copy()
        self.sample_destination = None
        self.sample_rate = 1
        # simulation parameters
        assert max_advance_min >= 0 and max_delay_min >= 0 and (max_advance_min % self.interval == max_delay_min % self.interval == 0)
        self.max_advance_min = max_advance_min
        self.max_delay_min = max_delay_min
        # simulation results
        self.mswa_errors = None 
        self.depart_counts = None   # veh/h
        self.hour_link_count = None # veh/h
        self.hour_link_time = None  # s
        self.hour_link_speed = None # m/s

    def setName(self, name:str):
        self.name = str(name)

    def subsampleDemands(self, destination_num, random_seed:int=None):
        '''Sampling of OD demands. Randomly select destination_num destinations.'''
        init_demand_size = self.init_demand_profile.sum()
        if random_seed is not None:
            random.seed(random_seed)
        self.sample_destination = random.sample(population=list(set(self.init_od_set[:,0])), k=destination_num)
        sample_od_index = np.isin(self.init_od_set[:,0], self.sample_destination)
        self.od_set = self.init_od_set[sample_od_index]
        self.demand_profile = self.init_demand_profile[sample_od_index]
        self.ordered_eff_links_r = self.init_ordered_eff_links_r[np.isin(self.init_ordered_eff_links_r[:,1], self.sample_destination)]
        sample_demand_size = self.demand_profile.sum()
        self.sample_rate = sample_demand_size / init_demand_size
        print(f'Sampling Ratio: {self.sample_rate:.2%}')
        self.capacity_factor = max(self.min_capacity_factor, self.init_capacity_factor * self.sample_rate)
    
    def resetDemand(self, demands:np.ndarray):
        assert demands.shape == self.init_demand_profile.shape
        self.init_demand_profile = demands.copy()
        if self.sample_destination is not None:
            sample_od_index = np.isin(self.init_od_set[:,0], self.sample_destination)
            self.demand_profile = self.init_demand_profile[sample_od_index]
            self.sample_rate = self.demand_profile.sum() / self.init_demand_profile.sum()
            self.capacity_factor = max(self.min_capacity_factor, self.init_capacity_factor * self.sample_rate)
        else: self.demand_profile = self.init_demand_profile.copy()

    def resetNetwork(self, network:nx.DiGraph, new_tidal_lane_data:np.ndarray=None, new_signal_control_data:pd.DataFrame=None, toll_links:list=None):
        self.network = copy.deepcopy(network)
        self.init_capacity_factor = self.utility_params['capacity_factor']
        self.capacity_factor = max(self.min_capacity_factor, self.init_capacity_factor * self.sample_rate)
        if new_tidal_lane_data is not None:
            tidal_x = np.round(new_tidal_lane_data[:,-1].astype(float), 0)
            for (tidal_links_1, tidal_links_2), tidal_policy in zip(new_tidal_lane_data[tidal_x != 0, :-1], tidal_x[tidal_x != 0]):
                increase_lane_link_ids, decrease_lane_link_ids = (tidal_links_1.split(','), tidal_links_2.split(',')) if tidal_policy == 1 else (tidal_links_2.split(','), tidal_links_1.split(','))
                for increase_lane_link_id in increase_lane_link_ids:
                    from_node_id, to_node_id = self.road_name_map[increase_lane_link_id]
                    increase_lane_link = self.network[from_node_id][to_node_id]
                    increase_lane_capacity = increase_lane_link['capacity'] / increase_lane_link['permlanes']
                    increase_lane_link['capacity'] += increase_lane_capacity
                for decrease_lane_link_id in decrease_lane_link_ids:
                    from_node_id, to_node_id = self.road_name_map[decrease_lane_link_id]
                    decrease_lane_link = self.network[from_node_id][to_node_id]
                    decrease_lane_link['capacity'] -= increase_lane_capacity
        if new_signal_control_data is not None:
            for from_node_id, to_node_id, link_attr in self.network.edges.data():
                if link_attr['type'] == 'road':
                    link_id = link_attr['link_id']
                    if link_id in new_signal_control_data['link'].values:
                        control_data = new_signal_control_data[new_signal_control_data['link'] == link_id]
                        link_attr['signal_plans'] = []
                        for start_sec, stop_sec, cycle_sec, green_sec in zip(control_data['start_time'], control_data['stop_time'], control_data['cycle_time'], control_data['green_time']):
                            start_min, stop_min = start_sec // 60, stop_sec // 60
                            link_attr['signal_plans'].append(dict(start_min=start_min, stop_min=stop_min, cycle_sec=cycle_sec, green_sec=green_sec))
                    else: link_attr.pop('signal_plans', None)
        if toll_links is not None:
            price_per_meter = self.utility_params['price_dist']
            for from_node_id, to_node_id, link_attr in self.network.edges.data():
                if link_attr['type'] == 'road':
                    link_attr['price'] = price_per_meter * link_attr['length'] if link_attr['link_id'] in toll_links else 0.0
    
    def _createAdvanceDelayCostMatrix(self, max_advance_min, max_delay_min):
        '''max_advance_min: Maximum advance departure time (min)；max_delay_min: Maximum delay departure time (min)'''
        matrix = np.tri(self.interval_num) - np.tri(self.interval_num).T
        diagonal_index = np.arange(self.interval_num)
        matrix[diagonal_index, diagonal_index] = 1e-5 # set the diagonal elements to zeros
        advance_intervals = max_advance_min // self.interval
        if advance_intervals > 0:
            for i in range(1, advance_intervals+1): # set the marginal utility for early departure
                offset_diagonal_index = diagonal_index[i:]
                matrix[offset_diagonal_index, offset_diagonal_index - i] = -1 * self.utility_params['early_depart'] / 60
        delay_intervals = max_delay_min // self.interval
        if delay_intervals > 0:
            for j in range(1, delay_intervals+1): # set the marginal utility of delayed departure (i.e., zero)
                offset_diagonal_index = diagonal_index[:-1*j]
                matrix[offset_diagonal_index, offset_diagonal_index + j] = -1e-5
        return matrix

    def _initRoadLinkSignalPlan(self, link_attr:dict):
        '''initialize signal planning parameters for a road segment, including traffic capacity, signal cycle length, and red light duration ratio
        '''
        if 'capacity' in link_attr:
            link_attr['capacity'] *= self.capacity_factor * np.ones(shape=self.interval_num)
        link_attr['cycle_min'] = np.zeros(shape=self.interval_num)
        link_attr['red_rate'] = np.zeros(shape=self.interval_num)
        if 'signal_plans' in link_attr:
            for signal_plan in link_attr['signal_plans']:
                start_time_index = (max(self.start_min, signal_plan['start_min']) - self.start_min) // self.interval
                stop_time_index = (min(self.stop_min, signal_plan['stop_min']) - self.start_min) // self.interval
                red_rate = (signal_plan['cycle_sec'] - signal_plan['green_sec']) / signal_plan['cycle_sec']
                cycle_min = signal_plan['cycle_sec'] / 60
                link_attr['cycle_min'][start_time_index: stop_time_index] = cycle_min
                link_attr['red_rate'][start_time_index: stop_time_index] = red_rate
            link_attr['capacity'] *= (1 - link_attr['red_rate'])
        
    def initAdditionalNetworkLinksInfo(self):
        '''initialize signal planning parameters for the road segments of a network
        '''
        for from_node_id, to_node_id, link_attr in self.network.edges.data():
            if link_attr['type'] == 'road':
                self._initRoadLinkSignalPlan(link_attr)
    
    def _initUserFlow(self, link_user_flow, init_user_flow=0):
        '''initialize user flows for dynamic traffic assignment
        '''
        assert init_user_flow >= 0
        zero_flow = np.zeros(shape=self.interval_num)
        init_user_flow = init_user_flow + zero_flow
        for from_node_id, to_node_id, link_attr in self.network.edges.data():
            link_type = link_attr['type']
            if link_type == 'road':
                link_user_flow[(from_node_id, to_node_id)] = init_user_flow * self.capacity_factor

    @staticmethod
    # @nb.jit(nb.float64[:](nb.int64, nb.int64, nb.int64, nb.float64, nb.float64[:], nb.float64[:]), nopython=True)  
    @nb.jit(nopython=True)                           
    def _calculateRoadOverSaturatedTravelTime(start_min, stop_min, interval, freedspeed_min, equi_flow, capacity):
        last_travel_min = start_min + freedspeed_min
        travel_min = [last_travel_min]
        for time, ef, cap in zip(range(start_min+interval, stop_min+interval, interval), equi_flow, capacity):
            last_travel_min = max(time + freedspeed_min, last_travel_min + ef / cap * interval)
            travel_min.append(last_travel_min)
        return np.array(travel_min)
    
    def _updateEquivalentFlowAndTravelTime(self, link_user_flow, iter_num, iter_num_sum):
        '''Update the roads' travel time based on traffic flows. 
           The travel time is defined as the leaving time of passengers who enter the adjacent side at time t, simplified as a piecewise linear function.
           Update the equivalent flows based on the MSWA algorithm.
        '''
        global_max_flow_error = 0 
        mswa_factor = (iter_num_sum - iter_num) / iter_num_sum
        for from_node_id, to_node_id, link_attr in self.network.edges.data():
            if link_attr['type'] == 'road':
                new_link_equi_flow = link_user_flow[(from_node_id, to_node_id)].copy()
                max_flow_error = np.fabs(new_link_equi_flow - link_attr.get('equi_flow', 0)).max()
                global_max_flow_error = max_flow_error if max_flow_error > global_max_flow_error else global_max_flow_error
                link_attr['equi_flow'] = new_link_equi_flow
                link_user_flow[(from_node_id, to_node_id)] *= mswa_factor # user flow update
                # travel time
                freedspeed_min = link_attr['length'] / link_attr['freespeed'] / 60
                travel_min = self._calculateRoadOverSaturatedTravelTime(self.start_min, self.stop_min, self.interval, freedspeed_min,
                                                                        link_attr['equi_flow'], link_attr['capacity'])
                # dalay time caused by queueing at signalized intersections
                if link_attr['red_rate'].sum() > 0:
                    travel_min_diff = travel_min[1:] - travel_min[:-1]
                    travel_min_diff[travel_min_diff == 0] = np.inf
                    signal_active = link_attr['red_rate'] > 0
                    red_rate = link_attr['red_rate'][signal_active]
                    signal_delay = np.zeros_like(travel_min_diff)
                    signal_delay[signal_active] = 0.5 * red_rate**2 * link_attr['cycle_min'][signal_active] / \
                                                  (1 - (1-red_rate) * link_attr['equi_flow'][signal_active] * self.interval / link_attr['capacity'][signal_active] / travel_min_diff[signal_active])
                    signal_delay = np.concatenate(([0.5 * link_attr['red_rate'][0]**2 * link_attr['cycle_min'][0]], signal_delay))
                    link_attr['travel_min'] = travel_min + 0.5 * (signal_delay + shift_ndarray(signal_delay, -1, signal_delay[0]))
                else: link_attr['travel_min'] = travel_min
        return global_max_flow_error

    def _updateRouteUtility(self, links_attr_r):
        zeros = np.zeros(shape=self.interval_num + 1)
        for from_node_id, to_node_id, link_attr in self.network.edges.data():
            link_type = link_attr['type']
            if link_type == 'road':
                link_utility_r = (link_attr['travel_min'] - self.time_index) * self.utility_params['car_time'] / 60 + link_attr['length'] * self.utility_params['car_dist'] - link_attr.get('price', 0)
                links_attr_r[(from_node_id, to_node_id)] = (link_utility_r, link_attr['travel_min'].copy())
    
    @staticmethod
    # @nb.jit(nb.void(nb.types.DictType(nb.types.UniTuple(nb.int64, 3), nb.float64[::1]), 
    #                 nb.types.DictType(nb.types.UniTuple(nb.int64, 2), nb.float64[::1]),
    #                 nb.types.DictType(nb.types.UniTuple(nb.int64, 2), nb.types.UniTuple(nb.float64[::1], 2)),
    #                 nb.int64[:,:], nb.float64[:], nb.int64), nopython=True) 
    @nb.jit(nopython=True)
    def _logitImplicitPathChoice(link_probability, node_satisfaction, links_attr, ordered_eff_links, time_index, interval_num):
        theta = 1
        b, update_b = 0, False
        init_satisfaction = np.zeros(shape=interval_num + 1)
        current_satisfaction = init_satisfaction.copy()
        last_ordered_node, last_destination = -1, -1
        fse_nodes = nb.typed.List.empty_list(nb.int64)
        for ordered_node, destination, fse_node in ordered_eff_links:
            if last_destination != destination:
                if last_destination != -1:
                    for last_fse_node in fse_nodes:
                        link_probability[(last_destination, last_ordered_node, last_fse_node)] /= current_satisfaction
                    node_satisfaction[(last_destination, last_ordered_node)] = theta * (np.log(current_satisfaction) + b)
                node_satisfaction[(destination, destination)] = init_satisfaction
                current_satisfaction = init_satisfaction.copy()
                fse_nodes.clear()
                update_b = True
            elif last_ordered_node != ordered_node:
                for last_fse_node in fse_nodes:
                    link_probability[(destination, last_ordered_node, last_fse_node)] /= current_satisfaction
                node_satisfaction[(destination, last_ordered_node)] = theta * (np.log(current_satisfaction) + b)
                current_satisfaction = init_satisfaction.copy()
                fse_nodes.clear()
                update_b = True
            fse_nodes.append(fse_node)
            if (ordered_node, fse_node) not in links_attr:
                print((ordered_node, fse_node))
            utility, travel_min = links_attr[(ordered_node, fse_node)]
            fse_satisfaction = node_satisfaction[(destination, fse_node)]
            utility = (utility + np.interp(x=travel_min, xp=time_index, fp=fse_satisfaction)) / theta
            if update_b:
                b = utility.max() # avoid overflow caused by exceeding the upper and lower bounds of np.exp (≈ [-700, 700]) 
            exp_utility = np.exp(utility - b)
            exp_utility[exp_utility == 0] = 1e-322
            link_probability[(destination, ordered_node, fse_node)] = exp_utility
            current_satisfaction += exp_utility
            last_ordered_node, last_destination = ordered_node, destination
            update_b = False
        # last OD pair
        for last_fse_node in fse_nodes:
            link_probability[(destination, last_ordered_node, last_fse_node)] /= current_satisfaction
        node_satisfaction[(destination, last_ordered_node)] = theta * (np.log(current_satisfaction) + b)
    
    @staticmethod
    # @nb.jit(nb.void(nb.types.DictType(nb.types.UniTuple(nb.int64, 2), nb.float64[::1]), 
    #                 nb.types.DictType(nb.types.UniTuple(nb.int64, 2), nb.float64[::1]),
    #                 nb.int64[:,:], nb.float64[:,:], nb.types.DictType(nb.types.UniTuple(nb.int64, 3), nb.float64[::1]),
    #                 nb.int64, nb.float64[::1], nb.int64, nb.float64[:,:], nb.float64, nb.float64, nb.float64), nopython=True)
    @nb.jit(nopython=True)
    def _departureTimeChoice(node_satisfaction_r, od_set, demand_profile, depart_profile, interval, time_index, interval_num, advance_delay_cost_matrix):
        theta = 2
        departure_time_p_matrix = np.empty((1,1))
        departure_time_p_sum = np.empty(1)
        time_index_start = time_index[:-1]
        time_interval_index = time_index_start / interval
        advance_delay_cost_matrix_temp = advance_delay_cost_matrix * time_index_start.reshape((interval_num, 1))
        for (arrive_node, depart_node), idea_depart_profile in zip(od_set, demand_profile):
            satisfaction = node_satisfaction_r.pop((arrive_node, depart_node))
            satisfaction_diff = satisfaction[1:] - satisfaction[:-1]
            beta = (satisfaction_diff / interval + advance_delay_cost_matrix) / theta
            alpha = (satisfaction[:-1] - time_interval_index * satisfaction_diff - advance_delay_cost_matrix_temp) / theta +\
                    beta * time_index_start
            departure_time_p_matrix = (np.exp(alpha + beta * interval) - np.exp(alpha)) / beta # 元素 [i,j] 表示预期于 j 时刻出发的个体实际于 i 时刻出发的概率，概率尚未归一化
            for i in range(interval_num): # 将 NaN 的概率替换为 0
                fillna(departure_time_p_matrix[i], 0.)
            departure_time_p_sum = departure_time_p_matrix.sum(axis=0)
            depart_profile[(arrive_node, depart_node)] = np.dot(np.ascontiguousarray(idea_depart_profile), departure_time_p_matrix) / departure_time_p_sum
        node_satisfaction_r.clear()
    
    @staticmethod
    # @nb.jit(nb.void(nb.types.DictType(nb.types.UniTuple(nb.int64, 3), nb.float64[::1]), 
    #                 nb.types.DictType(nb.types.UniTuple(nb.int64, 3), nb.float64[::1]),
    #                 nb.int64[:,:], nb.types.DictType(nb.types.UniTuple(nb.int64, 3), nb.float64[::1]),
    #                 nb.types.DictType(nb.types.UniTuple(nb.int64, 2), nb.types.UniTuple(nb.float64[::1], 2)),
    #                 nb.float64[:], nb.int64, nb.int64, nb.int64, nb.int64, nb.int64), nopython=True)
    @nb.jit(nopython=True)
    def _dynamicNetworkLoadingAndMswaUpdate(depart_profile, link_user_flow, reversed_ordered_eff_links, link_probability, links_attr, time_index,
                                            interval, interval_num, iter_num, iter_num_sum):
        init_flow = np.zeros(shape=interval_num)
        cumsum_flow = np.empty(shape=interval_num+1)
        cumsum_flow[0] = 0 # 初始流量为 0
        last_ordered_node, last_destination = -1, -1
        for ordered_node, destination, fse_node in reversed_ordered_eff_links:
            if last_destination != destination or last_ordered_node != ordered_node:
                node_in_user_flow = depart_profile.pop((destination, ordered_node), default=init_flow)
            if node_in_user_flow is not init_flow:
                link_prob = link_probability.pop((destination, ordered_node, fse_node))
                _, travel_min = links_attr[(ordered_node, fse_node)]
                link_in_user_flow = node_in_user_flow * 0.5 * (link_prob[:-1] + link_prob[1:])
                # 基于累积流量曲线将定义在 travel_min 上的流量曲线（分段常函数）变换至 time_index 上，同时保持流量守恒
                cumsum_flow[1:] = link_in_user_flow.cumsum() # link_ou_user_flow.cumsum()
                cumsum_flow_interp = np.interp(x=time_index, xp=travel_min, fp=cumsum_flow)
                link_ou_user_flow = cumsum_flow_interp[1:] - cumsum_flow_interp[:-1]
                link_user_flow[(ordered_node, fse_node)] += link_in_user_flow * iter_num / iter_num_sum # MSWA 算法更新
                if fse_node != destination:
                    depart_profile[(destination, fse_node)] = depart_profile.get((destination, fse_node), default=init_flow) + link_ou_user_flow
            last_ordered_node, last_destination = ordered_node, destination
        link_probability.clear()
        links_attr.clear()
    
    @staticmethod
    # @nb.jit(nb.types.UniTuple(nb.float64, 2)(nb.types.DictType(nb.types.UniTuple(nb.int64, 3), nb.float64[::1])), nopython=True)
    @nb.jit(nopython=True)
    def _departureTimeStatistics(depart_profile, interval_num):
        depart_counts = np.zeros(shape=interval_num)
        for (arrive_node, depart_node), depart_user_flow in depart_profile.items():
            depart_counts += depart_user_flow
        return depart_counts
    
    def _getHourLinkStats(self, hour_link_df):
        '''flow = f_equi * init_capacity_factor / capacity_factor
        '''
        start_hour, end_hour = max(hour_link_df['Hour'].min(), self.start_min // 60), min(hour_link_df['Hour'].max(), self.stop_min // 60)
        hour_link_df = hour_link_df[(hour_link_df['Hour'] >= start_hour) & (hour_link_df['Hour'] <= end_hour)]
        dta_hour_link_count, dta_hour_link_time, dta_hour_link_speed = [], [], []
        for (from_node_id, to_node_id), hour_df in hour_link_df.groupby(by=['from_node', 'to_node']):
            hour_index = hour_df['Hour'].values - self.start_min // 60
            road_link_attr = self.network[from_node_id][to_node_id]
            
            hour_count = road_link_attr['equi_flow'] * self.init_capacity_factor / self.capacity_factor
            hour_count = hour_count.reshape(-1, 60 // self.interval).sum(axis=1)
            dta_hour_link_count.append(hour_count[hour_index])
            
            hour_time = (road_link_attr['travel_min'] - self.time_index) * 60
            hour_time = (hour_time[:-1] + hour_time[1:]) / 2
            hour_time = hour_time.reshape(-1, 60 // self.interval).mean(axis=1)[hour_index]
            dta_hour_link_time.append(hour_time)
            
            hour_speed = road_link_attr['length'] / hour_time
            dta_hour_link_speed.append(hour_speed)
        return np.hstack(dta_hour_link_count), np.hstack(dta_hour_link_time), np.hstack(dta_hour_link_speed)

    def _saveUserFlow(self, save_nx_path):
        for from_node_id, to_node_id, link_attr in self.network.edges.data():
            link_attr['final_equi_flow'] = link_attr['equi_flow'] / self.capacity_factor
            if 'capacity' in link_attr:
                link_attr['capacity'] /= self.capacity_factor
                if 'signal_plans' in link_attr:
                    link_attr['capacity'] /= (1 - link_attr['red_rate'])
        nx.write_gpickle(self.network, save_nx_path)

    def _saveResults(self, hour_link_df:pd.DataFrame=None, save_nx=False, save_nx_path=None, suffixes:str=None):
        suffixes = '' if suffixes is None else str(suffixes)
        if save_nx:
            assert save_nx_path is not None
            self._saveUserFlow(save_nx_path + suffixes)
        if hour_link_df is not None: 
            self.hour_link_count, self.hour_link_time, self.hour_link_speed = self._getHourLinkStats(hour_link_df)
            np.save(f'tmp/{self.name}-hour-link-stats{suffixes}',
                    np.concatenate((self.hour_link_count[:,None], self.hour_link_time[:,None], self.hour_link_speed[:,None]), axis=1))
        return suffixes

    def mswaOptimization(self, total_iter_num, mswa_d=0.5, start_iter_num=1, init_user_flow=0, max_tol_error=10, hour_link_df:pd.DataFrame=None, 
                         save_nx=False, save_nx_path=None):
        '''Logit-DTA using MSWA algorithm
        '''
        iter_num = start_iter_num**mswa_d
        iter_num_sum = sum((i**mswa_d for i in range(1,start_iter_num+1)))
        mswa_errors = []
        ## network attributes
        ordered_eff_links_r = self.ordered_eff_links_r
        reversed_ordered_eff_links_r = ordered_eff_links_r[::-1]
        links_attr_r = nb.typed.Dict.empty(key_type=nb.types.UniTuple(nb.int64, 2), value_type=nb.types.UniTuple(nb.float64[::1], 2))
        ## behavioral model temporary parameters
        # route choice
        node_satisfaction_r = nb.typed.Dict.empty(key_type=nb.types.UniTuple(nb.int64, 2), value_type=nb.float64[::1])
        link_probability_r = nb.typed.Dict.empty(key_type=nb.types.UniTuple(nb.int64, 3), value_type=nb.float64[::1])
        # departure time choice
        advance_delay_cost_matrix = self._createAdvanceDelayCostMatrix(self.max_advance_min, self.max_delay_min)
        depart_profile = nb.typed.Dict.empty(key_type=nb.types.UniTuple(nb.int64, 2), value_type=nb.float64[::1])
        ## dynamic demand loading temporary parameters
        link_user_flow = nb.typed.Dict.empty(key_type=nb.types.UniTuple(nb.int64, 2), value_type=nb.float64[::1])
        self._initUserFlow(link_user_flow, init_user_flow)
        self._updateEquivalentFlowAndTravelTime(link_user_flow, iter_num, iter_num_sum)
        for i in range(total_iter_num):
            self._updateRouteUtility(links_attr_r=links_attr_r)
            
            self._logitImplicitPathChoice(link_probability_r, node_satisfaction_r, links_attr_r, ordered_eff_links_r,
                                          self.time_index, self.interval_num)
            
            self._departureTimeChoice(node_satisfaction_r, self.od_set, self.demand_profile, depart_profile, self.interval, 
                                      self.time_index, self.interval_num, advance_delay_cost_matrix)
            depart_counts = self._departureTimeStatistics(depart_profile, self.interval_num)
            
            self._dynamicNetworkLoadingAndMswaUpdate(depart_profile, link_user_flow, reversed_ordered_eff_links_r, link_probability_r, links_attr_r,
                                                     self.time_index, self.interval, self.interval_num, iter_num=iter_num, iter_num_sum=iter_num_sum)
            depart_profile.clear()
            # error
            iter_num = (start_iter_num+i+1)**mswa_d
            iter_num_sum += iter_num
            mswa_error = self._updateEquivalentFlowAndTravelTime(link_user_flow, iter_num, iter_num_sum)
            mswa_error *= (iter_num_sum - iter_num) / (start_iter_num+i)**mswa_d
            logging.info(f'Iter Num: {start_iter_num+i:02d}; Max Error (veh/h): {mswa_error:.3f}')
            # print(f'Iter Num: {start_iter_num+i:02d}; Max Error (veh/h): {mswa_error:.3f}')
            mswa_errors.append(mswa_error)
            if mswa_error < max_tol_error:
                break
            
        self.mswa_errors = mswa_errors
        self.depart_counts = depart_counts.reshape(self.interval_num * self.interval // 60, 60 // self.interval).sum(axis=1)
        self._saveResults(hour_link_df, save_nx, save_nx_path)