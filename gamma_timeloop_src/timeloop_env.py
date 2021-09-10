import numpy as np
import yaml
import os, sys
import copy
from subprocess import Popen, PIPE, call
from parse_timeloop_output import parse_timeloop_stats
class TimeloopEnv(object):
    def __init__(self, config_path='./out_config'):
        base_config = './config_template'
        self.config_path = config_path
        with open(os.path.join(base_config, 'arch.yaml'), 'r') as fd:
            self.arch = yaml.load(fd, Loader = yaml.SafeLoader)
        with open(os.path.join(base_config, 'problem.yaml'), 'r') as fd:
            self.problem = yaml.load(fd,Loader = yaml.SafeLoader)
        with open(os.path.join(base_config, 'map.yaml'), 'r') as fd:
            self.map = yaml.load(fd,Loader = yaml.SafeLoader)
        os.makedirs(self.config_path, exist_ok=True)
        self._executable = 'timeloop-model'
        self.arch_path = [os.path.abspath(os.path.join(self.config_path, 'arch.yaml'))]
        self.problem_path = [os.path.abspath(os.path.join(self.config_path, 'problem.yaml'))]
        self.map_path = [os.path.abspath(os.path.join(self.config_path, 'map.yaml'))]
        self.pool_path = [os.path.abspath(self.config_path)]

    def get_timeloop_notation(self, g):
        timeloop_dict = {'N': 'N', 'K': 'M', 'C': 'C', 'Y': 'Q', 'X': 'P', 'R': 'S', 'S': 'R'}
        return timeloop_dict[g]

    def get_gamma_notation(self, t):
        gamma_dict = {'N': 'N','M': 'K','C': 'C','Q': 'Y','P': 'X','S': 'R','R': 'S'}
        return gamma_dict[t]

    def get_dimension_dict(self, dim_value):
        dim_note = 'KCYXRS'
        return {note: value for note, value in zip(dim_note, dim_value)}

    def get_tp_tile_size(self, dim_value):
        dim_note =  'KCYXRS'
        series = ['N=1'] + [f'{self.get_timeloop_notation(note)}={value}' for note, value in zip(dim_note, dim_value)]
        return ' '.join(series)

    def get_tp_sp_tile_size(self, dim_value, sp_dim):
        dim_note =  'KCYXRS'
        temporal_series = ['N=1'] + [f'{self.get_timeloop_notation(note)}={value if note not in sp_dim else 1}' for note, value in zip(dim_note, dim_value)]
        spatial_series =  ['N=1'] + [f'{self.get_timeloop_notation(note)}={value if note in sp_dim else 1}' for note, value in zip(dim_note, dim_value)]
        return ' '.join(temporal_series), ' '.join(spatial_series)

    def get_loop_order(self, loop_order):
        loop_order = 'N' + loop_order
        series = [self.get_timeloop_notation(g) for g in loop_order]
        return ''.join(series)

    def get_implicit_l3_tile_size(self, dim_value, l2_tile_size, l1_tile_size):
        l3_tile_size = [int(d/(l2*l1)) for d, l2, l1 in zip(dim_value, l2_tile_size, l1_tile_size)]
        l3_tile_size_mode = [d%(l2*l1) for d, l2, l1 in zip(dim_value, l2_tile_size, l1_tile_size)]
        if np.sum(l3_tile_size_mode) == 0:
            return l3_tile_size
        else:
            print('Tile size not divisible')
            return None


    def create_pool_env(self, num_pools):
        arch_paths, problem_paths, map_paths, pool_paths = [], [], [], []
        for i in range(num_pools):
            pool_dir = os.path.join(self.config_path, f'pool-{i}')
            os.makedirs(pool_dir, exist_ok=True)
            pool_paths.append(pool_dir)
            arch_paths.append(os.path.abspath(os.path.join(pool_dir, 'arch.yaml')))
            problem_paths.append(os.path.abspath(os.path.join(pool_dir, 'problem.yaml')))
            map_paths.append(os.path.abspath(os.path.join(pool_dir, 'map.yaml')))
        self.arch_path, self.problem_path, self.map_path, self.pool_path =  arch_paths, problem_paths, map_paths, pool_paths



    def create_timeloop_config(self, dimension, l2_size, l1_size, num_pes, l2_tile_size, l1_tile_size, l2_loop_order, l1_loop_order, par_dims, pool_idx=0):
        arch, problem, map = copy.deepcopy(self.arch), copy.deepcopy(self.problem), copy.deepcopy(self.map)
        timeloop_l3_tp_tiles = self.get_tp_tile_size(self.get_implicit_l3_tile_size(dimension, l2_tile_size, l1_tile_size))
        timeloop_l2_tp_tiles, timeloop_l2_sp_tiles = self.get_tp_sp_tile_size(l2_tile_size, par_dims)
        timeloop_l1_tp_tiles = self.get_tp_tile_size(l1_tile_size)
        timeloop_l2_order, timeloop_l1_order = self.get_loop_order(l2_loop_order), self.get_loop_order(l1_loop_order)
        arch['architecture']['subtree'][0]['subtree'][0]['local'][0]['attributes']['depth'] = l2_size
        arch['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['local'][0]['name']=f'RegisterFile[0..{num_pes}]'
        arch['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['local'][0]['attributes']['depth'] = l1_size
        arch['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['local'][1]['name']=f'MACC[0..{num_pes}]'
        dimension_dict = self.get_dimension_dict(dimension)
        for key, value in dimension_dict.items():
            problem['problem']['instance'][self.get_timeloop_notation(key)] = value
        map['mapping'][0]['factors'] = timeloop_l3_tp_tiles
        map['mapping'][1]['factors'] = timeloop_l2_tp_tiles
        map['mapping'][2]['factors'] = timeloop_l2_sp_tiles
        map['mapping'][3]['factors'] = timeloop_l1_tp_tiles
        map['mapping'][0]['permutation'] = timeloop_l2_order
        map['mapping'][1]['permutation'] = timeloop_l1_order
        map['mapping'][2]['permutation'] = timeloop_l1_order

        with open(self.arch_path[pool_idx], 'w') as fd:
            yaml.dump(arch, fd)
        with open(self.problem_path[pool_idx], 'w') as fd:
            yaml.dump(problem, fd)
        with open(self.map_path[pool_idx], 'w') as fd:
            yaml.dump(map, fd)

    def run_timeloop(self, fitness_obj=['latency'], pool_idx=0):
        command = [self._executable, self.arch_path[pool_idx], self.problem_path[pool_idx], self.map_path[pool_idx]]
        process = Popen(command, stdout=PIPE, stderr=PIPE, cwd=self.pool_path[pool_idx])
        stdout, stderr = process.communicate()
        process.wait()
        if stderr:
            return [-float('Inf')] * len(fitness_obj)
        else:
            stats = parse_timeloop_stats(self.pool_path[pool_idx])
            fitness = self.judge(stats, fitness_obj)
            return fitness


    def judge(self, stats, fitness_obj):
        ret = []
        for f in fitness_obj:
            if f == 'latency':
                ret.append(-stats['cycles'])
            if f == 'utilization':
                ret.append(stats['utilization'])
            if f == 'energy':
                ret.append(-stats['energy_pJ'])
        return ret

if __name__ == '__main__':
    l2_size = 2**14
    l1_size = 2**12
    num_pes = 64
    dimension = [32, 32, 16, 16, 3, 3]
    K,C,Y,X,R,S = dimension
    l2_tile_size = [8, 8, 2, 2, 3, 3]
    l1_tile_size = [4, 4, 8, 8, 1, 1]
    l2_loop_order = 'KCYXRS'
    l1_loop_order = 'YXKCRS'
    par_dims = 'KC'
    config_path = '/home/felix/Documents/my_code/timeloop-accelergy-exercises/workspace/exercises/2020.ispass/timeloop/04-model-conv1d+oc-3levelspatial/config'
    # timeloop = TimeloopEnv(config_path)
    # timeloop.create_timeloop_config(dimension, l2_size, l1_size, num_pes, l2_tile_size, l1_tile_size, l2_loop_order, l1_loop_order, par_dims)
    timeloop = TimeloopEnv()
    timeloop.create_timeloop_config(dimension, l2_size, l1_size, num_pes, l2_tile_size, l1_tile_size, l2_loop_order, l1_loop_order, par_dims)
    timeloop.run_timeloop()
