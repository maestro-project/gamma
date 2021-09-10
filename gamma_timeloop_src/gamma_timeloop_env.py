import numpy as np
import yaml
import os, sys
import copy
from functools import reduce
import random
from timeloop_env import TimeloopEnv
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import shutil

class GammaTimeloopEnv(object):
    def __init__(self, num_pes=256, l2_size=10800, l1_size=512, fitness_obj=['latency'], report_dir='./report', use_pool=True):
        self.fitness_obj = fitness_obj
        self.num_pes = num_pes
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.loc_to_dim_note = {0: 'K', 1: 'C', 2: 'Y', 3: 'X', 4: 'R', 5: 'S'}
        self.dim_note = ['K', 'C', 'Y', 'X', 'R', 'S']
        self.len_dimension = len(self.dim_note)
        self.timeloop_configfile_path = './out_config'
        self.report_dir = report_dir
        self.timeloop_env = TimeloopEnv(config_path=self.timeloop_configfile_path)
        self.use_pool = use_pool


    def set_dimension(self, dimension):
        self.dimension = dimension
        self.dimension_dict = self.get_dimension_dict(dimension)
        self.dimension_factor = self.get_dimension_factors(self.dimension_dict)

    def get_dimension_dict(self, dim_value):
        return {note: value for note, value in zip(self.dim_note, dim_value)}



    def get_factors(self, n):
        return list(reduce(list.__add__,
                          ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

    def get_dimension_factors(self, dimension_dict):
        dimension_factors = dict()
        for key, value in dimension_dict.items():
            factors = self.get_factors(value)
            dimension_factors[key] = factors
        return dimension_factors

    def mutate_tiles(self, pops, parents, alpha=0.5, num_mu_loc=1):
        len_parents = len(parents)
        for i in range(len(pops)):
            if random.random() < alpha:
                sel_parent = random.randint(0, len_parents - 1)
                indv = copy.deepcopy(parents[sel_parent])
                l2_tile_size = indv['l2_tile_size']
                l1_tile_size = indv['l1_tile_size']
                for _ in range(num_mu_loc):
                    pick_loc = random.randint(0, self.len_dimension - 1)
                    pick_dim = self.loc_to_dim_note[pick_loc]
                    dim_value = self.dimension_dict[pick_dim]
                    factors = self.dimension_factor[pick_dim]
                    pick_factor_l2 =  np.random.choice(factors)
                    pick_factor_l1 =  np.random.choice(self.get_factors(dim_value//pick_factor_l2))
                    l2_tile_size[pick_loc] = pick_factor_l2
                    l1_tile_size[pick_loc] = pick_factor_l1
                pops[i] = indv
        return pops
    def mutate_par(self, pops, parents, alpha=0.5):
        len_parents = len(parents)
        for i in range(len(pops)):
            if random.random() < alpha:
                sel_parent = random.randint(0, len_parents - 1)
                indv = copy.deepcopy(parents[sel_parent])
                pick_loc = random.randint(0, 1)
                par_dims = indv['par_dims']
                par_dims[pick_loc] = np.random.choice(self.dim_note)
                pops[i] = indv
        return pops

    def mutate_order(self, pops, parents, alpha=0.5):
        len_parents = len(parents)
        for i in range(len(pops)):
            if random.random() < alpha:
                sel_parent = random.randint(0, len_parents - 1)
                indv = copy.deepcopy(parents[sel_parent])
                if random.random()<0.5:
                    pick = 'l2_loop_order'
                else:
                    pick = 'l1_loop_order'
                loop_order = indv[pick]
                loop_order = list(loop_order)
                idxs = random.sample(set(np.arange(0, self.len_dimension)), 2)
                loop_order[idxs[0]], loop_order[idxs[1]] = loop_order[idxs[1]], loop_order[idxs[0]]
                indv[pick] = ''.join(loop_order)
                pops[i] = indv
        return pops

    def init_indv(self):
        indv = {'l2_tile_size': [1]*6,
                'l1_tile_size': [1]*6,
                'l2_loop_order': 'KCYXRS',
                'l1_loop_order': 'KCYXRS',
                'par_dims': ['K', 'C']}
        return indv

    def init_pops(self, num_pops):
        return [self.init_indv() for _ in range(num_pops)], np.ones((num_pops, len(self.fitness_obj))) * np.NINF

    def select_parents(self, pops, fitness, num_parents, num_elites, num_pops):
        fitness_list = [tuple(list(ar)+[-i]) for i, ar in enumerate(fitness)]
        fitness_list = sorted(fitness_list, reverse=True)
        idx = [int(-ar[-1]) for ar in fitness_list]
        new_pop = [pops[i] for i in idx][:num_pops]
        new_fitness = fitness[idx][:num_pops]
        parents = copy.deepcopy(new_pop[:num_parents])
        elites = copy.deepcopy(new_pop[:num_elites])
        elites_fitness = copy.deepcopy(new_fitness[:num_elites])
        return new_pop, new_fitness, parents, elites, elites_fitness

    def thread_fun(self, indv, pool_idx=0):
        self.timeloop_env.create_timeloop_config(self.dimension, self.l2_size, self.l1_size, self.num_pes,
                                                 indv['l2_tile_size'], indv['l1_tile_size'], indv['l2_loop_order'],
                                                 indv['l1_loop_order'], indv['par_dims'], pool_idx=pool_idx)
        fit = self.timeloop_env.run_timeloop(pool_idx=pool_idx, fitness_obj=self.fitness_obj)
        return fit

    def evaluate(self, pops, fitness, pool=None):
        if not pool:
            for i, indv in enumerate(pops):
                fitness[i] = self.thread_fun(indv)
        else:
            rets = pool.starmap(self.thread_fun, zip(pops, np.arange(len(pops))))
            fitness = np.array(rets)
        return fitness

    def create_timeloop_report(self, indv, dir_path='./report'):
        fitness = self.thread_fun(indv, pool_idx=0)
        os.makedirs(dir_path, exist_ok=True)
        os.system(f'cp -d -r {os.path.join(self.timeloop_configfile_path, "pool-0")}/* {dir_path}')
        with open(os.path.join(dir_path,'Gamma-Timeloop.txt'), 'w') as fd:
            fd.write(f'Achieved Fitness: {fitness}')
            fd.write(f'GammaTimeloop-style Sol: {self.get_genome(indv)}')
            fd.write(f'Gamma-style Sol: {self.get_maestro_style_genome(indv)}')

    def run(self, dimension, num_pops=100, num_gens=100, elite_ratio=0.05, parents_ratio=0.4):
        self.set_dimension(dimension)
        num_parents = int(num_pops*parents_ratio)
        num_elites = max(1, int(num_pops*elite_ratio))
        pops, fitness = self.init_pops(num_pops)
        if self.use_pool:
            pool = Pool(num_pops)
            self.timeloop_env.create_pool_env(num_pops)
        else:
            pool = None
        for g in range(num_gens):
            if g == 0:
                pops, fitness, parents, elites, elites_fitness = self.select_parents(pops, fitness, num_parents, num_elites, num_pops)
            pops = self.mutate_par(pops, parents)
            pops = self.mutate_order(pops, parents)
            pops = self.mutate_tiles(pops, parents)

            fitness = self.evaluate(pops, fitness, pool)
            pops = elites + pops
            fitness = np.concatenate((elites_fitness, fitness), axis=0)


            pops, fitness, parents, elites, elites_fitness = self.select_parents(pops, fitness, num_parents, num_elites, num_pops)

            best_idx = 0
            best_sol = pops[best_idx]
            print(f'[Gen{g}] fitness: {fitness[best_idx]} Sol: {self.get_genome(best_sol)}')
        print(f'Achieved Fitness: {fitness[best_idx]}')
        print(f'GammaTimeloop-style Sol: {self.get_genome(best_sol)}')
        print(f'Gamma-style Sol: {self.get_maestro_style_genome(best_sol)}')
        self.create_timeloop_report(best_sol, dir_path=self.report_dir)
        self.clean_timeloop_output_files()

    def get_genome(self, indv):
        l2_tile_size, l1_tile_size = indv['l2_tile_size'], indv['l1_tile_size']
        l2_loop_order, l1_loop_order = indv['l2_loop_order'],indv['l1_loop_order']
        l2_par, l1_par = indv['par_dims']
        l2_tile_dict = self.get_dimension_dict(l2_tile_size)
        l1_tile_dict = self.get_dimension_dict(l1_tile_size)
        genome_l2 = [[l2_par, self.num_pes]] + [[d, l2_tile_dict[d]] for d in l2_loop_order]
        genome_l1 = [[l1_par, 1]] + [[d, l1_tile_dict[d]] for d in l1_loop_order]
        genome = genome_l2 + genome_l1
        return genome

    def get_maestro_style_genome(self, indv):
        l2_tile_size, l1_tile_size = indv['l2_tile_size'], indv['l1_tile_size']
        l2_tile_size = [l2 * l1 for l2, l1 in zip(l2_tile_size, l1_tile_size)]
        l2_loop_order, l1_loop_order = indv['l2_loop_order'],indv['l1_loop_order']
        l2_par, l1_par = indv['par_dims']
        l2_tile_dict = self.get_dimension_dict(l2_tile_size)
        l1_tile_dict = self.get_dimension_dict(l1_tile_size)
        l1_cluster_size = l1_tile_dict[l1_par]
        l1_tile_dict[l1_par] = 1
        l2_cluster_size = self.num_pes // l1_cluster_size
        l2_tile_dict[l2_par] = max(1, l2_tile_dict[l2_par] // l2_cluster_size)
        genome_l2 = [[l2_par, self.num_pes]] + [[d, l2_tile_dict[d]] for d in l2_loop_order]
        genome_l1 = [[l1_par, l1_cluster_size]] + [[d, l1_tile_dict[d]] for d in l1_loop_order]
        genome = genome_l2 + genome_l1
        return genome


    def clean_timeloop_output_files(self):
        # out_prefix = "./timeloop-model."
        # output_file_names = []
        # output_file_names.append(out_prefix + "accelergy.log")
        # output_file_names.append(out_prefix + ".log")
        # output_file_names.append(out_prefix + "ART.yaml")
        # output_file_names.append(out_prefix + "ART_summary.yaml")
        # output_file_names.append(out_prefix + "ERT.yaml")
        # output_file_names.append(out_prefix + "ERT_summary.yaml")
        # output_file_names.append(out_prefix + "flattened_architecture.yaml")
        # output_file_names.append(out_prefix + "map+stats.xml")
        # output_file_names.append(out_prefix + "map.txt")
        # output_file_names.append(out_prefix + "stats.txt")
        # for f in output_file_names:
        #     if os.path.exists(f):
        #         os.remove(f)
        shutil.rmtree(self.timeloop_configfile_path)










