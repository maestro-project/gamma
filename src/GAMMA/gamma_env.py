
import numpy as np
import copy, random
import os
from subprocess import Popen, PIPE
import pandas as pd
from multiprocessing.pool import Pool

m_type_dicts = {0:"CONV", 1:"CONV", 2:"DSCONV", 3:"CONV"}
class GAMMA(object):
    def __init__(self,dimension, num_pe=64, fitness="latency",par_RS=False, l1_size=512, l2_size=108000, NocBW=81920000, slevel_min=2,slevel_max=2, fixedCluster=0, log_level=2):
        super(GAMMA,self).__init__()
        self.dimension = dimension
        self.dimension_dict = {"K":dimension[0], "C":dimension[1], "Y":dimension[2], "X":dimension[3], "R":dimension[4],"S":dimension[5], "T":dimension[6]}
        self.lastcluster_dict = {"K":dimension[0], "C":dimension[1], "Y":dimension[2], "X":dimension[3], "R":dimension[4],"S":dimension[5], "T":dimension[6]}
        dst_path = "../../cost_model/maestro"

        maestro = dst_path
        self._executable = "{}".format(maestro)
        self.out_repr = set(["K", "C", "R", "S"])
        self.num_pe = num_pe
        self.fitness = fitness
        self.cluster_space = ["K", "C", "Y","X","R","S"] if par_RS else ["K", "C", "Y","X"]
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.NocBW = NocBW
        self.slevel_min = slevel_min
        self.slevel_max = slevel_max
        self.fixedCluster = fixedCluster
        self.log_level = log_level
    def reset_dimension(self, dimension, fitness):
        self.dimension = dimension
        self.fitness = fitness
    def create_genome(self, uni_base=False,last_cluster_dict=None):
        K,C,Y,X,R,S,T = [1]*len(self.dimension)  if uni_base else self.dimension
        sp = random.choice(self.cluster_space)
        lastcluster_sz = last_cluster_dict[sp] if last_cluster_dict else self.dimension_dict[sp]
        if uni_base == True and self.fixedCluster > 0:
            sp_sz = self.fixedCluster
        else:
            sp_sz = min(lastcluster_sz, self.num_pe)
        df = [["K", random.randint(1, K)], ["C", random.randint(1, C)], ["Y", random.randint(1, Y)],["X", random.randint(1, X)], ["R", random.randint(1, R)], ["S", random.randint(1, S)]]
        idx = np.random.permutation(len(df))
        return [[sp, sp_sz]] + [df[i] for i in idx]

    def create_genome_fixedSL(self):
        ind = self.create_genome()
        for _ in range(self.slevel_min-1):
            ind = self.born_cluster_ind(ind)

        return ind

    def select_parents(self, pop, fitness, num_parents, num_population, stage_idx=0, first_stage_value=None):
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.

        if stage_idx==0:
            idx = np.argsort(fitness[:,0])[::-1]
            new_pop = [pop[i] for i in idx][:num_population]
            new_fitness = fitness[idx][:num_population]
            parents = copy.deepcopy(new_pop[:num_parents])
        else:
            elite_pop = [pop[i] for i in range(len(pop)) if all([fitness[i][kk]>=first_stage_value[kk] for kk in range(len(first_stage_value))])]
            elite_fitness = fitness
            for kk in range(len(first_stage_value)):
                elite_fitness = elite_fitness[(elite_fitness[:, kk] >= first_stage_value[kk])]
            idx = np.argsort(elite_fitness[:, stage_idx])[::-1]
            elite_pop = [elite_pop[i] for i in idx][:num_population]
            parents = copy.deepcopy(elite_pop[:num_parents])  if len(elite_pop)>0 else copy.deepcopy(pop[:num_parents])
            new_pop = pop[:num_population]
            new_fitness = fitness[:num_population]

        return new_pop, new_fitness, parents

    def mutate_tile(self, pop, is_finetune=True, num_mu_loc=1, alpha=0.5, range_alpha=0.5):
        for idx in range(len(pop)):
            indv = pop[idx]
            for _ in range(num_mu_loc):
                if random.random() < alpha:
                    pick = random.randint(0, len(indv) - 1)
                    if pick % 7 == 0:
                        last_cluster_dict = self.scan_indv(indv) if pick != 0 else None
                        sp = random.choice(self.cluster_space)
                        lastcluster_sz = last_cluster_dict[sp] if last_cluster_dict else self.dimension_dict[sp]
                        sp_sz = min(lastcluster_sz, self.num_pe) if self.fixedCluster <1 else self.fixedCluster
                        pop[idx][pick] = [sp, sp_sz]
                    else:
                        d, d_sz = indv[pick]
                        thr = self.dimension_dict[d]
                        if is_finetune:
                            sampling = np.random.uniform(-range_alpha, range_alpha, 1)
                            sampling = int(sampling * thr)
                            new_d_sz = d_sz + sampling
                            new_d_sz = max(1, min(new_d_sz, self.dimension_dict[d]))
                        else:
                            new_d_sz = random.randint(1, thr)
                        pop[idx][pick][1] = new_d_sz

    def swap_order(self, pop, alpha=0.5):
        max_count = len(pop)
        while max_count > 0:
            max_count -= 1
            if random.random()< alpha:
                idx = random.randint(0, len(pop)-1)
                sel_cluster = random.randint(0, (len(pop[idx])-1)//7)
                swap_id = np.random.randint(1, 6, (2,)) + sel_cluster * 7
                pop[idx][swap_id[0]], pop[idx][swap_id[1]] = pop[idx][swap_id[1]], pop[idx][swap_id[0]]


    def crossover_order(self, parents, pop, alpha=0.5):
        for idx in range(0,len(pop),2):
            dad, mom = parents[random.randint(0, len(parents) - 1)], parents[random.randint(0, len(parents) - 1)]
            dad = copy.deepcopy(dad)
            mom = copy.deepcopy(mom)
            length = min(len(dad), len(mom))
            for k in range(0, length, 7):
                if random.random() < alpha:
                    order_dicts = {"K":0,"C":0,"R":0,"S":0,"X":0,"Y":0}
                    dad_dicts =  {"K":0,"C":0,"R":0,"S":0,"X":0,"Y":0}
                    mom_dicts =  {"K":0,"C":0,"R":0,"S":0,"X":0,"Y":0}
                    for i in range(k+1, k+7):
                        d, d_sz = dad[i]
                        order_dicts[d] += i
                        dad_dicts[d] = d_sz
                        d, d_sz = mom[i]
                        order_dicts[d] += i
                        mom_dicts[d] = d_sz
                    dad[k + 1:k + 7] = [[mom[i][0], dad_dicts[mom[i][0]]] for i in range(k+1, k+7)]
                    mom[k + 1:k + 7] = [[dad[i][0], mom_dicts[dad[i][0]]] for i in range(k+1, k+7)]
            pop[idx] = dad
            pop[idx+1] = mom
    def crossover_tile(self, parents, pop, alpha=0.5):

        if len(parents) ==1:
            for idx in range(len(pop)):
                pop[idx] = copy.deepcopy(parents[0])
        else:
            for idx in range(0,len(pop),2):
                dad, mom = parents[random.randint(0, len(parents)-1)], parents[random.randint(0, len(parents)-1)]
                dad = copy.deepcopy(dad)
                mom = copy.deepcopy(mom)
                length = min(len(dad), len(mom))
                for k in range(0, length, 7):
                    if random.random() < alpha:
                        cross_point = random.choice(["K", "C", "Y", "X", "R", "S"])
                        for i in range(k+1, k+7):
                            d, d_sz = dad[i]
                            if d== cross_point:
                                dad_sz = d_sz
                            d, d_sz = mom[i]
                            if d == cross_point:
                                mom_sz = d_sz
                        for i in range(k+1, k+7):
                            d, d_sz = dad[i]
                            if d== cross_point:
                                dad[i] = [d, mom_sz]
                            d, d_sz = mom[i]
                            if d == cross_point:
                                mom[i] = [d, dad_sz]
                pop[idx] = dad
                if idx + 1 < len(pop):
                    pop[idx+1] = mom

    def born_cluster_ind(self, ind):
        if (len(ind)) // 7 < self.slevel_max:
            last_cluster_dict = self.scan_indv(ind)
            new_ind = ind + self.create_genome(uni_base=True, last_cluster_dict=last_cluster_dict)
            ind = new_ind
        return ind

    def born_cluster(self, pop, alpha=0.1):
        max_count = len(pop)
        while max_count > 0:
            max_count -= 1
            if random.random() < alpha:
                idx = random.randint(0, len(pop) - 1)
                ind = self.born_cluster_ind(pop[idx])
                pop[idx] = ind



    def kill_cluster(self, pop, alpha=0.5):
        max_count = len(pop)
        while max_count > 0:
            max_count -= 1
            if random.random() < alpha:
                idx = random.randint(0, len(pop) - 1)
                if (len(pop[idx]))//7>self.slevel_min:
                    pop[idx] = pop[idx][:-7]

    def scan_indv(self,indv):
        last_cluster_dict=self.lastcluster_dict
        for i in range(len(indv)-6,len(indv), 1):
            d, d_sz = indv[i]
            last_cluster_dict[d] = d_sz
        return  last_cluster_dict
    def get_out_repr(self, x):
        if x in self.out_repr:
            return x
        else:
            return x + "'"

    def run(self, dimension, stage_idx, prev_stage_value=0, num_population=100, num_generations=100, elite_ratio=0.05,
                       parents_ratio=0.15, ratio_decay=1, num_finetune=1, best_sol_1st=None):

        num_generations = num_generations
        num_population = num_population
        num_elite = int(num_population * elite_ratio)
        pool = Pool(num_population+ num_elite)

        best_reward_list = []
        best_reward = [-float("Inf") for _ in range(stage_idx + 1)]
        best_sol = None
        population = [self.create_genome_fixedSL() for _ in range(num_population)] if (
                    (stage_idx == 0) or (best_sol_1st is None)) else [best_sol_1st for _ in range(num_population)]
        fitness = np.ones((num_population, len(self.fitness)), float)
        num_parents = num_population
        for g in range(num_generations):
            finetine_iter = 1 if g < num_generations // 2 else num_finetune
            for f in range(finetine_iter):
                is_finetune = f > 0
                gen_best = -float("Inf")
                gen_best_idx = 0
                count_non_valid = 0
                if num_parents < 1:  # restart
                    population = [self.create_genome_fixedSL() for _ in range(num_population)] if (
                                (stage_idx == 0) or (best_sol_1st is None)) else [best_sol_1st for _ in
                                                                                  range(num_population)]
                    fitness = np.ones((num_population, stage_idx + 1), float)
                    print("Reinitialize population")
                    num_parents = num_population
                population, fitness, parents = self.select_parents(population, fitness, num_parents, num_population,
                                                                  stage_idx, first_stage_value=prev_stage_value)
                elite = copy.deepcopy(parents[:num_elite])
                elite_fitness = copy.deepcopy(fitness[:(len(elite))])
                if is_finetune:
                    self.mutate_tile(population, num_mu_loc=3, range_alpha=0.1, alpha=0.52, is_finetune=True)
                else:
                    self.crossover_tile(parents, population, alpha=0.57)
                    self.mutate_tile(population, num_mu_loc=3, range_alpha=0.53, alpha=0.52, is_finetune=False)
                    self.swap_order(population, alpha=0.47)
                    self.born_cluster(population, alpha=0.57)
                    self.kill_cluster(population, alpha=0.27)

                population = elite + population
                fitness = np.concatenate((elite_fitness, fitness))
                reward_list = pool.map(self.thread_fun, population)
                for i in range(len(population)):
                    reward =reward_list[i]
                    if reward is None or any(np.array(reward) >= 0):
                        reward = [float("-Inf") for _ in range(len(best_reward))]
                        count_non_valid += 1
                    elif stage_idx > 0:
                        if any([reward[kk] < prev_stage_value[kk] for kk in range(len(prev_stage_value))]):
                            reward = [float("-Inf") for _ in range(len(best_reward))]
                            count_non_valid += 1
                    judging_reward = reward[stage_idx]
                    fitness[i] = reward
                    if gen_best < judging_reward:
                        gen_best = judging_reward
                        gen_best_idx = i
                judging_best_reward = best_reward[stage_idx]
                if judging_best_reward < gen_best:
                    best_reward = copy.deepcopy(fitness[gen_best_idx])
                    best_sol = copy.deepcopy(population[gen_best_idx])

                num_parents = int(num_population * parents_ratio)
                num_parents = min(num_parents, len(population) - count_non_valid)
                parents_ratio *= ratio_decay
                best_reward_list.append(best_reward)
                chkpt = {
                    "best_reward": best_reward,
                    "best_reward_list": best_reward_list,
                    "best_sol": best_sol,
                    "num_population": num_population,
                    "num_generations": num_generations,
                    "fitness_use": self.fitness,
                    "num_pe": self.num_pe,
                    "l1_size": self.l1_size,
                    "l2_size": self.l2_size,
                    "NocBW": self.NocBW,
                    "dimension": dimension
                }
                if self.log_level==2:
                    print( "[Stage {}]Gen {}: Gen reward: {:3e}, 1st stage Reward: {}, Best reward: {}, Non_valid: {}".format(
                        stage_idx + 1, (g + 1), gen_best, np.abs(prev_stage_value), np.abs(best_reward), count_non_valid))
                elif self.log_level==1:
                    if stage_idx == 0:
                        print( "[Stage {}]Gen {}: Best reward: {}".format( stage_idx + 1, (g + 1), np.abs(best_reward)[0]))
                    else:
                        print( "[Stage {}]Gen {}:  1st stage Reward: {}, Best reward: {}".format(stage_idx + 1, (g + 1), np.abs(prev_stage_value), np.abs(best_reward)))
        return chkpt

    def thread_fun(self, individual):
        reward = self.oberserve_maestro(individual)
        return reward




    def write_maestro(self, indv, layer_id=0, m_file=None):
        m_type = m_type_dicts[int(self.dimension[-1])]
        with open("{}.m".format(m_file), "w") as fo:
            fo.write("Network {} {{\n".format(layer_id))
            fo.write("Layer {} {{\n".format(m_type))
            fo.write("Type: {}\n".format(m_type))
            fo.write("Dimensions {{ K: {:.0f}, C: {:.0f}, Y: {:.0f}, X: {:.0f}, R: {:.0f}, S: {:.0f} }}\n".format(*self.dimension))
            fo.write("Dataflow {\n")
            for k in range(0, len(indv), 7):
                for i in range(k, k+7):
                    d, d_sz = indv[i]
                    if i%7==0:
                        if k != 0:
                            fo.write("Cluster({},P);\n".format(d_sz))
                    else:
                        sp = "SpatialMap" if d == indv[k][0] else "TemporalMap"
                        fo.write("{}({},{}) {};\n".format(sp, d_sz, d_sz, self.get_out_repr(d)))
            fo.write("}\n")
            fo.write("}\n")
            fo.write("}")

    def oberserve_maestro(self, indv):
        m_file = "{}".format(random.randint(0, 2**32))
        self.write_maestro(indv,m_file=m_file)

        # print(num_pe, bw, l1_size)
        os.remove("./{}.csv".format(m_file)) if os.path.exists("./{}.csv".format(m_file)) else None
        command = [self._executable,
                   "--DFSL_file={}.m".format(m_file),
                   "--full_buffer=false", "--noc_bw={}".format(self.NocBW),
                   "--noc_hops=1", "--noc_hop_latency=1",
                   "--noc_mc_support=true", "--num_pes={}".format(self.num_pe),
                   "--num_simd_lanes=1", "--l1_size={}".format(self.l1_size),
                   "--l2_size={}".format(self.l2_size), "--print_res=false", "--print_res_csv_file=true", "--print_log_file=false", "--print_design_space=false", "--msg_print_lv=0"]


        process = Popen(command, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        process.wait()
        os.remove("./{}.m".format(m_file)) if os.path.exists("./{}.m".format(m_file)) else None
        try:
            df = pd.read_csv("./{}.csv".format(m_file))
            layer_name = df[" Layer Number"]
            runtime = np.array(df[" Runtime (Cycles)"]).reshape(-1, 1)
            throughput = np.array(df[" Throughput (MACs/Cycle)"]).reshape(-1, 1)
            energy = np.array(df[" Activity count-based Energy (nJ)"]).reshape(-1, 1)
            area = np.array(df[" Area"]).reshape(-1, 1)
            power = np.array(df[" Power"]).reshape(-1, 1)
            l1_size = np.array(df[" L1 SRAM Size (Bytes)"]).reshape(-1, 1)
            l2_size = np.array(df["  L2 SRAM Size (Bytes)"]).reshape(-1, 1)
            mac = np.array(df[" Num MACs"]).reshape(-1, 1)
            os.remove("./{}.csv".format(m_file))  if os.path.exists("./{}.csv".format(m_file)) else None
            os.remove("./log.txt") if os.path.exists("./log.txt") else None
            self.observation = [np.mean(x) for x in [runtime, throughput, energy, area, l1_size, l2_size, mac, power]]
            if len(str(stdout))>3:
                return None
            return self.judge()
        except:
            return None

    def judge(self):
        runtime, throughput, energy, area, l1_size, l2_size, mac, power = self.observation
        values = []
        for term in self.fitness:
            if term == "energy":
                reward = -energy
            elif term == "thrpt_ave":
                reward = throughput
            elif term == "EDP":
                reward = -energy * runtime
            elif term == "LAP":
                reward = -area * runtime
            elif term == "EAP":
                reward = -area * energy
            elif term == "thrpt" or term == "thrpt_naive":
                reward = throughput
            elif term == "thrpt_btnk":
                reward = throughput
            elif term == "latency":
                reward = -runtime
            elif term == "area":
                reward = -area
            elif term == "l1_size":
                reward = - l1_size
            elif term == "l2_size":
                reward = -l2_size
            elif term == "power":
                reward = -power
            else:
                raise NameError('Undefined fitness type')
            values.append(reward)
        return values
    def print_indv(self, indv,fd=False):
        for k in range(0, len(indv), 7):
            if fd:
                fd.write("\n{}".format(indv[k:k+7]))
            else:
                print(indv[k:k+7])