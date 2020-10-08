import copy
import argparse
from datetime import datetime
import gamma_env as gamma
import glob
import os, sys
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)
from utils.utils import *
fitness_list = None
fitness = None
stage_idx = 0
prev_stage_value = []
tune_iter = 1

def save_chkpt(layerid_to_dim,dim_info,dim_set,first_stage_value=None):
    chkpt = {
        "layerid_to_dim": layerid_to_dim,
        "dim_info": dim_info,
        "dim_set": dim_set,
        "first_stage_value":first_stage_value
    }
    with open(chkpt_file, "wb") as fd:
        pickle.dump(chkpt, fd)




def train_model(model_defs,stages=2):
    layerid_to_dim = {}
    dim_infos = {}
    fitness_list = [opt.fitness1, opt.fitness2]
    global fitness,prev_stage_value, tune_iter, stage_idx
    dim_set = set((tuple(m) for m in model_defs))
    dimension = model_defs[0]

    env = gamma.GAMMA(dimension=dimension, num_pe=opt.num_pe, fitness=fitness, par_RS=opt.parRS,
                      l1_size=opt.l1_size,
                      l2_size=opt.l2_size, NocBW = opt.NocBW, slevel_min=opt.slevel_min, slevel_max=opt.slevel_max,
                      fixedCluster=opt.fixedCluster, log_level=opt.log_level)

    for i, dim in enumerate(model_defs):
        layerid_to_dim[i] = dim
    for s in range(stages):
        dim_stage = {}
        for dimension in dim_set:
            dim_stage[dimension] = {"best_reward": [float("-Inf") for _ in range(s+1)]}
        dim_infos["Stage{}".format(s+1)] = copy.deepcopy(dim_stage)

    for s in range(stages):
        stage_idx = s
        dim_list = list(dim_set)
        fitness = fitness_list
        chkpt_list = []
        for i, dimension in enumerate(dim_list):
            env.reset_dimension(dimension=dimension, fitness=fitness)
            chkpt_list.append(env.run(dimension, stage_idx=stage_idx, num_population=opt.num_pop,prev_stage_value=prev_stage_value,
                   num_generations=opt.epochs,best_sol_1st=dim_infos["Stage{}".format(s)][dimension]["best_sol"] if s!=0 else None))

        for i, chkpt in enumerate(chkpt_list):
            best_reward = chkpt["best_reward"]
            cur_best_reward =  dim_infos["Stage{}".format(s+1)][dim_list[i]]["best_reward"]
            if cur_best_reward[s] <= best_reward[s]:
                dim_infos["Stage{}".format(s+1)][dim_list[i]] = chkpt
        save_chkpt(layerid_to_dim, dim_infos, dim_set)
        if s+1==stages:
            return
        cur_stage_value = 0
        for dim in dim_set:
            cur_best_reward =  dim_infos["Stage{}".format(s+1)][dim]["best_reward"]
            cur_stage_value = min(cur_stage_value, cur_best_reward[stage_idx])
        prev_stage_value.append(cur_stage_value)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitness1', type=str, default="latency", help='first stage fitness')
    parser.add_argument('--fitness2', type=str, default="power", help='second stage fitness')
    parser.add_argument('--stages', type=int, default=2,help='number of stages', choices=[1,2])
    parser.add_argument('--num_pop', type=int, default=100,help='number of populations')
    parser.add_argument('--parRS', default=False, action='store_true', help='Parallize across R S dimension')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--outdir', type=str, default="outdir", help='output directiory')
    parser.add_argument('--num_pe', type=int, default=168, help='number of PEs')
    parser.add_argument('--l1_size', type=int, default=512, help='L1 size')
    parser.add_argument('--l2_size', type=int, default=108000, help='L2 size')
    parser.add_argument('--NocBW', type=int, default=8192000, help='NoC BW')
    parser.add_argument('--hwconfig', type=str, default=None, help='HW configuration file')
    parser.add_argument('--model', type=str, default="vgg16", help='Model to run')
    parser.add_argument('--num_layer', type=int, default=0, help='number of layers to optimize')
    parser.add_argument('--singlelayer', type=int, default=2, help='The layer index to optimize')
    parser.add_argument('--slevel_min', type=int, default=2, help='parallelization level min')
    parser.add_argument('--slevel_max', type=int, default=2, help='parallelization level max')
    parser.add_argument('--fixedCluster', type=int, default=0, help='Rigid cluster size')
    parser.add_argument('--log_level', type=int, default=1, help='Detail: 2, runtimeinfo: 1')
    opt = parser.parse_args()
    opt = set_hw_config(opt)
    m_file_path = "../../data/model/"
    m_file = os.path.join(m_file_path, opt.model + ".csv")
    df = pd.read_csv(m_file)
    model_defs = df.to_numpy()
    if opt.singlelayer:
        model_defs=model_defs[opt.singlelayer-1:opt.singlelayer]
    else:
        if opt.num_layer:
           model_defs = model_defs[:opt.num_layer]
    _, dim_size = model_defs.shape
    now = datetime.now()
    now_date = "{}".format(now.date())
    now_time = "{}".format(now.time())
    outdir = opt.outdir
    outdir = os.path.join("../../", outdir)
    if opt.fixedCluster>0:
        exp_name = "GAMMA_{}_SL-{}-{}_F1-{}_F2-{}_PE-{}_GEN-{}_POP-{}_L1-{}_L2-{}".format(opt.model,opt.slevel_min, opt.slevel_max,opt.fitness1, opt.fitness2, opt.num_pe, opt.epochs, opt.num_pop, opt.l1_size, opt.l2_size)
    else:
        exp_name = "GAMMA_{}_SL-{}-{}_FixCl-{}_F1-{}_F2-{}_PE-{}_GEN-{}_POP-{}_L1-{}_L2-{}".format(opt.model,opt.slevel_min, opt.slevel_max,opt.fixedCluster, opt.fitness1, opt.fitness2, opt.num_pe, opt.epochs, opt.num_pop, opt.l1_size, opt.l2_size)
    outdir_exp = os.path.join(outdir, exp_name)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_exp, exist_ok=True)
    chkpt_file_t = "{}".format("result")
    chkpt_file = os.path.join(outdir_exp, chkpt_file_t + "_c.plt")

    try:
        train_model(model_defs, stages=opt.stages)
        print_result(chkpt_file)
    finally:
        for f in glob.glob("*.m"):
            os.remove(f)
        for f in glob.glob("*.csv"):
            os.remove(f)
