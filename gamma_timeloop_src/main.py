from gamma_timeloop_env import GammaTimeloopEnv
import argparse
import os
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitness1', type=str, default="latency", help='1st order fitness objective')
    parser.add_argument('--fitness2', type=str, default="energy", help='2nd order fitness objective')
    parser.add_argument('--num_pops', type=int, default=5,help='number of populations')
    parser.add_argument('--epochs', type=int, default=5, help='number of generations/epochs')
    parser.add_argument('--num_pe', type=int, default=64, help='number of PEs')
    parser.add_argument('--l1_size', type=int, default=2**12, help='L1 size')
    parser.add_argument('--l2_size', type=int, default=2**14, help='L2 size')
    parser.add_argument('--model', type=str, default='example', help='DNN model')
    parser.add_argument('--layer_idx', type=int, default=-1, help='The layer to optimize')
    parser.add_argument('--report_dir', type=str, default='../report', help='The report directory')
    opt = parser.parse_args()
    opt.num_gens = opt.epochs
    m_file_path = "../data/model/"
    m_file = os.path.join(m_file_path, opt.model + ".csv")
    df = pd.read_csv(m_file)
    df = df.drop('T', axis=1)
    model_defs = df.to_numpy().tolist()
    model_defs = model_defs[opt.layer_idx-1:opt.layer_idx] if opt.layer_idx > 0 else model_defs
    fitness = [opt.fitness1]
    fitness.append(opt.fitness2) if opt.fitness2 is not None else None
    print(fitness)
    print(f'Fitness Objective: {fitness}')
    print('='*5 + f'Optimizing Model: {opt.model}' + '='*5)
    for layer_id in range(len(model_defs)):
        dimension = model_defs[layer_id]
        print('='*5 + f'Optimizing layer {layer_id+1}: {dimension}' + '='*5)
        gamma_timeloop = GammaTimeloopEnv(num_pes=opt.num_pe, l2_size=opt.l2_size, l1_size=opt.l1_size, fitness_obj=fitness, report_dir=opt.report_dir, use_pool=True)
        gamma_timeloop.run(dimension=dimension, num_pops=opt.num_pops, num_gens=opt.num_gens)
        print('-'*20)