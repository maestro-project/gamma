#!/bin/bash
GAMMADIR=../../
run_exp () {
  CURDIR=$(pwd)
  mkdir $CURDIR/$1
  mkdir $CURDIR/$1/runGamma
  mkdir $CURDIR/$1/runTimeloop
  mkdir $CURDIR/$1/runTimeloop_large
  cd $GAMMADIR/gamma_timeloop_src
  python main.py --model $5 --layer_idx $6 --num_pe $2 --l1_size $3 --l2_size $4 --num_pops 20 --epochs 30 --report_dir $CURDIR/$1/runGamma/

  cd $CURDIR/$1

  cp runGamma/arch.yaml ./runTimeloop
  cp runGamma/problem.yaml ./runTimeloop
  cp runGamma/arch.yaml ./runTimeloop_large
  cp runGamma/problem.yaml ./runTimeloop_large

  cd runTimeloop
  timeloop-mapper problem.yaml arch.yaml ../../mapper.yaml
  cd $CURDIR/$1

  cd runTimeloop_large
  timeloop-mapper problem.yaml arch.yaml ../../mapper_large.yaml
  cd $CURDIR
}
#run_exp outdir num_pe l1_size l2_size model layer_idx
run_exp exp_vgg16-1 256 100 8192 vgg16 1
run_exp exp_vgg16-2 256 100 8192 vgg16 2
run_exp exp_vgg16-3 256 100 8192 vgg16 3
run_exp exp_vgg16-5 256 100 8192 vgg16 5