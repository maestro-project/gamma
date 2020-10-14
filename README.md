# GAMMA #
This is the implementation of the paper [GAMMA: Automating the HW Mapping of DNN Models on
Accelerators via Genetic Algorithm](https://cpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/c/332/files/2020/08/gamma_iccad2020.pdf). 
GAMMA is an autonomous framework for optimizing the HW mapping of DNN models on the DNN Accelerators. This repository includes GAMMA, 
a specialized GA-based algorithm, a HW cost evaluation environment with a HW cost model, [MAESTRO](http://maestro.ece.gatech.edu/), embedded, 
and the other conventional optimization methods supported by [nevergrad](https://github.com/facebookresearch/nevergrad).

![GAMMA Framework](./others/gamma.jpg)


### Setup ###
* Clone Repo
```
git clone https://github.com/maestro-project/gamma.git
```
* Create virtual env
```
conda create --name gammaEnv python=3.6
conda activate gammaEnv
```
* Install requirement
```
pip install -r requirements.txt
```

* Download cost model and build symbolic link
```
python build.py
```

* Setup larger limitation for opened file if there is warning "Too many open files." (for threading)
```
ulimit -n 4096
```

### Run ###
* Run GAMMA
```
./run_gamma.sh
```
* Run GAMMA with HW configuration file
```
./run_gamma_with_hwconfig.sh
```
* Run other optimization methods
```
./run_others.sh
```
* Run other optimization methods with HW configuration file
```
./run_others_with_hwconfig.sh
```

#### Parameter ####
* fitness1: The first fitness objective (latency/ power/ energy)
* fitness2: The second fitness objective (latency/ power/ energy)
* stages: Number of stages, can choose from [1 or 2]
    * Choose 1, the system will only optimize on fitness1
    * Choose 2, the system will optimize on fitness1 and then fitness2 
* model: The model to run (available model in data/model)
* singlelayer: The layer index of the selected model, if want to optimize only single layer. If want to optimize all layers, skip this specification.
* num_pe: Number of PEs
* l1_size: L1 size (Bytes)
* l2_size: L2 size (Bytes)
* slevel_min: The minimum number of parallelism
* slevel_max: The maximum number of parallelism. The number of parallelism will be in the range [slevel_min, slevel_max]
* hwconfig: Read in HW configuration from file
* epochs: Number of generation for the optimization
* method: The optimization methods to choose from (PSO/ Portfolio/ OnePlusOne/ CMA/ DE/ TBPSA/ pureGA/ Random/ GA)
* outdir: The output result directory

##### To find out all the options
```
python main.py --help
```
### Note ###
* Support stable maestro version: commit id: f1df103

### Contributor ###
* Sheng-Chun (Felix) Kao
* Tushar Krishna

### Citation ###
```
@inproceedings{gamma,
    author       = {Kao, Sheng-Chun and Krishna, Tushar},
    title        = {{GAMMA: Automating the HW Mapping of DNN Models on Accelerators via Genetic Algorithm}},
    booktitle     = {ICCAD},
  year          = {2020}
}
```
