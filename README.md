# GAMMA #



### Setup ###
* Create virtural env
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


* If want to try out other optimization methods
```
pip install nevergrad
```

* Setup larger limitation for opened file if there is warning "Too many open files." (for threading).
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
* stages: Number of stages
* num_pe: Number of PEs
* l1_size: SL size (Bytes)
* l2_size: SG size (Bytes)
* slevel_min: The minimum number of parallelism
* slevel_max: The maximum number of parallelism
* epochs: Number of generation for the optimization
* model_def: The model to run (available model in model_dir)
* num_layers: The number of layers to optimized in the selected models (type -1, if want to optimize all)
* singlelayer: The layer index of the selected model, if want to optimize only single layer (type -1, if want to optimize mulitple layers by num_layers)
* outdir: The output result directory
* method: The optimization methods to choose from (PSO/ Portfolio/ OnePlusOne/ CMA/ DE/ TBPSA/ pureGA/ Random/ GA)

##### To find out all the options
```
python main.py --help
```

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