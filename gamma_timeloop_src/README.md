# GAMMA-TimeLoop #
Native GAMMA uses [MAESTRO](http://maestro.ece.gatech.edu/) as
HW cost model. GAMMA-Timeloop supports [Timeloop](https://github.com/NVlabs/timeloop.git) and uses it as HW cost model. 

---
### Setup Timeloop ###
Install and set up Timeloop. For more setup detail, please follow the setup user guide in [Timeloop](https://github.com/NVlabs/timeloop/blob/master/README.md).

Install the following dependencies.
```
scons
libconfig++-dev
libboost-dev
libboost-iostreams-dev
libboost-serialization-dev
libyaml-cpp-dev
libncurses-dev
libtinfo-dev
libgpm-dev
```

Clone the timeloop repository.
```
git clone ssh://path/to/timeloop.git
```

Place a symbolic link to the pat model like so:
```
cd timeloop/src
ln -s ../pat-public/src/pat .
cd ..
```

Install [Accelergy](http://accelergy.mit.edu). For more setup detail, please follow the setup user guide in [Accelergy](http://accelergy.mit.edu).
```
git clone https://github.com/Accelergy-Project/accelergy.git
cd accelergy
pip install .
```

Build Timeloop
```
scons --accelergy
```

Setup path for Timeloop
```
source [path to timeloop]/timeloop/env/setup-env.bash
export PATH="$PATH:[path to timeloop]/timeloop/build"
export LIBTIMELOOP_PATH="[path to timeloop]/timeloop"
```
---
### Run ###
Run GAMMA-Timeloop
```
./run_gamma_timeloop.sh
```

#### Parameter ####
* fitness: The fitness objective 
* model: The model to run (available model in data/model)
* layer_idx: The selected layer to run in the selected model. If want to optimize all layers, set layer_idx=-1.
* num_pes: Number of PEs
* l1_size: L1 size (Bytes)
* l2_size: L2 size (Bytes)
* epochs: Number of generation for the optimization

##### To find out all the options
```
python main.py --help
```

##### An example of output messages when executing GAMMA-Timeloop
```
more example_outputs/example.txt
```
---
### Gamma-Timeloop v.s. Timeloop's native search algorithm
Please refer to [comparisons_with_timeloop](./comparisons_with_timeloop).
