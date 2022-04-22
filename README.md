# GAMMA #
This is the implementation of the paper [GAMMA: Automating the HW Mapping of DNN Models on
Accelerators via Genetic Algorithm](https://cpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/c/332/files/2020/08/gamma_iccad2020.pdf). 
GAMMA is an autonomous framework for optimizing the HW mapping of DNN models on the DNN Accelerators. This repository includes GAMMA, 
a specialized GA-based algorithm, a HW cost evaluation environment with a HW cost model, [MAESTRO](http://maestro.ece.gatech.edu/) or [Timeloop](https://github.com/NVlabs/timeloop), embedded, 
and the other conventional optimization methods supported by [nevergrad](https://github.com/facebookresearch/nevergrad).

![GAMMA Framework](./gamma_maestro_src/figures/gamma.jpg)



### Gamma-Maestro
* The native GAMMA algorithm utilizes MAESTRO as cost model, now named [Gamma-Maestro](./gamma_maestro_src), can be found in [gamma_maestro_src](./gamma_maestro_src) directory. It searches through the design space of MAESTRO and proposes an optimized mapping.
  

### Gamma-Timeloop
* We add [Timeloop](https://github.com/NVlabs/timeloop) support, named [Gamma-Timeloop](https://github.com/maestro-project/gamma-timeloop).
  It enables using GAMMA algorithm to search through the design space of Timeloop, a DNN cost model from NVIDIA.

---
### Set-up
* Create virtual env
```
conda create --name gammaEnv python=3.6
conda activate gammaEnv
```
* Install requirement
```
pip install -r requirements.txt
```
---
### Resources
* Tutorial of GAMMA, in IEEE/ACM International Symposium on Microarchitecture (MICRO), 2020 [[video](https://www.youtube.com/watch?v=gfBFRBbcA10)]
* Main paper presentation, in IEEE/ACM International Conference On Computer Aided Design (ICCAD), 2020 [[video](https://www.youtube.com/watch?v=Q7oJBJmVbGw)] 
* Main paper: GAMMA: Automating the HW Mapping of DNN Models on Accelerators via Genetic Algorithm, ICCAD, 2020 [[paper](https://cpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/c/332/files/2020/08/gamma_iccad2020.pdf)]

### Contributor ###
* Sheng-Chun (Felix) Kao
* Tushar Krishna

### Citation ###
```
@inproceedings{gamma,
  author    = {Sheng{-}Chun Kao and
               Tushar Krishna},
  title     = {{GAMMA:} Automating the {HW} Mapping of {DNN} Models on Accelerators
               via Genetic Algorithm},
  booktitle = {{IEEE/ACM} International Conference On Computer Aided Design, {ICCAD}},
  pages     = {44:1--44:9},
  publisher = {{IEEE}},
  year      = {2020},
}
```
