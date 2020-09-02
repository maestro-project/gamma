cd ./src/GAMMA
python main.py --fitness1 latency --fitness2 power --stages 1 --slevel_min 1 --slevel_max 2 --epochs 10 \
              --model_def vgg16 --singlelayer 1 --num_layer 2 --hwconfig hw_config.m
cd ../..






