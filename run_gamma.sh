cd ./src/GAMMA
python main.py --fitness1 latency --fitness2 power --stages 1 --num_pe 168 --l1_size 512 --l2_size 108000 --NocBW 81920000 --slevel_min 1 --slevel_max 2 --epochs 10 \
              --model_def vgg16 --singlelayer 1 --num_layer 2
cd ../..






