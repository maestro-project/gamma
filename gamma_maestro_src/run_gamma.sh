cd ./GAMMA
python main.py --fitness1 latency --fitness2 power --stages 1 --num_pe 168 --l1_size 5120 --l2_size 1080000 --NocBW 81920000 --slevel_min 1 --slevel_max 2 --epochs 10 \
              --model vgg16 --singlelayer 1
cd ../






