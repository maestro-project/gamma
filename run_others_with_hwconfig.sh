cd ./src/Other_opts
python main.py --fitness1 latency --slevel 1 --epochs 10 \
              --model_def vgg16 --singlelayer 1 --num_layer 2 --hwconfig hw_config.m --method DE

cd ../..




