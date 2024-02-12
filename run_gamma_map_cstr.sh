cd ./src/GAMMA
python main.py --mapping_cstr dla_map --fitness1 latency --fitness2 power --num_pe 168 --l1_size 512 --l2_size 108000 --NocBW 81920000 --epochs 10 \
              --model vgg16 --outdir outdir_map
cd ../../






