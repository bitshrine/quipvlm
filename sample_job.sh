#!/bin/bash
#SBATCH --nodes 1
#SBATCH --time 72:00:00
#SBATCH --qos gpu
#SBATCH --mem 64G
#SBATCH --gpus 2

cd quip
python llava.py llava-hf/llava-1.5-7b-hf llava_instruct_150k --wbits 4 --nsamples 1 --quant gptq --pre_gptqH --eval