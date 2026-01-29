#!/bin/bash

gpus="0"
aug=20
topk=20
beam_size=20
repos_beam=5
mask_beam=1
token_beam=4
max_tokens=5000   #!!!TODO: reduce this number if encountering CUDA OOM

databin=./datasets/USPTO_50K/aug20/data-bin  # binaried test data

root_dir=./results/finetune_50k #TODO: point to the checkpoint path
exp_n=1.0_2.0_0.43 # run_n
outfile=generation
output_dir=${root_dir}/generations/${exp_n}
mkdir -p ${output_dir}


#ckpt_path=${root_dir}/${exp_n}/checkpoints/checkpoint_best.pt

###!!! If you fintune the model yourself, uncomment the following codes to process the checkpoints.
 model_dir=${root_dir}/${exp_n}/checkpoints
 ckpt_name=finetune.pt
 ckpt_path=${model_dir}/${ckpt_name}
###!!!TODO: average multiple (dozens of) checkpoints to get better performance
 python ./utils/average_checkpoints.py --inputs ${model_dir} \
     --output ${ckpt_path} \
     --num-epoch-checkpoints 40 \
 	--checkpoint-upper-bound 40 \

CUDA_VISIBLE_DEVICES=$gpus fairseq-generate \
	--user-dir editretro \
	$databin \
	-s src -t tgt \
	-r ref \
  -f frag \
  --sim-lang sim\
	--gen-subset test \
	--task translation_retro \
	--path ${ckpt_path} \
	--iter-decode-max-iter 10 \
	--iter-decode-eos-penalty 0 \
	--beam 1 --remove-bpe \
	--init-src \
	--TOPK ${beam_size} \
	--max-tokens ${max_tokens} \
	--repos-beam ${repos_beam} \
	--mask-beam ${mask_beam} \
	--token-beam ${token_beam} \
	--print-step --retain-iter-history >${output_dir}/${outfile}.txt
  #	--fp16 \

# post processing
src=src.txt
tgt=tgt.txt
pred=pred.txt
prob=prob.txt
grep ^S ${output_dir}/${outfile}.txt | LC_ALL=C sort -V | cut -f2- > ${output_dir}/${src}
grep ^T ${output_dir}/${outfile}.txt | LC_ALL=C sort -V | cut -f2- > ${output_dir}/${tgt}
grep ^H ${output_dir}/${outfile}.txt | LC_ALL=C sort -V | cut -f3- > ${output_dir}/${pred}
grep ^P ${output_dir}/${outfile}.txt | LC_ALL=C sort -V | cut -f2- > ${output_dir}/${prob}


#生成json文件 对应关系
python ./utils/post_process.py \
    -generate_path  ${output_dir}/${pred} \
    -prob_path ${output_dir}/${prob} \
    -tgt_path ${output_dir}/${tgt} \
    -out_path ${output_dir}/${outfile}.json


# evaluate the results
python ./utils/score.py \
	-n_best ${topk} \
	-beam_size ${beam_size} \
	-predictions ${output_dir}/${outfile}.json \
	-targets ${output_dir}/${outfile}.json \
	-augmentation ${aug} \
	-score_alpha 0.1 \
	-metrics_file ${root_dir}/metrics/metrics_${exp_n}.csv
