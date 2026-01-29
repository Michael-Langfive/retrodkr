#!/bin/bash

gpus='0'
lr=0.0001
max_tokens=16384
warmup=4000
update=1
max_epoch=40
max_update=100000
keep_last_epochs=40
noise_type=random_delete_shuffle
architecture=editretro_nat
task=translation_retro
loss=finetune_nat_loss

exp_n=finetune_50k
lambda1=1.0
lambda2=2.0
lambda3=0.43
run_n=1.0_2.0_0.43

root_dir=results
exp_dir=${root_dir}/${exp_n}
mkdir -p ${exp_dir}

model_dir=${exp_dir}/${run_n}/checkpoints
mkdir -p ${model_dir}

databin=datasets/USPTO_50K/aug20/data-bin

### Point to the processed pretrained checkpoint
ckpt_name=ckpt/pretrain.pt   #TODO: point to the pretrain checkpoint path

gpu_ids=$(echo $gpus | sed "s/,/ /g")
gpu_n=$(echo ${gpu_ids} | wc -w)

CUDA_VISIBLE_DEVICES=$gpus CUDA_LAUNCH_BLOCKING=1 fairseq-train \
	$databin \
  --user-dir editretro \
	-s src \
	-t tgt \
  -r ref \
  -f frag \
  --sim-lang sim\
  --lambda1 ${lambda1} \
  --lambda2 ${lambda2} \
  --lambda3 ${lambda3} \
  --save-dir ${model_dir} \
	--ddp-backend=no_c10d \
	--task ${task} \
	--criterion ${loss} \
	--arch ${architecture} \
	--noise ${noise_type} \
	--optimizer adam --adam-betas '(0.9,0.98)' \
	--lr ${lr} --lr-scheduler inverse_sqrt --min-lr '1e-09' \
	--warmup-updates ${warmup} --warmup-init-lr '1e-07' \
	--label-smoothing 0.1 \
	--dropout 0.2 --attention-dropout 0.2 \
	--weight-decay 0.01 \
	--share-all-embeddings \
	--decoder-learned-pos --encoder-learned-pos \
	--max-tokens-valid 4000 \
	--log-format 'simple' \
	--log-interval 200 \
	--fixed-validation-seed 7 \
	--max-tokens ${max_tokens} \
	--keep-last-epochs ${keep_last_epochs} \
	--max-epoch ${max_epoch} \
	--max-update ${max_update} \
	--alpha-ratio 0.5 \
	--dae-ratio 0.5 \
	--fp16  \
	--update-freq ${update} \
	--reset-optimizer --reset-lr-scheduler --reset-meters --reset-dataloader \
	--pretrained-ckpt ${ckpt_name} \
	--distributed-world-size ${gpu_n} > ${model_dir}/finetune_50k.log
  #--restore-file ${ckpt_name} \ 加载模型      倒数第二个参数是pretrained参数匹配
  #	--save-interval-updates 10000 \ 每隔10000步保存一次