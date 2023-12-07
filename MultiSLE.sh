#!/bin/bash

model_name='SLE'
pool='avg'
batch_size=16
epochs=100
warm_strategy='none'
test_every=2
explain_every=4
lr=0.001
backbone_lr=0.0001
scheduler='step'
t_max=5
lr_step_size=10
lr_step_gamma=0.5

loss_func='weighted_ce'
weight_decay=0.001
loss_weight='sklearn'
loss_weight_batch=0
orth_type=3

num_classes=2
apply_tte=0
test_apply_tte=0
apply_sampler=0
recog_act=0
norm_pos=0
obs_interval=0
obs_len=16
pred_len=1

dataset_name='TITAN'
cross_dataset_name='JAAD'
small_set=0
seq_type='crossing'
img_norm_mode='torch'
color_order='BGR'
bbox_type='default'
cross_dataset=1
balance_train=0
max_occ=2
test_max_occ=2
min_w=0
min_h=70
test_min_w=0
test_min_h=0
overlap=0.5
dataloader_workers=8
augment_mode='random_hflip_crop'
pop_occl_track=1

fusion_mode=1
separate_backbone=1
conditioned_proto=1
conditioned_relevance=1
num_explain=5
num_proto_per_modality=10
proto_dim=256
simi_func='channel_att+linear'
pred_traj=0
freeze_base=0
freeze_proto=0
freeze_relev=0
softmax_t='1'
proto_activate='softmax'
multi_label_cross=0
use_atomic=0
use_complex=0
use_communicative=0
use_transporting=0
use_age=0
use_cross=1
lambda1=0.01
lambda2=1.
lambda3=0
lambda_contrast=0.5
backbone_add_on=0
score_sum_linear=1

use_img=1
resize_mode='even_padded'
img_backbone_name='C3D'

use_skeleton=1
sk_mode='pseudo_heatmap'
sk_backbone_name='poseC3D_pretrained'

use_context=1
ctx_mode='local'
ctx_backbone_name='C3D'

use_traj=1
traj_mode='ltrb'
traj_backbone_name='lstm'

use_ego=1
ego_backbone_name='lstm'

test_only=0
model_path=''
config_path=''

use_robust=1

usage() {
	 echo "Usage: ${0} ${1} ${2} wrong arg" 1>&2
	 echo ${1}
	 exit 1
	 }
while [[ $# -gt 0 ]];do
	key=${1}
	case ${key} in
		--model_name)
			model_name=${2}
			shift 2
			;;
		--pool)
			pool=${2}
			shift 2
			;;
		-e|--epochs)
			epochs=${2}
			shift 2
			;;
		-b|--batch_size)
			batch_size=${2}
			shift 2
			;;
		-t|--test_every)
			test_every=${2}
			shift 2
			;;
		--warm_strategy)
			warm_strategy=${2}
			shift 2
			;;
		--explain_every)
			explain_every=${2}
			shift 2
			;;
		--lr)
			lr=${2}
			shift 2
			;;
		--backbone_lr)
			backbone_lr=${2}
			shift 2
			;;
		--scheduler)
			scheduler=${2}
			shift 2
			;;
		--t_max)
			t_max=${2}
			shift 2
			;;
		--lr_step_size)
			lr_step_size=${2}
			shift 2
			;;
		--lr_step_gamma)
			lr_step_gamma=${2}
			shift 2
			;;
		--loss_func)
			loss_func=${2}
			shift 2
			;;
		--weight_decay)
			weight_decay=${2}
			shift 2
			;;
		--loss_weight)
			loss_weight=${2}
			shift 2
			;;
		--loss_weight_batch)
			loss_weight_batch=${2}
			shift 2
			;;
		--orth_type)
			orth_type=${2}
			shift 2
			;;
		--num_classes)
			num_classes=${2}
			shift 2
			;;
		--apply_tte)
			apply_tte=${2}
			shift 2
			;;
		--test_apply_tte)
			test_apply_tte=${2}
			shift 2
			;;
		--apply_sampler)
			apply_sampler=${2}
			shift 2
			;;
		--recog_act)
			recog_act=${2}
			shift 2
			;;
		--norm_pos)
			norm_pos=${2}
			shift 2
			;;
		--obs_interval)
			obs_interval=${2}
			shift 2
			;;
		--obs_len)
			obs_len=${2}
			shift 2
			;;
		--pred_len)
			pred_len=${2}
			shift 2
			;;
		--dataset_name)
			dataset_name=${2}
			shift 2
			;;
		--cross_dataset_name)
			cross_dataset_name=${2}
			shift 2
			;;
		--cross_dataset)
			cross_dataset=${2}
			shift 2
			;;
		--balance_train)
			balance_train=${2}
			shift 2
			;;
		--seq_type)
			seq_type=${2}
			shift 2
			;;
		--img_norm_mode)
			img_norm_mode=${2}
			shift 2
			;;
		--color_order)
			color_order=${2}
			shift 2
			;;
		--small_set)
			small_set=${2}
			shift 2
			;;
		--max_occ)
			max_occ=${2}
			shift 2
			;;
		--min_w)
			min_w=${2}
			shift 2
			;;
		--min_h)
			min_h=${2}
			shift 2
			;;
		--test_max_occ)
			test_max_occ=${2}
			shift 2
			;;
		--test_min_w)
			test_min_w=${2}
			shift 2
			;;
		--test_min_h)
			test_min_h=${2}
			shift 2
			;;
		--overlap)
			overlap=${2}
			shift 2
			;;
		--dataloader_workers)
			dataloader_workers=${2}
			shift 2
			;;
		--augment_mode)
			augment_mode=${2}
			shift 2
			;;
		--pop_occl_track)
			pop_occl_track=${2}
			shift 2
			;;
		--fusion_mode)
			fusion_mode=${2}
			shift 2
			;;
		--num_proto_per_modality)
			num_proto_per_modality=${2}
			shift 2
			;;
		--proto_dim)
			proto_dim=${2}
			shift 2
			;;
		--simi_func)
			simi_func=${2}
			shift 2
			;;
		--pred_traj)
			pred_traj=${2}
			shift 2
			;;
		--freeze_base)
			freeze_base=${2}
			shift 2
			;;
		--freeze_proto)
			freeze_proto=${2}
			shift 2
			;;
		--freeze_relev)
			freeze_relev=${2}
			shift 2
			;;
		--softmax_t)
			softmax_t=${2}
			shift 2
			;;
		--multi_label_cross)
			multi_label_cross=${2}
			shift 2
			;;
		--use_atomic)
			use_atomic=${2}
			shift 2
			;;
		--use_complex)
			use_complex=${2}
			shift 2
			;;
		--use_communicative)
			use_communicative=${2}
			shift 2
			;;
		--use_transporting)
			use_transporting=${2}
			shift 2
			;;
		--use_age)
			use_age=${2}
			shift 2
			;;
		--use_cross)
			use_cross=${2}
			shift 2
			;;
		--lambda1)
			lambda1=${2}
			shift 2
			;;
		--lambda2)
			lambda2=${2}
			shift 2
			;;
		--lambda3)
			lambda3=${2}
			shift 2
			;;
		--lambda_contrast)
			lambda_contrast=${2}
			shift 2
			;;
		--backbone_add_on)
			backbone_add_on=${2}
			shift 2
			;;
		--score_sum_linear)
			score_sum_linear=${2}
			shift 2
			;;
		--use_img)
			use_img=${2}
			shift 2
			;;
		--resize_mode)
			resize_mode=${2}
			shift 2
			;;
		--proto_activate)
			proto_activate=${2}
			shift 2
			;;
		--img_backbone_name)
			img_backbone_name=${2}
			shift 2
			;;
		--use_skeleton)
			use_skeleton=${2}
			shift 2
			;;
		--sk_mode)
			sk_mode=${2}
			shift 2
			;;
		--sk_backbone_name)
			sk_backbone_name=${2}
			shift 2
			;;
		--use_context)
			use_context=${2}
			shift 2
			;;
		--ctx_mode)
			ctx_mode=${2}
			shift 2
			;;
		--ctx_backbone_name)
			ctx_backbone_name=${2}
			shift 2
			;;
		--use_traj)
			use_traj=${2}
			shift 2
			;;
		--traj_mode)
			traj_mode=${2}
			shift 2
			;;
		--use_ego)
			use_ego=${2}
			shift 2
			;;
		--use_robust)
			use_robust=${2}
			shift 2
			;;
		*)
			usage
			shift
			;;
	esac
done


python _main_SLENN.py --epochs ${epochs} --batch_size ${batch_size} --test_every ${test_every} --explain_every ${explain_every} --lr ${lr} --dataset_name ${dataset_name} --cross_dataset_name ${cross_dataset_name} --cross_dataset ${cross_dataset} --small_set ${small_set} --max_occ ${max_occ} --min_w ${min_w} --min_h ${min_h} --test_max_occ ${test_max_occ} --test_min_w ${test_min_w} --test_min_h ${test_min_h} --overlap ${overlap} --fusion_mode ${fusion_mode} --simi_func ${simi_func} --use_img ${use_img} --use_skeleton ${use_skeleton} --use_context ${use_context} --use_traj ${use_traj} --use_ego ${use_ego} --loss_func ${loss_func} --pred_traj ${pred_traj} --model_name ${model_name} --balance_train ${balance_train} --seq_type ${seq_type} --weight_decay ${weight_decay} --loss_weight ${loss_weight} --num_classes ${num_classes} --apply_tte ${apply_tte} --apply_sampler ${apply_sampler} --recog_act ${recog_act} --dataloader_workers ${dataloader_workers} --test_apply_tte ${test_apply_tte} --ctx_mode ${ctx_mode} --ctx_backbone_name ${ctx_backbone_name} --freeze_base ${freeze_base} --freeze_proto ${freeze_proto} --freeze_relev ${freeze_relev} --backbone_lr ${backbone_lr} --norm_pos ${norm_pos} --softmax_t ${softmax_t} --orth_type ${orth_type} --proto_activate ${proto_activate} --num_proto_per_modality ${num_proto_per_modality} --obs_len ${obs_len} --obs_interval ${obs_interval} --sk_backbone_name ${sk_backbone_name} --sk_mode ${sk_mode} --resize_mode ${resize_mode} --multi_label_cross ${multi_label_cross} --use_atomic ${use_atomic} --img_backbone_name ${img_backbone_name} --use_complex ${use_complex} --use_communicative ${use_communicative} --use_transporting ${use_transporting} --use_age ${use_age} --loss_weight_batch ${loss_weight_batch} --pool ${pool} --use_cross ${use_cross} --img_norm_mode ${img_norm_mode} --color_order ${color_order} --lr_step_gamma ${lr_step_gamma} --lr_step_size ${lr_step_size} --traj_mode ${traj_mode} --augment_mode ${augment_mode} --use_robust ${use_robust} --lambda1 ${lambda1} --lambda2 ${lambda2} --lambda3 ${lambda3} --lambda_contrast ${lambda_contrast} --scheduler ${scheduler} --t_max ${t_max} --warm_strategy ${warm_strategy} --backbone_add_on ${backbone_add_on} --proto_dim ${proto_dim} --pred_len ${pred_len} --score_sum_linear ${score_sum_linear} --pop_occl_track ${pop_occl_track}
