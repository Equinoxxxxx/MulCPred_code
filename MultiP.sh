#!/bin/bash


batch_size=64
project_batch_size=64
epochs=100
warm_epochs=5
linear_epochs=10
push_start=10
test_every=5
backbone_lr=0.0001
add_on_lr=0.003
last_lr=0.0001
joint_last=0
warm_last=0
orth_type=0
orth_eff=0.01
clst_eff=0.8
sep_eff=-0.08
l1_eff=0.0001

dataset_name='PIE'
cross_dataset_name='JAAD'
small_set=0
resize_mode='even_padded'
bbox_type='default'
cross_dataset=1
balance_train=0
max_occ=2
test_max_occ=2
min_w=0
min_h=0
test_min_w=0
test_min_h=0
overlap=0.9
dataloader_workers=8

is_prototype_model=1
last_nonlinear=0
simi_func='log'
update_proto=1

use_traj=0

use_img=1
img_backbone_name='C3D'
img_p_per_cls=10
img_add_on_activation='sigmoid'

use_skeleton=1
skeleton_mode='coord'
sk_backbone_name='SK'
sk_p_per_cls=10
sk_add_on_activation='sigmoid'

use_context=1
ctx_backbone_name='C3D'
ctx_mode='mask_ped'
ctx_p_per_cls=20
ctx_add_on_activation='sigmoid'

use_single_img=0
single_img_backbone_name='segC2D'

test_only=0

usage() {
	 echo "Usage: ${0} ${1} ${2} wrong arg [--epochs] [--batch_size]" 1>&2
	 exit 1
	  }
while [[ $# -gt 0 ]];do
	key=${1}
	case ${key} in
		-e|--epochs)
			epochs=${2}
			shift 2
			;;
		-w|--warm_epochs)
			warm_epochs=${2}
			shift 2
			;;
		-l|--linear_epochs)
			linear_epochs=${2}
			shift 2
			;;
		-p|--project_batch_size)
			project_batch_size=${2}
			shift 2
			;;
		-b|--batch_size)
			batch_size=${2}
			shift 2
			;;
		--push_start)
			push_start=${2}
			shift 2
			;;
		--test_every)
			test_every=${2}
			shift 2
			;;
		--backbone_lr)
			backbone_lr=${2}
			shift 2
			;;
		--add_on_lr)
			add_on_lr=${2}
			shift 2
			;;
		--last_lr)
			last_lr=${2}
			shift 2
			;;
		--joint_last)
			joint_last=${2}
			shift 2
			;;
		--warm_last)
			warm_last=${2}
			shift 2
			;;
		--orth_type)
			orth_type=${2}
			shift 2
			;;
		--orth_eff)
			orth_eff=${2}
			shift 2
			;;
		--clst_eff)
			clst_eff=${2}
			shift 2
			;;
		--sep_eff)
			sep_eff=${2}
			shift 2
			;;
		--l1_eff)
			l1_eff=${2}
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
		--small_set)
			small_set=${2}
			shift 2
			;;
		--cross_dataset)
			cross_dataset=${2}
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
		--overlap)
			overlap=${2}
			shift 2
			;;
		--resize_mode)
			resize_mode=${2}
			shift 2
			;;
		--dataloader_workers)
			dataloader_workers=${2}
			shift 2
			;;
		--is_prototype_model)
			is_prototype_model=${2}
			shift 2
			;;
		--last_nonlinear)
			last_nonlinear=${2}
			shift 2
			;;
		--simi_func)
			simi_func=${2}
			shift 2
			;;
		--update_proto)
			update_proto=${2}
			shift 2
			;;
		--use_traj)
			use_traj=${2}
			shift 2
			;;
		--use_img)
			use_img=${2}
			shift 2
			;;
		--img_backbone_name)
			img_backbone_name=${2}
			shift 2
			;;
		--img_p_per_cls)
			img_p_per_cls=${2}
			shift 2
			;;
		--img_add_on_activation)
			img_add_on_activation=${2}
			shift 2
			;;
		--use_skeleton)
			use_skeleton=${2}
			shift 2
			;;
		--skeleton_mode)
			skeleton_mode=${2}
			shift 2
			;;
		--sk_backbone_name)
			sk_backbone_name=${2}
			shift 2
			;;
		--sk_p_per_cls)
			sk_p_per_cls=${2}
			shift 2
			;;
		--sk_add_on_activation)
			sk_add_on_activation=${2}
			shift 2
			;;
		--use_context)
			use_context=${2}
			shift 2
			;;
		--ctx_backbone_name)
			ctx_backbone_name=${2}
			shift 2
			;;
		--ctx_mode)
			ctx_mode=${2}
			shift 2
			;;
		--ctx_p_per_cls)
			ctx_p_per_cls=${2}
			shift 2
			;;
		--ctx_add_on_activation)
			ctx_add_on_activation=${2}
			shift 2
			;;
		--use_single_img)
			use_single_img=${2}
			shift 2
			;;
		--single_img_backbone_name)
			single_img_backbone_name=${2}
			shift 2
			;;
		--test_only)
			test_only=${2}
			shift 2
			;;
		*)
			usage
			shift
			;;
	esac
done

python _multi_main.py --batch_size ${batch_size} --project_batch_size ${project_batch_size} --epochs ${epochs} --warm_epochs ${warm_epochs} --test_every ${test_every} --orth_type ${orth_type} --orth_eff ${orth_eff} --clst_eff ${clst_eff} --sep_eff ${sep_eff} --l1_eff ${l1_eff} --resize_mode ${resize_mode} --bbox_type ${bbox_type} --push_start ${push_start} --max_occ ${max_occ} --min_w ${min_w} --min_h ${min_h} --cross_dataset ${cross_dataset} --is_prototype_model ${is_prototype_model} --balance_train ${balance_train} --use_img ${use_img} --img_backbone_name ${img_backbone_name} --use_skeleton ${use_skeleton} --sk_backbone_name ${sk_backbone_name} --use_context ${use_context} --ctx_backbone_name ${ctx_backbone_name} --ctx_mode ${ctx_mode} --small_set ${small_set} --dataset_name ${dataset_name} --cross_dataset_name ${cross_dataset_name} --overlap ${overlap} --linear_epochs ${linear_epochs} --backbone_lr ${backbone_lr} --add_on_lr ${add_on_lr} --last_lr ${last_lr} --ctx_p_per_cls ${ctx_p_per_cls} --dataloader_workers ${dataloader_workers} --test_max_occ ${test_max_occ} --test_min_w ${test_min_w} --test_min_h ${test_min_h} --last_nonlinear ${last_nonlinear} --joint_last ${joint_last} --warm_last ${warm_last} --img_p_per_cls ${img_p_per_cls} --sk_p_per_cls ${sk_p_per_cls} --use_single_img ${use_single_img} --single_img_backbone_name ${single_img_backbone_name} --img_add_on_activation ${img_add_on_activation} --sk_add_on_activation ${sk_add_on_activation} --ctx_add_on_activation ${ctx_add_on_activation} --use_traj ${use_traj} --simi_func ${simi_func} --test_only ${test_only} --skeleton_mode ${skeleton_mode} --update_proto ${update_proto}
