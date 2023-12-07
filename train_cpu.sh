#!/bin/bash

#data
#pre_dataset_name='PIE JAAD'
#dataset_name='TITAN'
#train
p_epochs=50
batch_size=16
lr=0.01
backbone_lr=0.001
#loss
#model
concept_mode='mlp_fuse'
contrast_mode='bridge'
uncertainty='none'


usage() {
	echo "Usage: ${0} ${1} ${2} wrong arg" 1>&2
	echo ${1}
	exit 1
          }
while [[ $# -gt 0 ]];do
	key=${1}
	case ${key} in
		-p|--p_epochs)
			p_epochs=${2}
			shift 2
			;;
		-b|--batch_size)
			batch_size=${2}
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
		--concept_mode)
			concept_mode=${2}
			shift 2
			;;
		--contrast_mode)
			contrast_mode=${2}
			shift 2
			;;
		--uncertainty)
			uncertainty=${2}
			shift 2
			;;
		*)
			usage
			shift
			;;
	esac
done
python _main_CPU.py --p_epochs ${p_epochs} --batch_size ${batch_size} --concept_mode ${concept_mode} --contrast_mode ${contrast_mode} --uncertainty ${uncertainty} --lr ${lr} --backbone_lr ${backbone_lr}
