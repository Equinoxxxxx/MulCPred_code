#!/bin/bash


batch_size=64
project_batch_size=64
epochs=100
warm_epochs=5
push_start=10
test_every=5
backbone_name='C3D'
save_proto_every_epoch=1
resize_mode='padded'
bbox_type='default'
max_occ=2
cross_dataset=0
is_prototype_model=1
prototype_dim=128
balance_train=1

while getopts b:p:e:w:t:n:s:r:x:u:m:c:i:d:l: option
do
	case "${option}" in
		b) batch_size=${OPTARG};;
		p) project_batch_size=${OPTARG};;
		e) epochs=${OPTARG};;
		w) warm_epochs=${OPTARG};;
		t) test_every=${OPTARG};;
		n) backbone_name=${OPTARG};;
		s) save_proto_every_epoch=${OPTARG};;
		r) resize_mode=${OPTARG};;
		x) bboxx_type=${OPTARG};;
		u) push_start=${OPTARG};;
		m) max_occ=${OPTARG};;
		c) cross_dataset=${OPTARG};;
		i) is_prototype_model=${OPTARG};;
		d) prototype_dim=${OPTARG};;
		l) balance_train=${OPTARG};;
	esac
done

python _main.py --batch_size ${batch_size} --project_batch_size ${project_batch_size} --epochs ${epochs} --warm_epochs ${warm_epochs} --test_every ${test_every} --backbone_name ${backbone_name} --save_proto_every_epoch ${save_proto_every_epoch} --resize_mode ${resize_mode} --bbox_type ${bbox_type} --push_start ${push_start} --max_occ ${max_occ} --cross_dataset ${cross_dataset} --is_prototype_model ${is_prototype_model} --prototype_dim ${prototype_dim} --balance_train ${balance_train}
