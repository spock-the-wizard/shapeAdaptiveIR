#! /bin/bash

root=/sss/InverseTranslucent/examples/python/scripts

exp_name="final/cleanup"

name=$1
echo $exp_name
xml_file=$root/../../scenes/${name}_out.xml
out_file=$root/../../scenes/tmp/${name}_out_tmp.xml

sigma_t="100.0, 70.0, 60.0" # init
albedo="0.9, 0.9, 0.9" #sphere1

python ../replace_xml.py --sigma_t "$sigma_t" \
--albedo "$albedo" \
--in_xml "$xml_file" \
--out_xml "$out_file" \
# --is_baseline

spp=128
spp_inv=4
n_crops=1
sppse=0 # 0 for autograd -1 for naive FD 1 for FD_ours
sppe=4 # set to nonzero when disabling interior term
bash ./run_inverse.sh $exp_name $name $spp $out_file $spp_inv $n_crops $sppse $sppe
