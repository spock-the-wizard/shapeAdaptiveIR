#! /bin/bash
root=/sss/InverseTranslucent/examples/python/scripts

exp_name="debug/largesteps"
exp_name="exp7/var39"
# exp_name="exp7/var31"

name=$1
echo $exp_name
xml_file=$root/../../scenes/inverse/${name}_out.xml
out_file=$root/../../scenes/inverse/${name}_out_tmp.xml
sigma_t="100.0, 70.0, 60.0" # init
# sigma_t='3.3824356, 4.320124, 5.453843' # init
# sigma_t="12.0" # init (duck)
# sigma_t="40.0" # init (pig1)

albedo="0.9, 0.9, 0.9" #sphere1
# albedo="0.8, 0.9, 0.7" # green init (duck) 
# albedo="0.6, 0.3, 0.9" # blue init (pig)
python $root/replace_xml.py --sigma_t "$sigma_t" \
--albedo "$albedo" \
--in_xml "$xml_file" \
--out_xml "$out_file" \
--is_baseline

spp=8
spp_inv=8
n_crops=2
sppse=0 # 0 for autograd -1 for naive FD 1 for FD_ours
sppe=4 # set to nonzero when disabling interior term
bash ./inverse/learn_inverse.sh $exp_name $name $spp $out_file $spp_inv $n_crops $sppse $sppe
