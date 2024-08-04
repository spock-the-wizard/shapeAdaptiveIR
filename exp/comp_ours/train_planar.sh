#! /bin/bash
root=/sss/InverseTranslucent/examples/python/scripts

exp_name="fig9planar/var19"

name=$1
echo $exp_name
xml_file=$root/../../scenes/inverse/${name}_out.xml
out_file=$root/../../scenes/inverse/${name}_out_tmp.xml
sigma_t="50.0, 50.0, 50.0" # init
albedo="0.85" #sphere1
spp=128
spp_inv=32
n_crops=4
sppse=0 # 0 for autograd -1 for naive FD 1 for FD_ours
sppe=0 # set to nonzero when disabling interior term

python $root/replace_xml.py --sigma_t "$sigma_t" \
--albedo "$albedo" \
--in_xml "$xml_file" \
--out_xml "$out_file" \
--is_baseline


python3.8 ./inverse_render.py \
        --stats_folder $exp_name \
        --d_type "custom" \
        --seed 2 \
        --scene $name \
        --ref_folder "exr_ref" \
        --mesh_lr 0.00 \
        --sigma_lr 0.015 \
        --albedo_lr 0.005 \
        --epsM_lr 0.00 \
        --rough_lr 0.00 \
        --eta_lr 0.00 \
        --n_reduce_step 100 \
        --n_iters 300 \
        --n_dump 50 \
        --n_remesh 10000 \
        --laplacian 30 \
        --sigma_laplacian 0 \
        --albedo_laplacian 0 \
        --rough_laplacian 0 \
        --no_init "no" \
        --spp $spp_inv \
        --sppe $sppe \
        --sppse $sppse \
        --ref_spp $spp \
        --rough_texture 0 \
        --silhouette "no" \
        --n_crops $n_crops \
        --range_weight 100.0 \
        --sigma_texture  0 \
        --albedo_texture 0 \
        --epsM_texture 0 \
        --vaeMode 0 \
        --onlySSS \
        --scene_file $out_file \
        --isBaseline \
        --randomInit \
        --sweep_num 2 \
        # --debug \
        # --n_vaeModeChange 200 \
        # --render_gradient
        # --debug \
        # --opaque \
        # --isSweep \
