#! /bin/bash
root=/sss/InverseTranslucent/examples/python/scripts

exp_name="test"

name=$1
echo $exp_name
xml_file=$root/../../scenes/inverse/${name}_out.xml
out_file=$root/../../scenes/inverse/${name}_out_tmp.xml

# Specify Medium Parameters Here
sigma_t="93.0, 88.0, 43.0" # init
albedo="0.89, 0.89, 0.90" #sphere1
sigma_t="50.0"
sigma_t="76.95"
albedo="0.90, 0.90, 0.90"
# sigma_t="52.0, 94.0, 92.0" # init
# albedo="0.97, 0.96, 0.96" #sphere1

# sigma_t="27.0, 56.0, 55.0" # init
# albedo="0.97, 0.97, 0.97" #sphere
# sigma_t="67.0, 50.0, 72.0" # init
# albedo="0.98, 0.98, 0.98" #sphere11

# Init (Head1)
albedo="0.81, 0.88, 0.79"
sigma_t="38, 80, 70"

# # Ours (Head1) Final
# sigma_t="117, 131, 64"
# albedo="0.93, 0.92, 0.93"

# Ours (Naive)
albedo="0.91, 0.92, 0.92"
sigma_t="32, 56, 31"

sigma_t="50.0, 50.0, 50.0"
albedo="0.70, 0.50, 0.99" 

sigma_t="44.0, 92.0, 102.0"
albedo="0.97, 0.98, 0.98"

sigma_t="29, 56, 54"
albedo="0.97, 0.97, 0.97"

# init 
sigma_t="50.0, 50.0, 50.0"
albedo="0.50, 0.50, 0.50" 

# init 
sigma_t="21.95, 47.21, 51.20"
albedo="0.94, 0.93, 0.93"

# # Ours (32spp)
# sigma_t="25, 52, 54"
# albedo="0.94, 0.94, 0.94"


# albedo="0.93, 0.84, 0.77"
# sigma_t="73, 42, 69"

# # cool-dream
# albedo="0.78, 0.85, 0.80"
# sigma_t="46, 75, 61"


# #revived-sun
# albedo="0.79, 0.85, 0.85"
# sigma_t="91, 57, 30"

spp=1024 #1024
spp_inv=1
n_crops=1
sppse=0 # 0 for autograd -1 for naive FD 1 for FD_ours
sppe=0 # set to nonzero when disabling interior term

python $root/replace_xml.py --sigma_t "$sigma_t" \
--albedo "$albedo" \
--in_xml "$xml_file" \
--out_xml "$out_file" \

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
        --n_reduce_step 200 \
        --n_iters 1 \
        --n_dump 10 \
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
        --render_results \
        --maxDistScale 2.0 \
        --debug \
        # --randomInit \
        # --debug \
        # --n_vaeModeChange 200 \
        # --isBaseline \
        # --render_gradient
        # --debug \
        # --opaque \
        # --isSweep \
