#! /bin/bash
exp_name="exp1/var48"
exp_name="exp3/var25"
# exp_name="test/gpu"
spp=32
root=/sss/InverseTranslucent/examples/python/scripts
for name in duck #kettle1 cylinder4 #head1 # botijo cone4 pyramid4 cube4 cylinder4
do

    # Exp 5. Render Gradient Image
    exp_name="test/was_debug"
    exp_name="test/was_v196"
    name="$1"
    echo $exp_name
    xml_file=$root/../../scenes/inverse/${name}_out.xml
    out_file=$root/../../scenes/inverse/${name}_out_tmp.xml
    albedo="0.8"
    sigma_t="100.0"
    sigma_t="52.0"

    python $root/replace_xml.py --sigma_t "$sigma_t" \
    --albedo "$albedo" \
    --in_xml "$xml_file" \
    --out_xml "$out_file" \
    # --is_baseline
    
    spp=16
    spp_inv=32
    n_crops=8
    sppse=256
    # sppse=0
    sppe=0 # set to nonzero when disabling interior term
    sensor_id="$2"
    bash $root/grad/render_grad.sh $exp_name $name $spp $out_file \
    $spp_inv $n_crops $sppse $sensor_id $sppe

    # Render baseline gradients
    python $root/replace_xml.py --sigma_t "$sigma_t" \
    --albedo "$albedo" \
    --in_xml "$xml_file" \
    --out_xml "$out_file" \
    --is_baseline
    exp_name="$exp_name"_deng
    sppse=0 # Disable boundary
    sppe=0 # Compute interior
    spp_inv=32
    ncrops=1
    bash $root/grad/render_grad.sh $exp_name $name $spp $out_file \
    $spp_inv $n_crops $sppse $sensor_id $sppe

    # # Exp 6. Check asymmetric backward
    # exp_name="exp3/var38"
    # # exp_name="test/time"
    # spp_inv=8
    # n_crops=2
    # echo $exp_name
    # xml_file=../../scenes/inverse/${name}_out.xml
    # out_file=../../scenes/inverse/${name}_out_tmp.xml
    # sigma_t="80.0, 80.0, 80.0"
    # albedo="0.8, 0.8, 0.8"
    # python ./replace_xml.py --sigma_t "$sigma_t" \
    # --albedo "$albedo" \
    # --in_xml "$xml_file" \
    # --out_xml "$out_file" \
    # # --is_baseline
    # bash ./inverse/learn_inverse.sh $exp_name $name $spp $out_file $spp_inv $n_crops
    
done