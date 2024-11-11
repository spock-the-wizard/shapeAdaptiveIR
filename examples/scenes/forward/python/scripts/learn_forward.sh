#! /bin/bash
exp_name="exp1/var48"
# exp_name="test/gpu"
spp=32
root=/sss/InverseTranslucent/examples/python/scripts
# echo $SHELL
for name in duck #kettle1 cylinder4 #head1 # botijo cone4 pyramid4 cube4 cylinder4
do
    # echo Running $name

    # # Exp 0. Rendering with complex geometry
    # echo $exp_name
    
    # Exp 1. Rendering with subdivided meshes for basic shapes
    # echo $exp_name
    # bash ./template/learn_$name.sh $exp_name $spp ../../scenes/subdiv/${name}_out.xml

    # Exp 2. Rendering with mismatching shape descriptor medium values
    # for med in 0 1 2 3 4 5
    # do
    #     exp_name=exp2/var$med
    #     echo Running $exp_name
    #     bash ./template/learn_$name.sh $exp_name $spp ../../scenes/subdiv_medium/${name}_med${med}.xml
    # done
    
    # Exp 3. Forward Rendering Baseline method
    # bash ./template/learn_$name.sh $exp_name $spp ../../scenes/scenes_baseline/${name}_out.xml

    # Exp 4. Inverse Rendering experiment
    exp_name="exp1/var88"

    name=$1
    echo $exp_name
    xml_file=$root/../../scenes/inverse/${name}_out.xml
    out_file=$root/../../scenes/inverse/${name}_out_tmp.xml
    sigma_t="80.0, 80.0, 40.0" # init
    # sigma_t="40.0" # init
    sigma_t="80.0" # init
    # sigma_t="100.0, 90.0, 60.0" # kettle
    albedo="0.9, 0.9, 0.9" #sphere1
    python $root/replace_xml.py --sigma_t "$sigma_t" \
    --albedo "$albedo" \
    --in_xml "$xml_file" \
    --out_xml "$out_file" \
    # --is_baseline
    
    spp=128
    spp_inv=1 # don't go lower than this...
    n_crops=1
    # sppse=128
    sppse=0 # zero for albedo/baseline, nonzero for FD
    sppe=0 # set to nonzero when disabling interior term
    bash ./render_forward.sh $exp_name \
        $name $spp $out_file $n_crops

    # # --is_baseline
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