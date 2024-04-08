#! /bin/bash
exp_name="exp1/var48"
exp_name="exp3/var25"
# exp_name="test/gpu"
spp=16
root=/sss/InverseTranslucent/examples/python/scripts
# echo $SHELL
# export PYTHONPATH=/sss/InverseTranslucent/build/lib:/sss/InverseTranslucent/build:$PYTHONPATH
# export LD_LIBRARY_PATH=/sss/InverseTranslucent/build/lib:/sss/InverseTranslucent/build:$LD_LIBRARY_PATH
# cat /root/.bashrc
# export PYTHONPATH=/home/learned-subsurface-scattering/dist/python/3.8:/home/nanogui2/build/python:/home/nanogui2/build:/sss/InverseTranslucent/build/lib:/home/enoki/build:/home/enoki:/home/InverseTranslucent/build:/home/InverseTranslucent/build/lib:$PYTHONPATH
# export LD_LIBRARY_PATH=/home/nanogui2/build:/home/learned-subsurface-scattering/dist/python:/home/nanogui2/build/python:/sss/InverseTranslucent/build/lib:/sss/InverseTranslucent/build:/home/enoki/build:/home/enoki:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
# cd $root
# echo $PYTHONPATH
# echo $LD_LIBRARY_PATH
# for name in head1 #buddha1 kettle1 duck # cone4 sphere1 pyramid4 cylinder4 
# for name in duck kettle1 head1 # head1 # botijo cone4 pyramid4 cube4 cylinder4
for name in duck #kettle1 cylinder4 #head1 # botijo cone4 pyramid4 cube4 cylinder4
do
    # echo Running $name

    # # Exp 0. Rendering with complex geometry
    # echo $exp_name
    # bash ./template/learn_$name.sh $exp_name $spp None
    
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
    exp_name="exp3/var69"
    name="head1" #sphere1"
    # name="duck" #head1" #sphere1"
    # name="sphere1"
    echo $exp_name
    xml_file=$root/../../scenes/inverse/${name}_out.xml
    out_file=$root/../../scenes/inverse/${name}_out_tmp.xml

     # sigma_t="25.0, 25.0, 25.0" # duck
    # sigma_t="54.0, 72.0, 98.0" # sphere1
    # sigma_t="109.0, 109.0, 52.0" # head1
    sigma_t="80.0, 80.0, 80.0" # init
    sigma_t="50.0, 80.0, 60.0" # init
    # sigma_t="40.0, 40.0, 40.0" # init
    
    # albedo="0.9, 0.9, 0.9" #sphere1
    # albedo="0.88305, 0.183, 0.011" # duck
    # albedo="0.9, 0.9, 0.9" # head1
    albedo="0.8, 0.8, 0.8"

    python $root/replace_xml.py --sigma_t "$sigma_t" \
    --albedo "$albedo" \
    --in_xml "$xml_file" \
    --out_xml "$out_file" \
    # --is_baseline
    spp_inv=16
    n_crops=2
    # bash $root/inverse/learn_inverse_2.sh $exp_name $name $spp $out_file $spp_inv $n_crops
    bash $root/inverse/learn_inverse.sh $exp_name $name $spp $out_file $spp_inv $n_crops

    # # Exp 5. Render Gradient Image
    # exp_name="test/grad2"
    # name="sphere1"
    # # name="head1"
    # # name="duck"
    # # name="cylinder4"
    # echo $exp_name
    # xml_file=$root/../../scenes/inverse/${name}_out.xml
    # out_file=$root/../../scenes/inverse/${name}_out_tmp.xml
    # # sigma_t="80.0, 80.0, 80.0"
    # albedo="0.8, 0.8, 0.8"
    # sigma_t="109.0, 109.0, 52.0"
    # # albedo="0.9, 0.9, 0.9"

    # python $root/replace_xml.py --sigma_t "$sigma_t" \
    # --albedo "$albedo" \
    # --in_xml "$xml_file" \
    # --out_xml "$out_file" \
    # # --is_baseline
    # spp=256

    # bash $root/grad/render_grad.sh $exp_name $name $spp $out_file
    

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