#! /bin/bash
exp_name="exp1/var48"
exp_name="exp3/var25"
# exp_name="test/gpu"
spp=32
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

    # Exp 4. Inverse Rendering experiment
    exp_name="exp4/var196"
    # exp_name="exp5/var10"
    exp_name="exp6/var24"
    exp_name="exp7/var18"
    # exp_name="exp1/var91"
    # exp_name="exp7/var13"

    name=$1
    echo $exp_name
    xml_file=$root/../../scenes/inverse/${name}_out.xml
    out_file=$root/../../scenes/inverse/${name}_out_tmp.xml
    sigma_t="100.0, 70.0, 60.0" # init
    sigma_t='3.3824356, 4.320124, 5.453843' # init
    # sigma_t="12.0" # init (duck)

    # sigma_t="40.0" # init (pig1)

    albedo="0.9, 0.9, 0.9" #sphere1
    # albedo="0.8, 0.9, 0.7" # green init (duck) 
    # albedo="0.6, 0.3, 0.9" # blue init (pig)
    python $root/replace_xml.py --sigma_t "$sigma_t" \
    --albedo "$albedo" \
    --in_xml "$xml_file" \
    --out_xml "$out_file" \
    # --is_baseline
    
    spp=128
    spp_inv=8
    n_crops=1
    sppse=1 # 0 for autograd -1 for naive FD 1 for FD_ours
    sppe=0 # set to nonzero when disabling interior term
    bash ./inverse/learn_inverse.sh $exp_name $name $spp $out_file $spp_inv $n_crops $sppse $sppe

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