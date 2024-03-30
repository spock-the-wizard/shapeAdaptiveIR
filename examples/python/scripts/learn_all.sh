
exp_name="exp1/var48"
exp_name="exp3/var18"
spp=64


for name in head1 #buddha1 kettle1 duck # cone4 sphere1 pyramid4 cylinder4 
# for name in botijo cone4 pyramid4 cube4 cylinder4
do
    echo Running $name

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
    
    # bash ./inverse/learn_inverse.sh $exp_name $name $spp $out_file

    # Exp 5. Render Gradient Image
    echo $exp_name
    xml_file=../../scenes/inverse/${name}_out.xml
    out_file=../../scenes/inverse/${name}_out_tmp.xml
    sigma_t="80.0, 80.0, 80.0"
    albedo="0.8, 0.8, 0.8"
    python ./replace_xml.py --sigma_t "$sigma_t" \
    --albedo "$albedo" \
    --in_xml "$xml_file" \
    --out_xml "$out_file" \
    --is_baseline
    
    bash ./grad/render_grad.sh $exp_name $name $spp $out_file
    
done