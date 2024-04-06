spp=64
spp_inv=32
n_crops=8
name=$1
xml_file=../../scenes/inverse/${name}_out.xml
out_file=../../scenes/inverse/${name}_out_tmp.xml

echo Running $name
# # Medium 0
exp_name="test/grad"
echo $exp_name
sigma_t="80.0, 80.0, 80.0"
albedo="0.8, 0.8, 0.8"
# spp=32
# spp_inv=8 # 8 for cylinder
python ./replace_xml.py --sigma_t "$sigma_t" \
--albedo "$albedo" \
--in_xml "$xml_file" \
--out_xml "$out_file" \
# --is_baseline
bash ./inverse/learn_inverse_2.sh $exp_name $name $spp $out_file $spp_inv $n_crops
# bash ./inverse/learn_inverse.sh $exp_name $name $spp $out_file $spp_inv $n_crops

# # Medium 1
# exp_name="exp3/var35"
# echo $exp_name
# sigma_t="30.0, 50.0 98.0"
# albedo="0.8, 0.7, 0.95"
# python ./replace_xml.py --sigma_t "$sigma_t" \
# --albedo "$albedo" \
# --in_xml "$xml_file" \
# --out_xml "$out_file" \
# # --is_baseline
# bash ./inverse/learn_inverse.sh $exp_name $name $spp $out_file $spp_inv $n_crops


# # Medium 2
# exp_name="exp3/var36"
# echo $exp_name
# sigma_t="65.0, 40.0, 90.0"
# albedo="0.5, 0.5, 0.5"
# python ./replace_xml.py --sigma_t "$sigma_t" \
# --albedo "$albedo" \
# --in_xml "$xml_file" \
# --out_xml "$out_file" \
# # --is_baseline
# bash ./inverse/learn_inverse.sh $exp_name $name $spp $out_file $spp_inv $n_crops


# # Medium 3
# exp_name="exp3/var37"
# echo $exp_name
# sigma_t="20.0, 40.0, 30.0"
# albedo="0.8, 0.5, 0.8"
# python ./replace_xml.py --sigma_t "$sigma_t" \
# --albedo "$albedo" \
# --in_xml "$xml_file" \
# --out_xml "$out_file" \
# # --is_baseline
# bash ./inverse/learn_inverse.sh $exp_name $name $spp $out_file $spp_inv $n_crops