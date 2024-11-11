spp=64
name=$1
xml_file=../../scenes/inverse/${name}_out.xml
out_file=../../scenes/inverse/${name}_out_tmp.xml

echo Running $name

# Medium 0
exp_name="exp3/var42"
echo $exp_name
sigma_t="80.0, 80.0, 80.0"
albedo="0.8, 0.8, 0.8"
python ./replace_xml.py --sigma_t "$sigma_t" \
--albedo "$albedo" \
--in_xml "$xml_file" \
--out_xml "$out_file" \
--is_baseline
bash ./inverse/learn_inverse_baseline.sh $exp_name $name $spp $out_file

# # Medium 1
# exp_name="exp3/var40"
# echo $exp_name
# sigma_t="30.0, 50.0 98.0"
# albedo="0.8, 0.7, 0.95"
# python ./replace_xml.py --sigma_t "$sigma_t" \
# --albedo "$albedo" \
# --in_xml "$xml_file" \
# --out_xml "$out_file" \
# --is_baseline
# bash ./inverse/learn_inverse_baseline.sh $exp_name $name $spp $out_file

# # Medium 2
# exp_name="exp3/var41"
# echo $exp_name
# sigma_t="65.0, 40.0, 90.0"
# albedo="0.5, 0.5, 0.5"
# python ./replace_xml.py --sigma_t "$sigma_t" \
# --albedo "$albedo" \
# --in_xml "$xml_file" \
# --out_xml "$out_file" \
# --is_baseline
# bash ./inverse/learn_inverse_baseline.sh $exp_name $name $spp $out_file

# # Medium 3
# exp_name="exp3/var42"
# echo $exp_name
# sigma_t="20.0, 40.0, 30.0"
# albedo="0.8, 0.5, 0.8"
# python ./replace_xml.py --sigma_t "$sigma_t" \
# --albedo "$albedo" \
# --in_xml "$xml_file" \
# --out_xml "$out_file" \
# --is_baseline
# bash ./inverse/learn_inverse_baseline.sh $exp_name $name $spp $out_file
