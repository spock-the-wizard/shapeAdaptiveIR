scene=$1
exp_name="render_results"

# python ./train.py \
python ./related/train.py \
    --scene $scene \
    --exp_name $exp_name \
    --ref_spp 1024 \
    --sweep_num -1 \
    --debug \
    --render 