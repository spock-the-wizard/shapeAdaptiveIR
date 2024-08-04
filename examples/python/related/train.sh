scene=$1
exp_name="exp10/var4"

python ./related/train.py \
    --scene $scene \
    --exp_name $exp_name \
    --n_dump 50 \
    --spp 8 \
    --sweep_num -1 \
    --onlySig
    # --sweep_num 0 \
    # --debug