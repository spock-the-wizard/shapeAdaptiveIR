scene=$1
exp_name="exp10/var10"

# python ./train.py \
python ./related/train.py \
    --scene $scene \
    --exp_name $exp_name \
    --n_dump 50 \
    --spp 32 \
    --sweep_num -1 \
    --lr 0.01 \
    # --onlySig
    # --sweep_num 0 \
    # --debug