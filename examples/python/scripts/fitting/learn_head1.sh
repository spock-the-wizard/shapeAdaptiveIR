python3.8 ../learn_real_data_2.py \
        --stats_folder $1 \
        --d_type "custom" \
        --seed 2 \
        --scene "head1" \
        --n_dump 25 \
        --ref_folder "exr_ref" \
        --mesh_lr 0.00 \
        --sigma_lr 0.1 \
        --albedo_lr 0.05 \
        --rough_lr 0.05 \
        --eta_lr 0.005 \
        --n_reduce_step 1000 \
        --n_iters 1 \
        --laplacian 30 \
        --sigma_laplacian 0 \
        --albedo_laplacian 0 \
        --rough_laplacian 0 \
        --no_init "no" \
        --spp 4 \
        --sppe 4 \
        --sppse 4 \
        --ref_spp $2 \
        --albedo_texture 0 \
        --rough_texture 0 \
        --sigma_texture 0 \
        --scene_file $3 \
