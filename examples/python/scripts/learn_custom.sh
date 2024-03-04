python3.8 ../learn_real_data.py \
        --stats_folder "test_smallNet" \
        --d_type "custom" \
        --seed 2 \
        --scene "head" \
        --n_dump 50 \
        --ref_folder "exr_ref" \
        --mesh_lr 0.00 \
        --sigma_lr 0.1 \
        --albedo_lr 0.05 \
        --rough_lr 0.05 \
        --eta_lr 0.005 \
        --n_reduce_step 1000 \
        --n_iters 3000 \
        --laplacian 30 \
        --sigma_laplacian 0 \
        --albedo_laplacian 0 \
        --rough_laplacian 0 \
        --no_init "no" \
        --spp 16 \
        --sppe 32 \
        --sppse 32 \
        --ref_spp 50 \
        --albedo_texture 0 \
        --rough_texture 0 \
        --sigma_texture 0 \