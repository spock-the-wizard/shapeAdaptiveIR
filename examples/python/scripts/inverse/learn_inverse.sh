python3.8 ../learn_real_data_2.py \
        --stats_folder $1 \
        --d_type "custom" \
        --seed 2 \
        --scene $2 \
        --n_dump 25 \
        --ref_folder "exr_ref" \
        --mesh_lr 0.00 \
        --sigma_lr 0.01 \
        --albedo_lr 0.00 \
        --rough_lr 0.05 \
        --eta_lr 0.001 \
        --n_reduce_step 25 \
        --n_iters 300 \
        --laplacian 30 \
        --sigma_laplacian 0 \
        --albedo_laplacian 0 \
        --rough_laplacian 0 \
        --no_init "no" \
        --spp $5 \
        --sppe $8 \
        --sppse $7 \
        --ref_spp $3 \
        --albedo_texture 0 \
        --rough_texture 0 \
        --sigma_texture 0 \
        --silhouette "no" \
        --scene_file $4 \
        --n_crops $6 \
        --range_weight 100.0 \
        # --debug \
