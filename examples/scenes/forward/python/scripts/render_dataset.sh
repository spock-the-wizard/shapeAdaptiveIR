python3.8 ../render_dataset.py \
        --d_type "custom" \
        --seed 2 \
        --scene $1 \
        --ref_folder "exr_ref" \
        --mesh_lr 0.001 \
        --sigma_lr 0.00 \
        --albedo_lr 0.00 \
        --epsM_lr 0.00 \
        --rough_lr 0.00 \
        --eta_lr 0.00 \
        --n_reduce_step 1000 \
        --n_iters 1 \
        --n_dump 20 \
        --laplacian 10 \
        --sigma_laplacian 0 \
        --albedo_laplacian 0 \
        --rough_laplacian 0 \
        --no_init "no" \
        --spp 0 \
        --sppe 0 \
        --sppse 0 \
        --ref_spp 128 \
        --rough_texture 0 \
        --n_crops 1 \
        --sigma_texture  0 \
        --albedo_texture 0 \
        --epsM_texture 0 \
        --scene_file ../../scenes/forward/$1_out_baseline.xml \
        # --opaque \
        # --scene_file ../../scenes/forward/$1_out_opaque.xml \
        # --scene_file ../../scenes/forward/$1_out.xml \
        # --debug \
        # --isSweep \
        # --stats_folder 'test' \