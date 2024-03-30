mesh_file="../../smoothshape/test/cube_subdiv25.obj"
out_file="../../polycoeffs/test/cube_subdiv25"
    # /usr/bin/python3.8 extract_shape_features.py --mesh $mesh_file --out_file $out_file \
    # --albedo 0.9 --sigma_t $sigma_t --g 0.0 --eta 1.0

# for name in cone cylinder pyramid cube
# do
#     for sigma_t in 5 10 25 50 75 100
#         do
#             echo Extracting descriptor for $name with sigmaT $sigma_t
#     cd ../viz
#     /usr/bin/python3.8 extract_shape_features.py --mesh ../../smoothshape/vicini/${name}_subdiv.obj \
#     --out_file ../../polycoeffs/${name}_subdiv_sigmat${sigma_t} \
#     --albedo 0.98 --sigma_t $sigma_t --g 0.0 --eta 1.0
#     cd ../scripts
#     done
# done
for name in botijo
do
    for sigma_t in 5 10 25 50 75 100
        do
            echo Extracting descriptor for $name with sigmaT $sigma_t
    cd ../viz
    /usr/bin/python3.8 extract_shape_features.py --mesh ../../smoothshape/${name}.obj \
    --out_file ../../polycoeffs/${name}_sigmat${sigma_t} \
    --albedo 0.98 --sigma_t $sigma_t --g 0.0 --eta 1.0
    cd ../scripts
    done
done