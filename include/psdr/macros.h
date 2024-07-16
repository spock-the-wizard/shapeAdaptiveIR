#pragma once

#ifdef _WIN32
#	define likely(x)       (x)
#	define unlikely(x)     (x)
#else
#	define likely(x)       __builtin_expect((x),1)
#	define unlikely(x)     __builtin_expect((x),0)
#endif

// #define PSDR_OPTIX_DEBUG
#define PSDR_MESH_ENABLE_1D_VERTEX_OFFSET
// #define PSDR_PRIMARY_EDGE_VIS_CHECK


#define LOAD_NETWORK \
    void loadNetwork(const std::string variablePath, const std::string featStatsFilePath) override {\
\
        const std::string shapeFeaturesName = "mlsPolyLS3";\
        int PolyOrder = 3;\
\
        int size = 32;\
        \
        NetworkHelpers::loadMat(variablePath + "/scatter_decoder_fcn_fcn_0_weights.bin",(float*) &m_scatter_decoder_fcn_fcn_0_weights);\
        NetworkHelpers::loadMat(variablePath + "/scatter_decoder_fcn_fcn_1_weights.bin",(float*) &m_scatter_decoder_fcn_fcn_1_weights);\
        NetworkHelpers::loadMat(variablePath + "/scatter_decoder_fcn_fcn_2_weights.bin",(float*) &m_scatter_decoder_fcn_fcn_2_weights);\
        NetworkHelpers::loadVector(variablePath + "/scatter_decoder_fcn_fcn_0_biases.bin",(float*) &m_scatter_decoder_fcn_fcn_0_biases);\
        NetworkHelpers::loadVector(variablePath + "/scatter_decoder_fcn_fcn_1_biases.bin",(float*) &m_scatter_decoder_fcn_fcn_1_biases);\
        NetworkHelpers::loadVector(variablePath + "/scatter_decoder_fcn_fcn_2_biases.bin",(float*) &m_scatter_decoder_fcn_fcn_2_biases);\
        NetworkHelpers::loadMat(variablePath + "/scatter_dense_2_kernel.bin", (float*) &m_scatter_dense_2_kernel,3);\
        NetworkHelpers::loadVector(variablePath + "/scatter_dense_2_bias.bin",(float*) &m_scatter_dense_2_bias);\
\
        NetworkHelpers::loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_weights.bin",(float*) &m_shared_preproc_mlp_2_shapemlp_fcn_0_weights);\
        NetworkHelpers::loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_weights.bin",(float*) &m_shared_preproc_mlp_2_shapemlp_fcn_1_weights);\
        NetworkHelpers::loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_weights.bin",(float*) &m_shared_preproc_mlp_2_shapemlp_fcn_2_weights);\
        NetworkHelpers::loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_biases.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_0_biases);\
        NetworkHelpers::loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_biases.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_1_biases);\
        NetworkHelpers::loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_biases.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_2_biases);\
\
        NetworkHelpers::loadMat(variablePath + "/absorption_dense_kernel.bin",(float*) &m_absorption_dense_kernel,1);\
        NetworkHelpers::loadMat(variablePath + "/absorption_mlp_fcn_0_weights.bin",(float*) &m_absorption_mlp_fcn_0_weights,32);\
        NetworkHelpers::loadVector(variablePath + "/absorption_dense_bias.bin", (float*) &m_absorption_dense_bias);\
        NetworkHelpers::loadVector(variablePath + "/absorption_mlp_fcn_0_biases.bin", (float*) &m_absorption_mlp_fcn_0_biases);\
\
        std::ifstream inStreamStats(featStatsFilePath);\
        if (inStreamStats) {\
            std::cout << "Loading statistics...\n";\
\
            inStreamStats >> stats;\
        } else {\
            std::cout << "STATS NOT FOUND " << featStatsFilePath << std::endl;\
        }\
        m_gMean            = stats["g_mean"][0];\
        m_gStdInv          = stats["g_stdinv"][0];\
        m_albedoMean       = stats["effAlbedo_mean"][0];\
        m_albedoStdInv     = stats["effAlbedo_stdinv"][0];\
        \
        std::string degStr = std::to_string(PolyOrder);\
        for (int i = 0; i < NetworkHelpers::nPolyCoeffs(PolyOrder); ++i) {\
            m_shapeFeatMean[i]   = stats[shapeFeaturesName + "_mean"][i];\
            m_shapeFeatStdInv[i] = stats[shapeFeaturesName + "_stdinv"][i];\
        }\
    }

#define PSDR_CLASS_DECL_BEGIN(_class_, _mode_, _parent_)    \
    class _class_ _mode_ : public _parent_ {


#define PSDR_CLASS_DECL_END(_class_)                        \
    public:                                                 \
        virtual std::string type_name() const override {    \
            return #_class_;                                \
        }                                                   \
    };


#define PSDR_IMPORT_BASE_HELPER(...) Base, ##__VA_ARGS__

#define PSDR_IMPORT_BASE(Name, ...)                         \
    using Base = Name;                                      \
    ENOKI_USING_MEMBERS(PSDR_IMPORT_BASE_HELPER(__VA_ARGS__))
