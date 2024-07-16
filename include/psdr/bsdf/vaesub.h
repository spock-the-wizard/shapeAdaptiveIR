#pragma once

#include <psdr/core/bitmap.h>
#include <enoki/matrix.h>
#include <enoki/array.h>
#include <enoki/dynamic.h>

#include <type_traits>
// Added Eigen and Json 
// #include <Eigen/Core>
// #include <Eigen/Dense>
#include <psdr/bsdf/json.hpp>
// #include <sdf/sdf>
// #include <sdf/sdf>
#include "sdf/sdf.hpp"
// #include "sdf.hpp"
#include "../src/bsdf/scattereigen.h"

#include "bsdf.h"
#define PreLayerWidth 64
#define LayerWidth 64
#define NLatent 4

using json = nlohmann::json;

namespace psdr
{
// template <int C, int R,bool ad>
// Array<Float<ad>,R> matmul(Array<Array<float,R>,C> mat, Array<Float<ad>,C> vec) {
//     using EVector = Array<Float<ad>,R>;
//     // Array<Float<ad>,R> sum = mat.coeff(0) * EVector::full_(vec[0],1); //[64] * [1,N]
//     auto sum = mat.coeff(0) * EVector::full_(vec[0],1); //[64] * [1,N]
    
//     for (int i=1;i<slices(mat);i++){
//         sum = fmadd(mat.coeff(i),EVector::full_(vec.coeff(i),1),sum);
//     }
//     return sum;
// }
// typedef DynamicArray<DynamicArray<Packet<float,1>>> VectorXf;
using InnerDynamicArray = DynamicArray<float>;
using OuterDynamicArray = DynamicArray<InnerDynamicArray>;

// PSDR_CLASS_DECL_BEGIN(VaeSub, final, BSDF)
PSDR_CLASS_DECL_BEGIN(VaeSub,, BSDF)
public:
    // Lightweight model
    const std::string variablePath = "../../../variables";
    const std::string featStatsFilePath = "../../../data_stats.json";
    const static int hidden_dim = 32;
    const static int hidden_dim_abs = 16;
    
    // // Full model 
    // const std::string variablePath = "../../../forward_model_weights/variables_org";
    // const std::string featStatsFilePath = "../../../forward_model_weights/data_stats_0118.json";
    // const static int hidden_dim = 64;
    // const static int hidden_dim_abs = 32; 
    VaeSub() : BSDF()
        {   
            m_anisotropic = false;
        loadNetwork(variablePath,featStatsFilePath);
         }
    VaeSub(const Bitmap1fD&alpha, const Bitmap1fD &eta, const Bitmap3fD &reflectance, const Bitmap3fD &sigma_t, const Bitmap3fD &albedo)
        : BSDF(alpha,eta,reflectance,sigma_t,albedo)
    { 
        m_anisotropic = false; 
        loadNetwork(variablePath,featStatsFilePath);
        const std::string variablePath = "None";
    }

    void setAlbedoTexture(std::string filename){
        m_albedo.load_openexr(filename.c_str());
    }

    void setSigmaTexture(std::string filename){
        m_sigma_t.load_openexr(filename.c_str());
    }

    void setAlphaTexture(std::string filename){
        m_alpha_u.load_openexr(filename.c_str());
        m_alpha_v.load_openexr(filename.c_str());
    }

    void setEpsMTexture(std::string filename){
        m_epsM.load_openexr(filename.c_str());
    }

    void setAlbedo(ScalarVector3f &albedo){
        m_albedo.fill(albedo);
    }

    void setEpsM(ScalarVector3f &epsM){
        m_epsM.fill(epsM);
    }

    void setSigmaT(ScalarVector3f &sigma_t){
        m_sigma_t.fill(sigma_t);
    }
    void loadNetwork(const std::string variablePath= "../../../variables",
    const std::string featStatsFilePath = "../../../data_stats.json"
    ) {

        const std::string shapeFeaturesName = "mlsPolyLS3";
        int PolyOrder = 3;

        int size = 32;
        
        NetworkHelpers::loadMat(variablePath + "/scatter_decoder_fcn_fcn_0_weights.bin",(float*) &m_scatter_decoder_fcn_fcn_0_weights);
        NetworkHelpers::loadMat(variablePath + "/scatter_decoder_fcn_fcn_1_weights.bin",(float*) &m_scatter_decoder_fcn_fcn_1_weights);
        NetworkHelpers::loadMat(variablePath + "/scatter_decoder_fcn_fcn_2_weights.bin",(float*) &m_scatter_decoder_fcn_fcn_2_weights);
        NetworkHelpers::loadVector(variablePath + "/scatter_decoder_fcn_fcn_0_biases.bin",(float*) &m_scatter_decoder_fcn_fcn_0_biases);
        NetworkHelpers::loadVector(variablePath + "/scatter_decoder_fcn_fcn_1_biases.bin",(float*) &m_scatter_decoder_fcn_fcn_1_biases);
        NetworkHelpers::loadVector(variablePath + "/scatter_decoder_fcn_fcn_2_biases.bin",(float*) &m_scatter_decoder_fcn_fcn_2_biases);
        NetworkHelpers::loadMat(variablePath + "/scatter_dense_2_kernel.bin", (float*) &m_scatter_dense_2_kernel,3);
        NetworkHelpers::loadVector(variablePath + "/scatter_dense_2_bias.bin",(float*) &m_scatter_dense_2_bias);

        NetworkHelpers::loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_weights.bin",(float*) &m_shared_preproc_mlp_2_shapemlp_fcn_0_weights);
        NetworkHelpers::loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_weights.bin",(float*) &m_shared_preproc_mlp_2_shapemlp_fcn_1_weights);
        NetworkHelpers::loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_weights.bin",(float*) &m_shared_preproc_mlp_2_shapemlp_fcn_2_weights);
        NetworkHelpers::loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_biases.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_0_biases);
        NetworkHelpers::loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_biases.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_1_biases);
        NetworkHelpers::loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_biases.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_2_biases);

        NetworkHelpers::loadMat(variablePath + "/absorption_dense_kernel.bin",(float*) &m_absorption_dense_kernel,1);
        NetworkHelpers::loadMat(variablePath + "/absorption_mlp_fcn_0_weights.bin",(float*) &m_absorption_mlp_fcn_0_weights,32);
        NetworkHelpers::loadVector(variablePath + "/absorption_dense_bias.bin", (float*) &m_absorption_dense_bias);
        NetworkHelpers::loadVector(variablePath + "/absorption_mlp_fcn_0_biases.bin", (float*) &m_absorption_mlp_fcn_0_biases);

        std::ifstream inStreamStats(featStatsFilePath);
        if (inStreamStats) {
            std::cout << "Loading statistics...n";

            inStreamStats >> stats;
        } else {
            std::cout << "STATS NOT FOUND " << featStatsFilePath << std::endl;
        }
        m_gMean            = stats["g_mean"][0];
        m_gStdInv          = stats["g_stdinv"][0];
        m_albedoMean       = stats["effAlbedo_mean"][0];
        m_albedoStdInv     = stats["effAlbedo_stdinv"][0];
        
        std::string degStr = std::to_string(PolyOrder);
        for (int i = 0; i < NetworkHelpers::nPolyCoeffs(PolyOrder); ++i) {
            m_shapeFeatMean[i]   = stats[shapeFeaturesName + "_mean"][i];
            m_shapeFeatStdInv[i] = stats[shapeFeaturesName + "_stdinv"][i];
        }
    }

    FloatC pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const override;
    FloatD pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const override;

    FloatD getKernelEps(const IntersectionD& its,FloatD idx) const override;
    FloatC getKernelEps(const IntersectionC& its,FloatC idx) const override;
    
    FloatC getEta(const IntersectionC& its) const {
        return m_eta.eval<false>(its.uv);
    };
    FloatC getFresnel(const IntersectionC& its) const override {
        FloatC cos_theta_i = FrameC::cos_theta(its.wi);
        masked(cos_theta_i,isnan(cos_theta_i)) = 1.0f;
        SpectrumC Fresnel = __FersnelDi<false>(1.0f, m_eta.eval<false>(its.uv), cos_theta_i);
        return Fresnel.x();
    };

    // FloatD absorption(const IntersectionD& its,Vector8fD& sample) const override;
    // FloatC absorption(const IntersectionC& its,Vector8fC& sample) const override;

    FloatC pdfpoint(const IntersectionC &its, const BSDFSampleC &bs, MaskC active) const override;
    FloatD pdfpoint(const IntersectionD &its, const BSDFSampleD &bs, MaskD active) const override;

    SpectrumC eval(const IntersectionC &its, const Vector3fC &bs, MaskC active = true) const override;
    SpectrumD eval(const IntersectionD &its, const Vector3fD &bs, MaskD active = true) const override;

    SpectrumC eval(const IntersectionC &its, const BSDFSampleC &bs, MaskC active = true) const override;
    SpectrumD eval(const IntersectionD &its, const BSDFSampleD &bs, MaskD active = true) const override;

    BSDFSampleC sample(const Scene *scene, const IntersectionC &its, const Vector8fC &sample, MaskC active = true) const override;
    BSDFSampleD sample(const Scene *scene, const IntersectionD &its, const Vector8fD &sample, MaskD active = true) const override;

    std::pair<BSDFSampleD,FloatD> sample_v2(const Scene *scene, const IntersectionD &its, const Vector8fD &sample, MaskD active = true) const override;

    FloatC pdf(const IntersectionC &its, const BSDFSampleC &wo, MaskC active = true) const override;
    FloatD pdf(const IntersectionD &its, const BSDFSampleD &wo, MaskD active = true) const override;
    template <bool ad>
    using RetTypeSampleSp = std::tuple<Intersection<ad>,Float<ad>,Vector3f<ad>,Vector3f<ad>,Float<ad>,Intersection<ad>,Intersection<ad>>;

   template <bool ad>
    RetTypeSampleSp<ad> __sample_sp(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Float<ad> &pdf, Mask<ad>active) const;
    template <bool ad>
    BSDFSample<ad> __sample_sub(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Mask<ad> active) const;
    template <bool ad>
    Float<ad> __pdf_sub(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const;
    template <bool ad>
    Spectrum<ad> __eval_sub(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const;
    void setMonochrome(bool isMonochrome){
        m_monochrome = isMonochrome;
    }

    bool anisotropic() const override { return m_anisotropic; }
    bool hasbssdf() const override { return true; }
    template <bool ad>
    Float<ad> getKernelEps(const Intersection<ad>& its,Float<ad> idx) const;
    template <bool ad>
    Float<ad> getFitScaleFactor(const Intersection<ad> &its, Float<ad> sigma_t_ch, Float<ad> albedo_ch, Float<ad> g_ch) const;
    
    // FloatD absorption(const IntersectionD &its, Vector8fD &sample) const override {
    FloatD absorption(const IntersectionD &its, Vector8fD sample) const override {
        FloatD absorption;;
        FloatD rnd = sample[5];
        float is_plane = false;
        float is_light_space = true;
        std::cout << "Inisde absorption " << std::endl;
        std::cout << "its.wi " << its.wi << std::endl;
        
        auto x = _preprocessFeatures<true>(its,rnd,is_plane,its.wi,is_light_space);
        Array<FloatD,4> latent(sample[2],sample[3],sample[6],sample[7]);
        // std::tie(std::ignore,absorption)= _run<true>(x,latent);
        return absorption;
    } 
    std::string to_string() const override { return std::string("VaeSub[id=") + m_id + "]"; }


                                          //
    using Vector1f = Array<float,1>;
    using Vector20f = Array<float,20>;
                                          
    using Matrix64f = Matrix<float,hidden_dim>;
    using Matrix68f = Matrix<float,hidden_dim+4>;
    using Vector64f = Array<float,hidden_dim>;
    using Vector32f = Array<float,hidden_dim_abs>;
    using Matrix64_23 = Array<Array<float,hidden_dim>,23>;
    using Matrix32_64 = Array<Array<float,hidden_dim_abs>,hidden_dim>;
    using Matrix64_64 = Array<Array<float,hidden_dim>,hidden_dim>;
    using Matrix1_32= Array<Array<float,1>,hidden_dim_abs>;
    using Matrix64_68 = Array<Array<float,hidden_dim>,hidden_dim + 4>;
    using Matrix3_64 = Array<Array<float,3>,hidden_dim>;

    bool m_isFull;
    
    Matrix32_64 m_absorption_mlp_fcn_0_weights = zero<Matrix32_64>(); // Eigen::Matrix<float,32,64>
    Vector32f m_absorption_mlp_fcn_0_biases = zero<Vector32f>();
    Matrix1_32 m_absorption_dense_kernel = zero<Matrix1_32>();
    Vector1f m_absorption_dense_bias = zero<Vector1f>();

    Matrix64_68 m_scatter_decoder_fcn_fcn_0_weights = zero<Matrix64_68>(); // [64,68]
    Matrix64_64 m_scatter_decoder_fcn_fcn_1_weights = zero<Matrix64_64>();
    Matrix64_64 m_scatter_decoder_fcn_fcn_2_weights = zero<Matrix64_64>();
    Vector64f m_scatter_decoder_fcn_fcn_0_biases = zero<Vector64f>();
    Vector64f m_scatter_decoder_fcn_fcn_1_biases = zero<Vector64f>();
    Vector64f m_scatter_decoder_fcn_fcn_2_biases = zero<Vector64f>();
    Matrix3_64 m_scatter_dense_2_kernel = zero<Matrix3_64>();
    Array<float,3> m_scatter_dense_2_bias = zero<Array<float,3>>();

    Vector64f shared_preproc_mlp_2_shapemlp_fcn_0_biases = zero<Vector64f>();
    Vector64f shared_preproc_mlp_2_shapemlp_fcn_1_biases = zero<Vector64f>();
    Vector64f shared_preproc_mlp_2_shapemlp_fcn_2_biases = zero<Vector64f>();

    Matrix64_23 m_shared_preproc_mlp_2_shapemlp_fcn_0_weights = zero<Matrix64_23>(); // [64,23]
    Matrix64_64 m_shared_preproc_mlp_2_shapemlp_fcn_1_weights = zero<Matrix64_64>();  // [64,64]
    Matrix64_64 m_shared_preproc_mlp_2_shapemlp_fcn_2_weights = zero<Matrix64_64>(); // [64,64]

    Vector20f m_shapeFeatMean,m_shapeFeatStdInv;
    
    json stats; 
    
    float m_albedoMean, m_albedoStdInv, m_gMean, m_gStdInv;

    bool m_anisotropic;
    bool m_monochrome = false;

protected:
    template <bool ad>
    BSDFSample<ad> __sample(const Scene *scene, const Intersection<ad>&, const Vector8f<ad>&, Mask<ad>) const;
    
    void set_grad(int id, const Vector3fC grad) override {
        if(id==0){
            set_gradient(m_sigma_t.m_data,grad);
        }
        //else : other parameters

    }
   
    template <bool ad>
    Float<ad> __pdf(const Intersection<ad> &, const BSDFSample<ad> &, Mask<ad>) const;
    // template <bool ad>
    // Float<ad> __pdf_better(const Intersection<ad> &, const BSDFSample<ad> &, Mask<ad>) const;
    template <bool ad>
    Float<ad> __pdfspecular(const Intersection<ad> &, const BSDFSample<ad> &, Mask<ad>) const;

    template <bool ad>
    Spectrum<ad> __eval(const Intersection<ad>&, const BSDFSample<ad>&, Mask<ad>) const;

    // template<bool ad>
    // Intersection<ad> projectPointToSurface(const Intersection<ad> &its, Vector3f<ad> outPos, Float<ad> sigma_t_ch, Float<ad> rnd, Vector20f shapeFeatures) const;
    template<bool ad>
    Intersection<ad> projectPointToSurface(const Scene* scene, const Intersection<ad> &its, Vector3f<ad> outPos, Float<ad> fitScaleFactor, Vectorf<20,ad> shapeFeatures, Vector3f<ad> projDir,Mask<ad> active,bool isPlane = false) const;
    
    // Joon added
    // template <bool ad,size_t size>
    // Spectrum<ad> _run(Array<Float<ad>,size> x, Array<Float<ad>,4> latent) const;

    template <bool ad,int size=64>
    std::pair<Vector3f<ad>,Float<ad>> _run(Array<Float<ad>,23> x, Array<Float<ad>,4> latent, bool isFullModel = false) const;

    // std::pair<Vector3fC,FloatC> _run(Array<FloatC,23> x, Array<FloatC,4> latent, bool isFullModel = false) const;
    // std::pair<Vector3fD,FloatD> _run(Array<FloatD,23> x, Array<FloatD,4> latent, bool isFullModel = false) const;

    template <bool ad,size_t size>
    Array<Float<ad>,size> _preprocessFeatures(const Intersection<ad>&its, int idx, bool isPlane) const;

    template <bool ad>
    Array<Float<ad>,23> _preprocessFeatures(const Intersection<ad>&its, Float<ad> idx, bool isPlane, Array<Float<ad>,3> wi, bool light_space = false) const;

    // Xi Deng added
    //
    template <bool ad>
    Spectrum<ad> __Sp(const Intersection<ad>&its, const Intersection<ad>&bsp) const;

    template <bool ad>
    Spectrum<ad> __Sp_better(const Intersection<ad>&its, const Intersection<ad>&bs) const;

    template <bool ad>
    Spectrum<ad> __Sw(const Intersection<ad>&, const Vector3f<ad>&, Mask<ad>) const;

    template <bool ad>
    Float<ad> __pdf_wo(const Intersection<ad> &, const Vector3f<ad> &, Mask<ad>) const;

    template <bool ad>
    Float<ad> __pdf_sr(const Float<ad> &, const Float<ad> &, Mask<ad>) const;

    template <bool ad>
    Float<ad> __sample_sr(const Float<ad> &, const Float<ad> &) const;

    // template <bool ad>
    // Intersection<ad> __sample_sp(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Float<ad> &pdf, Mask<ad>active) const;
     // template <bool ad>
    // Intersection<ad> __get_intersection(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Vector3f<ad> vx, Vector3f<ad> vy, Vector3f<ad> vz, Float<ad> l, Float<ad> r, Float<ad> phi, Mask<ad> active) const;
    // template <bool ad>
    // Intersection<ad> __selectRandomSample(const Int<ad> randomsample, std::vector<Intersection<ad>> inters) const;
    
    template <bool ad>
    Float<ad> __pdf_bsdf(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const;

    template <bool ad>
    BSDFSample<ad> __sample_bsdf(const Intersection<ad> &its, const Vector8f<ad> &sample, Mask<ad> active) const;
    template <bool ad>
    Intersection<ad> __sample_better_sp(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Float<ad> &pdf, Mask<ad> active) const;
    template <bool ad>
    Spectrum<ad> __eval_bsdf(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const;
    // help functions
    template <bool ad> Spectrum<ad> __FresnelMoment1(Spectrum<ad> eta) const;
    template <bool ad> Spectrum<ad> __FresnelMoment2(Spectrum<ad> eta) const;
    template <bool ad> Spectrum<ad> __FersnelDi(Spectrum<ad> eta_i, Spectrum<ad> eta_o, Float<ad> cos_theta_i) const;

PSDR_CLASS_DECL_END(VaeSub)
    
    /*
class VaeSubFullModel : public VaeSub {
    public:
    const std::string variablePath = "../../../forward_model_weights/variables_org";
    const std::string featStatsFilePath = "../../../forward_model_weights/data_stats_0118.json";
    VaeSubFullModel() : VaeSub()
        {   
            m_anisotropic = false;
        loadNetwork(variablePath,featStatsFilePath);
         }
    // void loadNetwork();
    VaeSubFullModel(const Bitmap1fD&alpha, const Bitmap1fD &eta, const Bitmap3fD &reflectance, const Bitmap3fD &sigma_t, const Bitmap3fD &albedo)
        : VaeSub(alpha,eta,reflectance,sigma_t,albedo)
    { 
        m_anisotropic = false; 
        loadNetwork(variablePath,featStatsFilePath);
        const std::string variablePath = "None";
    }
    bool m_isFull = true;
    using Matrix64f = Matrix<float,64>;
    using Matrix68f = Matrix<float,68>;
    using Vector64f = Array<float,64>;
    using Vector1f = Array<float,1>;
    using Vector20f = Array<float,20>;
    using Vector32f = Array<float,32>;
    
    using Matrix64_23 = Array<Array<float,64>,23>;
    using Matrix32_64 = Array<Array<float,32>,64>;
    using Matrix64_64 = Array<Array<float,64>,64>;
    using Matrix1_32= Array<Array<float,1>,32>;
    using Matrix64_68 = Array<Array<float,64>,64 + 4>;
    using Matrix3_64 = Array<Array<float,3>,64>;
    
    Matrix32_64 m_absorption_mlp_fcn_0_weights = zero<Matrix32_64>(); // Eigen::Matrix<float,32,64>
    Vector32f m_absorption_mlp_fcn_0_biases = zero<Vector32f>();
    Matrix1_32 m_absorption_dense_kernel = zero<Matrix1_32>();
    Vector1f m_absorption_dense_bias = zero<Vector1f>();

    Matrix64_68 m_scatter_decoder_fcn_fcn_0_weights = zero<Matrix64_68>(); // [64,68]
    Matrix64_64 m_scatter_decoder_fcn_fcn_1_weights = zero<Matrix64_64>();
    Matrix64_64 m_scatter_decoder_fcn_fcn_2_weights = zero<Matrix64_64>();
    Vector64f m_scatter_decoder_fcn_fcn_0_biases = zero<Vector64f>();
    Vector64f m_scatter_decoder_fcn_fcn_1_biases = zero<Vector64f>();
    Vector64f m_scatter_decoder_fcn_fcn_2_biases = zero<Vector64f>();
    Matrix3_64 m_scatter_dense_2_kernel = zero<Matrix3_64>();
    Array<float,3> m_scatter_dense_2_bias = zero<Array<float,3>>();

    Vector64f shared_preproc_mlp_2_shapemlp_fcn_0_biases = zero<Vector64f>();
    Vector64f shared_preproc_mlp_2_shapemlp_fcn_1_biases = zero<Vector64f>();
    Vector64f shared_preproc_mlp_2_shapemlp_fcn_2_biases = zero<Vector64f>();

    Matrix64_23 m_shared_preproc_mlp_2_shapemlp_fcn_0_weights = zero<Matrix64_23>(); // [64,23]
    Matrix64_64 m_shared_preproc_mlp_2_shapemlp_fcn_1_weights = zero<Matrix64_64>();  // [64,64]
    Matrix64_64 m_shared_preproc_mlp_2_shapemlp_fcn_2_weights = zero<Matrix64_64>(); // [64,64]

    LOAD_NETWORK;
    
// template <bool ad,int size>
// std::pair<Vector3f<ad>,Float<ad>> _run(Array<Float<ad>,23> x,Array<Float<ad>,4> latent, bool isFullModel) const override {
//     Vector3f<ad> outPos2;
//     Float<ad> absorption;
//         const int hidden_dim = 64;
//         const int hidden_dim_abs = 32;
        
//         auto features = max(matmul<23,hidden_dim,ad>(m_shared_preproc_mlp_2_shapemlp_fcn_0_weights,x) + shared_preproc_mlp_2_shapemlp_fcn_0_biases,0.0f);
//         features = max(matmul<hidden_dim,hidden_dim,ad>(m_shared_preproc_mlp_2_shapemlp_fcn_1_weights,features) + shared_preproc_mlp_2_shapemlp_fcn_1_biases,0.0f);
//         features = max(matmul<hidden_dim,hidden_dim,ad>(m_shared_preproc_mlp_2_shapemlp_fcn_2_weights,features) + shared_preproc_mlp_2_shapemlp_fcn_2_biases,0.0f);

//         auto abs1 = max(matmul<hidden_dim,hidden_dim_abs,ad>(m_absorption_mlp_fcn_0_weights, features) + m_absorption_mlp_fcn_0_biases,0.0f); // Vector hidden_dim_abs
//         auto abs2 = matmul<hidden_dim_abs,1,ad>(m_absorption_dense_kernel, abs1) + m_absorption_dense_bias;
//         absorption = NetworkHelpers::sigmoid<ad>(abs2.x());

//         auto featLatent = concat(latent,features);
//         auto y2 = max(matmul<hidden_dim + 4,hidden_dim,ad>(m_scatter_decoder_fcn_fcn_0_weights,featLatent)+m_scatter_decoder_fcn_fcn_0_biases,0.0f);
//         y2 = max(matmul<hidden_dim,hidden_dim,ad>(m_scatter_decoder_fcn_fcn_1_weights,y2)+m_scatter_decoder_fcn_fcn_1_biases,0.0f);
//         y2 = max(matmul<hidden_dim,hidden_dim,ad>(m_scatter_decoder_fcn_fcn_2_weights,y2)+m_scatter_decoder_fcn_fcn_2_biases,0.0f);
//         outPos2 = matmul<hidden_dim,3,ad>(m_scatter_dense_2_kernel,y2) + m_scatter_dense_2_bias;
//         return std::pair(outPos2,absorption);
// }

};

class VaeSubLightModel : public VaeSub {
    public:
    const std::string variablePath = "../../../variables";\
    const std::string featStatsFilePath = "../../../data_stats.json";\
    VaeSubLightModel() : VaeSub()
        {   
            m_anisotropic = false;
        loadNetwork(variablePath,featStatsFilePath);
         }
    VaeSubLightModel(const Bitmap1fD&alpha, const Bitmap1fD &eta, const Bitmap3fD &reflectance, const Bitmap3fD &sigma_t, const Bitmap3fD &albedo)
        : VaeSub(alpha,eta,reflectance,sigma_t,albedo)
    { 
        m_anisotropic = false; 
        loadNetwork(variablePath,featStatsFilePath);
        const std::string variablePath = "None";
    }
    bool m_isFull = false;
    using Matrix64f = Matrix<float,32>;
    using Matrix68f = Matrix<float,36>;
    using Vector64f = Array<float,32>;
    using Vector32f = Array<float,16>;
    
    using Matrix64_23 = Array<Array<float,32>,23>;
    using Matrix32_64 = Array<Array<float,16>,32>;
    using Matrix64_64 = Array<Array<float,32>,32>;
    using Matrix1_32= Array<Array<float,1>,16>;
    using Matrix64_68 = Array<Array<float,32>,36>;
    using Matrix3_64 = Array<Array<float,3>,32>;

    Matrix32_64 m_absorption_mlp_fcn_0_weights = zero<Matrix32_64>(); // Eigen::Matrix<float,32,64>
    Vector32f m_absorption_mlp_fcn_0_biases = zero<Vector32f>();
    Matrix1_32 m_absorption_dense_kernel = zero<Matrix1_32>();
    Vector1f m_absorption_dense_bias = zero<Vector1f>();

    Matrix64_68 m_scatter_decoder_fcn_fcn_0_weights = zero<Matrix64_68>(); // [64,68]
    Matrix64_64 m_scatter_decoder_fcn_fcn_1_weights = zero<Matrix64_64>();
    Matrix64_64 m_scatter_decoder_fcn_fcn_2_weights = zero<Matrix64_64>();
    Vector64f m_scatter_decoder_fcn_fcn_0_biases = zero<Vector64f>();
    Vector64f m_scatter_decoder_fcn_fcn_1_biases = zero<Vector64f>();
    Vector64f m_scatter_decoder_fcn_fcn_2_biases = zero<Vector64f>();
    Matrix3_64 m_scatter_dense_2_kernel = zero<Matrix3_64>();
    Array<float,3> m_scatter_dense_2_bias = zero<Array<float,3>>();

    Vector64f shared_preproc_mlp_2_shapemlp_fcn_0_biases = zero<Vector64f>();
    Vector64f shared_preproc_mlp_2_shapemlp_fcn_1_biases = zero<Vector64f>();
    Vector64f shared_preproc_mlp_2_shapemlp_fcn_2_biases = zero<Vector64f>();

    Matrix64_23 m_shared_preproc_mlp_2_shapemlp_fcn_0_weights = zero<Matrix64_23>(); // [64,23]
    Matrix64_64 m_shared_preproc_mlp_2_shapemlp_fcn_1_weights = zero<Matrix64_64>();  // [64,64]
    Matrix64_64 m_shared_preproc_mlp_2_shapemlp_fcn_2_weights = zero<Matrix64_64>(); // [64,64]

    LOAD_NETWORK;
    
        
};
*/
} // namespace psdr
