#pragma once

#include <psdr/core/bitmap.h>
#include <enoki/matrix.h>
#include <enoki/array.h>
#include <enoki/dynamic.h>


// Added Eigen and Json 
#include <Eigen/Core>
#include <Eigen/Dense>
#include <psdr/bsdf/json.hpp>

#include "bsdf.h"
#define PreLayerWidth 64
#define LayerWidth 64
#define NLatent 4

using json = nlohmann::json;

namespace psdr
{
// typedef DynamicArray<DynamicArray<Packet<float,1>>> VectorXf;
using InnerDynamicArray = DynamicArray<float>;
using OuterDynamicArray = DynamicArray<InnerDynamicArray>;

PSDR_CLASS_DECL_BEGIN(VaeSub, final, BSDF)
public:
    VaeSub() : 
    m_alpha_u(0.1f), m_alpha_v(0.1f), m_eta(0.8f), 
    m_albedo(1.0f), m_sigma_t(0.5f),
    m_specular_reflectance(1.0f), m_g(0.0f) 
        {   
            m_anisotropic = false;
        loadNetwork();
         }
    void loadNetwork();
    VaeSub(const Bitmap1fD &alpha, const Bitmap1fD &eta, const Bitmap3fD &reflectance, const Bitmap3fD &sigma_t, const Bitmap3fD &albedo)
        : 
    m_alpha_u(alpha), m_alpha_v(alpha), m_eta(eta), m_specular_reflectance(1.0f), 
    m_albedo(albedo), m_sigma_t(sigma_t), m_g(0.0f) { 
        m_anisotropic = false; 
        loadNetwork();
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

    void setAlbedo(ScalarVector3f &albedo){
        m_albedo.fill(albedo);
    }

    void setSigmaT(ScalarVector3f &sigma_t){
        m_sigma_t.fill(sigma_t);
    }

    FloatC pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const override;
    FloatD pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const override;

    FloatC pdfpoint(const IntersectionC &its, const BSDFSampleC &bs, MaskC active) const override;
    FloatD pdfpoint(const IntersectionD &its, const BSDFSampleD &bs, MaskD active) const override;

    SpectrumC eval(const IntersectionC &its, const Vector3fC &bs, MaskC active = true) const override;
    SpectrumD eval(const IntersectionD &its, const Vector3fD &bs, MaskD active = true) const override;

    SpectrumC eval(const IntersectionC &its, const BSDFSampleC &bs, MaskC active = true) const override;
    SpectrumD eval(const IntersectionD &its, const BSDFSampleD &bs, MaskD active = true) const override;

    BSDFSampleC sample(const Scene *scene, const IntersectionC &its, const Vector8fC &sample, MaskC active = true) const override;
    BSDFSampleD sample(const Scene *scene, const IntersectionD &its, const Vector8fD &sample, MaskD active = true) const override;

    FloatC pdf(const IntersectionC &its, const BSDFSampleC &wo, MaskC active = true) const override;
    FloatD pdf(const IntersectionD &its, const BSDFSampleD &wo, MaskD active = true) const override;
    

    bool anisotropic() const override { return m_anisotropic; }
    bool hasbssdf() const override { return true; }

    std::string to_string() const override { return std::string("VaeSub[id=") + m_id + "]"; }

    Bitmap1fD m_alpha_u, m_alpha_v; // surface roughness
    Bitmap1fD m_eta, m_g;
    Bitmap3fD m_albedo, m_sigma_t; // medium features
    Bitmap3fD m_specular_reflectance; // reflectance

    using Matrix64f = Matrix<float,64>;
    using Matrix68f = Matrix<float,68>;
    using Vector64f = Array<float,64>;
    using Vector1f = Array<float,1>;
    using Vector20f = Array<float,20>;
    // using Vector3f = Array<float,3>;
    
    // Eigen::Matrix<float, 32, PreLayerWidth> absorption_mlp_fcn_0_weights; // Eigen::Matrix<float,32,64>
    // Eigen::Matrix<float, 32, 1> absorption_mlp_fcn_0_biases;
    // Eigen::Matrix<float, 1, 32> absorption_dense_kernel;
    // Eigen::Matrix<float, 1, 1> absorption_dense_bias;
    Matrix64f absorption_mlp_fcn_0_weights = zero<Matrix64f>(); // Eigen::Matrix<float,32,64>
    Vector64f absorption_mlp_fcn_0_biases = zero<Vector64f>();
    Matrix64f absorption_dense_kernel = zero<Matrix64f>();
    Vector1f absorption_dense_bias = zero<Vector1f>();

    // Eigen::Matrix<float, LayerWidth, 1> scatter_decoder_fcn_fcn_0_biases;
    // Eigen::Matrix<float, LayerWidth, 1> scatter_decoder_fcn_fcn_1_biases;
    // Eigen::Matrix<float, LayerWidth, 1> scatter_decoder_fcn_fcn_2_biases;
    Vector64f scatter_decoder_fcn_fcn_0_biases = zero<Vector64f>();
    Vector64f scatter_decoder_fcn_fcn_1_biases = zero<Vector64f>();
    Vector64f scatter_decoder_fcn_fcn_2_biases = zero<Vector64f>();

    Vector64f shared_preproc_mlp_2_shapemlp_fcn_0_biases = zero<Vector64f>();
    Vector64f shared_preproc_mlp_2_shapemlp_fcn_1_biases = zero<Vector64f>();
    Vector64f shared_preproc_mlp_2_shapemlp_fcn_2_biases = zero<Vector64f>();
    // Eigen::Matrix<float, 3, 1> scatter_dense_2_bias;
    Array<float,3> scatter_dense_2_bias = zero<Array<float,3>>();

    // Eigen::Matrix<float, LayerWidth, PreLayerWidth + NLatent> scatter_decoder_fcn_fcn_0_weights;
    // Eigen::Matrix<float, LayerWidth, LayerWidth> scatter_decoder_fcn_fcn_1_weights;
    // Eigen::Matrix<float, LayerWidth, LayerWidth> scatter_decoder_fcn_fcn_2_weights;
    // Eigen::Matrix<float, 3, LayerWidth> scatter_dense_2_kernel;
    Matrix68f scatter_decoder_fcn_fcn_0_weights = zero<Matrix68f>();
    Matrix64f scatter_decoder_fcn_fcn_1_weights = zero<Matrix64f>();
    Matrix64f scatter_decoder_fcn_fcn_2_weights = zero<Matrix64f>();
    Matrix64f scatter_dense_2_kernel = zero<Matrix64f>();

    Matrix64f shared_preproc_mlp_2_shapemlp_fcn_0_weights = zero<Matrix64f>(); // [64,23]
    Matrix64f shared_preproc_mlp_2_shapemlp_fcn_1_weights = zero<Matrix64f>();  // [64,64]
    Matrix64f shared_preproc_mlp_2_shapemlp_fcn_2_weights = zero<Matrix64f>(); // [64,64]

    // Eigen::Matrix<float, 20, 1> m_shapeFeatMean, m_shapeFeatStdInv;
    Vector20f m_shapeFeatMean,m_shapeFeatStdInv;
    
    json stats; 
    
    float m_albedoMean, m_albedoStdInv, m_gMean, m_gStdInv;

    bool m_anisotropic;

protected:
    template <bool ad>
    BSDFSample<ad> __sample(const Scene *scene, const Intersection<ad>&, const Vector8f<ad>&, Mask<ad>) const;
   
    template <bool ad>
    Float<ad> __pdf(const Intersection<ad> &, const BSDFSample<ad> &, Mask<ad>) const;
    // template <bool ad>
    // Float<ad> __pdf_better(const Intersection<ad> &, const BSDFSample<ad> &, Mask<ad>) const;
    template <bool ad>
    Float<ad> __pdfspecular(const Intersection<ad> &, const BSDFSample<ad> &, Mask<ad>) const;

    template <bool ad>
    Spectrum<ad> __eval(const Intersection<ad>&, const BSDFSample<ad>&, Mask<ad>) const;
    
    // Joon added
    template <bool ad,size_t size>
    Spectrum<ad> _run(Array<Float<ad>,size> x, Array<Float<ad>,4> latent) const;

    template <bool ad,size_t size>
    Array<Float<ad>,size> _preprocessFeatures(const Intersection<ad>&its, int idx, bool isPlane) const;

    template <bool ad,size_t size>
    Array<Float<ad>,size> _preprocessFeatures(const Intersection<ad>&its, Int<ad> idx, bool isPlane) const;

    template <bool ad>
    Float<ad> getKernelEps(const Intersection<ad>& its,int idx) const;

    template <bool ad>
    Float<ad> getKernelEps(const Intersection<ad>& its,Int<ad> idx) const;
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
    template <bool ad>
    std::pair<Intersection<ad>,Int<ad>> __sample_sp(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Float<ad> &pdf, Mask<ad>active) const;
    // template <bool ad>
    // Intersection<ad> __get_intersection(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Vector3f<ad> vx, Vector3f<ad> vy, Vector3f<ad> vz, Float<ad> l, Float<ad> r, Float<ad> phi, Mask<ad> active) const;
    // template <bool ad>
    // Intersection<ad> __selectRandomSample(const Int<ad> randomsample, std::vector<Intersection<ad>> inters) const;
    
    template <bool ad>
    Float<ad> __pdf_sub(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const;
    template <bool ad>
    Float<ad> __pdf_bsdf(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const;

    template <bool ad>
    BSDFSample<ad> __sample_bsdf(const Intersection<ad> &its, const Vector8f<ad> &sample, Mask<ad> active) const;
    template <bool ad>
    BSDFSample<ad> __sample_sub(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Mask<ad> active) const;
    template <bool ad>
    Intersection<ad> __sample_better_sp(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Float<ad> &pdf, Mask<ad> active) const;
    template <bool ad>
    Spectrum<ad> __eval_bsdf(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const;
    template <bool ad>
    Spectrum<ad> __eval_sub(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const;
    // help functions
    template <bool ad> Spectrum<ad> __FresnelMoment1(Spectrum<ad> eta) const;
    template <bool ad> Spectrum<ad> __FresnelMoment2(Spectrum<ad> eta) const;
    template <bool ad> Spectrum<ad> __FersnelDi(Spectrum<ad> eta_i, Spectrum<ad> eta_o, Float<ad> cos_theta_i) const;

PSDR_CLASS_DECL_END(VaeSub)

} // namespace psdr
