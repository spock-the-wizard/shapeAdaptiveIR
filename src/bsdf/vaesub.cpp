#include <psdr/core/bitmap.h>
#include <psdr/core/warp.h>
#include <psdr/core/ray.h>
#include <psdr/bsdf/vaesub.h>
#include <psdr/scene/scene.h>
#include <psdr/utils.h>
#include <psdr/bsdf/ggx.h>

#include <enoki/cuda.h>
#include <enoki/autodiff.h>
#include <enoki/special.h>
#include <enoki/random.h>

#define UNUSED(x) (void)(x)
#include "scattereigen.h"
template <typename T, int R, int C>
using Matrix = enoki::Array<Array<T,C>, R>;

namespace psdr
{
    void VaeSub::loadNetwork() {
        //Load Network
        const std::string variablePath = "../../../variables";
        const std::string featStatsFilePath = "../../../data_stats.json";
        const std::string shapeFeaturesName = "mlsPolyLS3";
        int PolyOrder = 3;
        
        
        NetworkHelpers::loadVector(variablePath + "/absorption_dense_bias.bin", (float*) &absorption_dense_bias);
        NetworkHelpers::loadVector(variablePath + "/absorption_mlp_fcn_0_biases.bin", (float*) &absorption_mlp_fcn_0_biases);
        // absorption_dense_bias =
        //     NetworkHelpers::loadVectorDynamic(variablePath + "/absorption_dense_bias.bin");
        // absorption_mlp_fcn_0_biases =
        //     NetworkHelpers::loadVectorDynamic(variablePath + "/absorption_mlp_fcn_0_biases.bin");

        NetworkHelpers::loadVector(variablePath + "/scatter_decoder_fcn_fcn_0_biases.bin",(float*) &scatter_decoder_fcn_fcn_0_biases);
        NetworkHelpers::loadVector(variablePath + "/scatter_decoder_fcn_fcn_1_biases.bin",(float*) &scatter_decoder_fcn_fcn_1_biases);
        NetworkHelpers::loadVector(variablePath + "/scatter_decoder_fcn_fcn_2_biases.bin",(float*) &scatter_decoder_fcn_fcn_2_biases);
        // scatter_decoder_fcn_fcn_0_biases =
        //     NetworkHelpers::loadVectorDynamic(variablePath + "/scatter_decoder_fcn_fcn_0_biases.bin");
        // scatter_decoder_fcn_fcn_1_biases =
        //     NetworkHelpers::loadVectorDynamic(variablePath + "/scatter_decoder_fcn_fcn_1_biases.bin");
        // scatter_decoder_fcn_fcn_2_biases =
        //     NetworkHelpers::loadVectorDynamic(variablePath + "/scatter_decoder_fcn_fcn_2_biases.bin");
        
        NetworkHelpers::loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_biases.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_0_biases);
        NetworkHelpers::loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_biases.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_1_biases);
        NetworkHelpers::loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_biases.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_2_biases);
        // shared_preproc_mlp_2_shapemlp_fcn_0_biases =
        //     NetworkHelpers::loadVectorDynamic(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_biases.bin");
        // shared_preproc_mlp_2_shapemlp_fcn_1_biases =
        //     NetworkHelpers::loadVectorDynamic(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_biases.bin");
        // shared_preproc_mlp_2_shapemlp_fcn_2_biases =
            // NetworkHelpers::loadVectorDynamic(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_biases.bin");
        NetworkHelpers::loadVector(variablePath + "/scatter_dense_2_bias.bin",(float*) &scatter_dense_2_bias);
        NetworkHelpers::loadMat(variablePath + "/scatter_decoder_fcn_fcn_0_weights.bin",(float*) &scatter_decoder_fcn_fcn_0_weights,68);
        NetworkHelpers::loadMat(variablePath + "/scatter_decoder_fcn_fcn_1_weights.bin",(float*) &scatter_decoder_fcn_fcn_1_weights);
        NetworkHelpers::loadMat(variablePath + "/scatter_decoder_fcn_fcn_2_weights.bin",(float*) &scatter_decoder_fcn_fcn_2_weights);
        // scatter_dense_2_bias = NetworkHelpers::loadVectorDynamic(variablePath + "/scatter_dense_2_bias.bin");
        // scatter_decoder_fcn_fcn_0_weights =
        //     NetworkHelpers::loadMatrixDynamic(variablePath + "/scatter_decoder_fcn_fcn_0_weights.bin");
        // scatter_decoder_fcn_fcn_1_weights =
        //     NetworkHelpers::loadMatrixDynamic(variablePath + "/scatter_decoder_fcn_fcn_1_weights.bin");
        // scatter_decoder_fcn_fcn_2_weights =
        //     NetworkHelpers::loadMatrixDynamic(variablePath + "/scatter_decoder_fcn_fcn_2_weights.bin");
        
        NetworkHelpers::loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_weights.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_0_weights);
        NetworkHelpers::loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_weights.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_1_weights);
        NetworkHelpers::loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_weights.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_2_weights);

        // shared_preproc_mlp_2_shapemlp_fcn_0_weights =
        //     NetworkHelpers::loadMatrixDynamic(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_weights.bin");
        // shared_preproc_mlp_2_shapemlp_fcn_1_weights =
        //     NetworkHelpers::loadMatrixDynamic(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_weights.bin");
        // shared_preproc_mlp_2_shapemlp_fcn_2_weights =
        //     NetworkHelpers::loadMatrixDynamic(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_weights.bin");
        // scatter_dense_2_kernel = NetworkHelpers::loadMatrixDynamic(variablePath + "/scatter_dense_2_kernel.bin");
        NetworkHelpers::loadMat(variablePath + "/scatter_dense_2_kernel.bin", (float*) &scatter_dense_2_kernel);
        
        
        NetworkHelpers::loadMat(variablePath + "/absorption_dense_kernel.bin",(float*) &absorption_dense_kernel);
        NetworkHelpers::loadMat(variablePath + "/absorption_mlp_fcn_0_weights.bin",(float*) &absorption_mlp_fcn_0_weights);
        // absorption_dense_kernel = NetworkHelpers::loadMatrixDynamic(variablePath + "/absorption_dense_kernel.bin");
        // absorption_mlp_fcn_0_weights = NetworkHelpers::loadMatrixDynamic(variablePath + "/absorption_mlp_fcn_0_weights.bin");


        std::ifstream inStreamStats(featStatsFilePath);
        if (inStreamStats) {
            std::cout << "Loading statistics...\n";

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

SpectrumC VaeSub::eval(const IntersectionC &its, const Vector3fC &wo, MaskC active) const {
    // check and reverse this
    BSDFSampleC bs;
    bs.po = its;
    bs.wo = wo;
    bs.is_valid = true;
    return __eval<false>(its, bs, active);
}

SpectrumD VaeSub::eval(const IntersectionD &its, const Vector3fD &wo, MaskD active) const {
    // check and reverse this
    BSDFSampleD bs;
    bs.po = its;
    bs.wo = wo;
    bs.is_valid = true;
    return __eval<true>(its, bs, active);
}

SpectrumC VaeSub::eval(const IntersectionC &its, const BSDFSampleC &bs, MaskC active) const {
    return __eval<false>(its, bs, active);
}


SpectrumD VaeSub::eval(const IntersectionD &its, const BSDFSampleD &bs, MaskD active) const {
    return __eval<true>(its, bs, active);
}

FloatC VaeSub::pdfpoint(const IntersectionC &its, const BSDFSampleC &bs, MaskC active) const {
    FloatC cos_theta_i = Frame<false>::cos_theta(detach(its.wi));
    SpectrumC Fersnelterm = __FersnelDi<false>(1.0f, m_eta.eval<false>(detach(its.uv)), cos_theta_i);
    FloatC F = Fersnelterm.x();
    FloatC value = select(bs.is_sub,  __pdf_sub<false>(its, bs, active), F);
    return value & active;
}

FloatD VaeSub::pdfpoint(const IntersectionD &its, const BSDFSampleD &bs, MaskD active) const {
    FloatD cos_theta_i = Frame<true>::cos_theta(its.wi);
    SpectrumD Fersnelterm = __FersnelDi<true>(1.0f, m_eta.eval<true>(its.uv), cos_theta_i);
    FloatD F = Fersnelterm.x();
    FloatD value = select(bs.is_sub, __pdf_sub<true>(its, bs, active), F);
    return value & active;
}

using namespace enoki;

// new version
template <bool ad>
Spectrum<ad> VaeSub::__Sp(const Intersection<ad>&its, const Intersection<ad>&bs) const {
    // Classic diphole
    Float<ad> dist = norm(its.p - bs.p);
    Float<ad> r = dist; //detach(dist);

    Spectrum<ad> albedo = m_albedo.eval<ad>(its.uv);
    Spectrum<ad> sigma_s = m_sigma_t.eval<ad>(its.uv) * albedo;
    Spectrum<ad> sigma_a = m_sigma_t.eval<ad>(its.uv) - sigma_s;


    Spectrum<ad> miu_s_p = (1.0f - m_g.eval<ad>(its.uv)) * sigma_s;
    Spectrum<ad> miu_t_p = miu_s_p + sigma_a;
    Spectrum<ad> alpha_p = miu_s_p / miu_t_p;

    Spectrum<ad> eta = m_eta.eval<ad>(its.uv);
    Spectrum<ad> C_1 = __FresnelMoment1<ad>(eta);

    Spectrum<ad> D = 1.0f / (3.0f * miu_t_p);
    Spectrum<ad> A = (1.0f + 2.0f * C_1) / (1.0f - 2.0f * C_1);
    Spectrum<ad> Z_b = 2.0f * A * D;
    Spectrum<ad> Z_r = 1.0f / miu_t_p;
    Spectrum<ad> Z_v = -Z_r - 2.0f * Z_b;
    
    Spectrum<ad> d_r = sqrt(Z_r * Z_r + r * r);
    Spectrum<ad> d_v = sqrt(Z_v * Z_v + r * r);
    Spectrum<ad> miu_tr = sqrt(sigma_a / D);
    
    Spectrum<ad> real_part = (Z_r * (miu_tr * d_r + 1.0f) / (d_r * d_r)) * exp(-miu_tr * d_r) / d_r;
    Spectrum<ad> virtual_part = (Z_v * (miu_tr * d_v + 1.0f) / (d_v * d_v)) * exp(-miu_tr * d_v) / d_v;

    Spectrum<ad> scaler = alpha_p / (4.0f * Pi);
    Spectrum<ad> value = (real_part - virtual_part) * scaler; // -Z_v * miu_tr * d_v / (d_v * d_v) * exp(-miu_tr * d_v) / d_v; //
    
    return value;
}


template <bool ad>
Spectrum<ad> VaeSub::__Sw(const Intersection<ad>&its, const Vector3f<ad>&wo, Mask<ad >active) const {
    Spectrum<ad> c = 1.0f - 2.0f * __FresnelMoment1<ad>(1.0f/m_eta.eval<ad>(its.uv)); 
    Spectrum<ad> value = (1.0f - __FersnelDi<ad>(1.0f, m_eta.eval<ad>(its.uv), Frame<ad>::cos_theta(wo))) / (c * Pi);
    return value & active;
}

template <bool ad>
Float<ad> VaeSub::__pdf_wo(const Intersection<ad> &its, const Vector3f<ad> &wo, Mask<ad> active) const{
    FloatC cos_theta_i, cos_theta_o;
    if constexpr ( ad ) {
        cos_theta_i = FrameC::cos_theta(detach(its.wi));
        cos_theta_o = FrameC::cos_theta(detach(wo));
    } else {
        cos_theta_i = FrameC::cos_theta(its.wi);
        cos_theta_o = FrameC::cos_theta(wo);
    }
    active &= (cos_theta_i > 0.f && cos_theta_o > 0.f);
    Float<ad> value = cos_theta_o * InvPi;
    return value & active;
}

// help functions
template <bool ad> Spectrum<ad> VaeSub::__FresnelMoment1(Spectrum<ad> eta) const {
    Spectrum<ad> eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta, eta5 = eta4 * eta;

    return select(eta < 1.0f, 0.45966f - 1.73965f * eta + 3.37668f * eta2 - 3.904945f * eta3 +
            2.49277f * eta4 - 0.68441f * eta5, -4.61686f + 11.1136f * eta - 10.4646f * eta2 + 5.11455f * eta3 -
            1.27198f * eta4 + 0.12746f * eta5);
}

template <bool ad> Spectrum<ad> VaeSub::__FresnelMoment2(Spectrum<ad> eta) const {
    Spectrum<ad> eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta, eta5 = eta4 * eta;
    Spectrum<ad> r_eta = 1 / eta, r_eta2 = r_eta * r_eta, r_eta3 = r_eta2 * r_eta;

    return select(eta < 1.0f, 0.27614f - 0.87350f * eta + 1.12077f * eta2 - 0.65095f * eta3 +
            0.07883f * eta4 + 0.04860f * eta5, -547.033f + 45.3087f * r_eta3 - 218.725f * r_eta2 +
            458.843f * r_eta + 404.557f * eta - 189.519f * eta2 +
            54.9327f * eta3 - 9.00603f * eta4 + 0.63942f * eta5);
}

template <bool ad>
    Spectrum<ad> VaeSub::__FersnelDi(Spectrum<ad> etai, Spectrum<ad> etat, Float<ad> cos_theta_i) const {
        Float<ad> cosThetaI = clamp(cos_theta_i, -1.0f, 1.0f);
        Spectrum<ad> etaI = select(cosThetaI > 0.f, etai , etat);
        Spectrum<ad> etaT = select(cosThetaI > 0.f, etat , etai);
        cosThetaI = select(cosThetaI > 0.f, cosThetaI, -cosThetaI);

        Float<ad> sinThetaI = sqr(max(0.f, 1.0f - cosThetaI * cosThetaI));
        // Mask<ad> verti = sinThetaI >= 1.0f;

        Spectrum<ad> sinThetaT = etaI / etaT * sinThetaI;
        Spectrum<ad> cosThetaT = sqr(max(0.f, 1.0f - sinThetaT * sinThetaT));
        
        Spectrum<ad> Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + etaI * cosThetaT);
        Spectrum<ad> Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /  ((etaI * cosThetaI) + etaT * cosThetaT);

        Spectrum<ad> result = (Rparl * Rparl + Rperp * Rperp) * 0.5f;
        // masked(result, verti) = Spectrum<ad>(1.0f);
        result = select(sinThetaI >= 1.0f, Spectrum<ad>(1.0f), result);

        return result;
    }

template <bool ad>
Spectrum<ad> VaeSub::__eval(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const {
    
    return select(bs.is_sub, __eval_sub<ad>(its, bs, active), __eval_bsdf<ad>(its, bs, active));
}

template <bool ad>
Spectrum<ad> VaeSub::__eval_bsdf(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const {
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi),
              cos_theta_o = Frame<ad>::cos_theta(bs.wo);
    active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

    Float<ad> alpha_u = m_alpha_u.eval<ad>(its.uv);
    Float<ad> alpha_v = m_alpha_u.eval<ad>(its.uv);

    GGXDistribution m_distr(alpha_u, alpha_v);
    Vector3f<ad> H = normalize(bs.wo + its.wi);

    Float<ad> D = m_distr.eval<ad>(H);
    active &= neq(D, 0.f);
    Float<ad> G = m_distr.G<ad>(its.wi, bs.wo, H);
    Spectrum<ad> F = __FersnelDi<ad>(1.0f, m_eta.eval<ad>(its.uv), dot(its.wi, H));
    Spectrum<ad> result = F * D * G / (4.f * Frame<ad>::cos_theta(its.wi));
    
    Spectrum<ad> specular_reflectance = m_specular_reflectance.eval<ad>(its.uv);

    return (result * specular_reflectance * cos_theta_o) & active;
}

template <bool ad>
Spectrum<ad> VaeSub::__eval_sub(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const {
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
    Float<ad> cos_theta_o = Frame<ad>::cos_theta(bs.wo);
    active &= (cos_theta_i > 0.f);
    active &= (cos_theta_o > 0.f);

    Spectrum<ad> F = __FersnelDi<ad>(1.0f, m_eta.eval<ad>(its.uv), cos_theta_i);

    // Joon added: sample contribution for each channel
    Spectrum<ad> r = zero<Spectrum<ad>>();
    Spectrum<ad> g = zero<Spectrum<ad>>();
    Spectrum<ad> b = zero<Spectrum<ad>>();
    r[0] = 1.0f;
    g[1] = 1.0f;
    b[2] = 1.0f;
    // std::cout << "bs.rgb" << bs.rgb << std::endl;
    // Spectrum<ad> sp = sqrt(__Sp<ad>(bs.po, its)) * sqrt(__Sp<ad>(its, bs.po));
    Spectrum<ad> sp = select(bs.rgb == 0, r, g);
    sp = select(bs.rgb == 2, b, sp); 
    std::cout << "bs.rgb " << bs.rgb << std::endl;
    std::cout << "sp " << sp << std::endl;
    Spectrum<ad> sw =  __Sw<ad>(bs.po, bs.wo, active);
    
    Spectrum<ad> value = sp * sw * (1.0f - F) * cos_theta_o;
    if constexpr ( ad ) {
        value = value * bs.po.J;    
    }
    
    return value & active;
}

BSDFSampleC VaeSub::sample(const Scene *scene, const IntersectionC &its, const Vector8fC &rand, MaskC active) const {
    return __sample<false>(scene, its, rand, active);
}


BSDFSampleD VaeSub::sample(const Scene *scene, const IntersectionD &its, const Vector8fD &rand, MaskD active) const {

    return __sample<true>(scene, its, rand, active);
}


template <bool ad>
BSDFSample<ad> VaeSub::__sample(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Mask<ad> active) const {
    std::cout << "entering __sample" << std::endl;
    BSDFSample<ad> bs = __sample_sub<ad>(scene, its, sample, active);
    BSDFSample<ad> bsdf_bs =  __sample_bsdf<ad>(its, sample, active);

    FloatC cos_theta_i = Frame<false>::cos_theta(detach(its.wi));
    SpectrumC probv = __FersnelDi<false>(1.0f, m_eta.eval<false>(detach(its.uv)), cos_theta_i);
    FloatC prob = probv.x();
    
    bs.wo = select(sample.x() > prob, bs.wo, bsdf_bs.wo);
    bs.is_sub = select(sample.x() > prob, bs.is_sub, bsdf_bs.is_sub);
    bs.pdf = select(sample.x() > prob, bs.pdf, bsdf_bs.pdf);
    bs.is_valid = select(sample.x() > prob, bs.is_valid, bsdf_bs.is_valid);

    bs.po.num = select(sample.x() > prob, bs.po.num, bsdf_bs.po.num);
    bs.po.n = select(sample.x() > prob, bs.po.n, bsdf_bs.po.n);
    bs.po.p = select(sample.x() > prob, bs.po.p, bsdf_bs.po.p);
    bs.po.J = select(sample.x() > prob, bs.po.J, bsdf_bs.po.J);
    bs.po.uv = select(sample.x() > prob, bs.po.uv, bsdf_bs.po.uv);
    bs.po.shape = select(sample.x() > prob, bs.po.shape, bsdf_bs.po.shape);
    bs.po.t = select(sample.x() > prob, bs.po.t, bsdf_bs.po.t);
    bs.po.wi = select(sample.x() > prob, bs.po.wi, bsdf_bs.po.wi);
    bs.po.sh_frame.s = select(sample.x() > prob, bs.po.sh_frame.s, bsdf_bs.po.sh_frame.s);
    bs.po.sh_frame.t = select(sample.x() > prob, bs.po.sh_frame.t, bsdf_bs.po.sh_frame.t);
    bs.po.sh_frame.n = select(sample.x() > prob, bs.po.sh_frame.n, bsdf_bs.po.sh_frame.n);

    // Joon added
    std::cout << "sample.x() > prob " << (sample.x() > prob) << std::endl;
    std::cout << "bs.rgb at sample()" << bs.rgb << std::endl;
    bs.rgb = select(sample.x() > prob, bs.rgb, bsdf_bs.rgb);
    return bs;
}

template <bool ad>
BSDFSample<ad> VaeSub::__sample_bsdf(const Intersection<ad> &its, const Vector8f<ad> &sample, Mask<ad> active) const {
    BSDFSample<ad> bs;
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
    Float<ad> alpha_u = m_alpha_u.eval<ad>(its.uv);
    Float<ad> alpha_v = m_alpha_u.eval<ad>(its.uv);
    GGXDistribution distr(alpha_u, alpha_v);
    Vector3f<ad> wo = distr.sample<ad>(its.wi, tail<3>(sample));
    bs.wo = fmsub(Vector3f<ad>(wo), 2.f * dot(its.wi, wo), its.wi);
    bs.wo = normalize(bs.wo);
    Vector3f<ad> H = normalize(bs.wo + its.wi);
    Spectrum<ad> fresnel = __FersnelDi<ad>(1.0f, m_eta.eval<ad>(its.uv), cos_theta_i);
    Float<ad> F = fresnel.x();
    bs.po = its;
    bs.pdf = __pdf_bsdf<ad>(its, bs, active) * F;
    bs.is_valid = (cos_theta_i > 0.f) & active;
    bs.is_sub = false;
    //Joon added
    bs.rgb = full<Int<ad>>(0);
    return bs;
}


template <bool ad>
BSDFSample<ad> VaeSub::__sample_sub(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Mask<ad> active) const {
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
    BSDFSample<ad> bs;
    Float<ad> pdf_po = Frame<ad>::cos_theta(its.wi);
    std::tie(bs.po, bs.rgb) = __sample_sp<ad>(scene, its, sample, pdf_po, active);
    std::cout << "sample_sub" << bs.rgb << std::endl;
    bs.wo = warp::square_to_cosine_hemisphere<ad>(tail<2>(sample));
    bs.pdf = pdf_po;
    // TODO: add an id check
    bs.is_valid = active && (cos_theta_i > 0.f) && bs.po.is_valid();// && ( == its.shape->m_id);&& 
    bs.is_sub = true;
    
    pdf_po = __pdf_sub<ad>(its, bs, active) * warp::square_to_cosine_hemisphere_pdf<ad>(bs.wo);
    bs.pdf = pdf_po;
    return bs;
}

FloatC VaeSub::pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active) const {
    return __pdf_wo<false>(its, wo, active);
}

FloatD VaeSub::pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active) const {
    return __pdf_wo<true>(its, wo, active);
}

FloatC VaeSub::pdf(const IntersectionC &its, const BSDFSampleC &bs, MaskC active) const {
    return __pdf<false>(its, bs, active);
}


FloatD VaeSub::pdf(const IntersectionD &its, const BSDFSampleD &bs, MaskD active) const {
    return __pdf<true>(its, bs, active);
}

template <bool ad>
Float<ad> VaeSub::__pdf(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const {

    Float<ad> value = select(bs.is_sub, __pdf_sub<ad>(its, bs, active) * warp::square_to_cosine_hemisphere_pdf<ad>(bs.wo),  __pdf_bsdf<ad>(its, bs, active));
    return value & active;
}

template <bool ad>
Float<ad> VaeSub::__pdf_bsdf(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const {
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi),
              cos_theta_o = Frame<ad>::cos_theta(bs.wo);

    Spectrum<ad> m = normalize(bs.wo + its.wi);
    
    active &= cos_theta_i > 0.f && cos_theta_o > 0.f &&
              dot(its.wi, m) > 0.f && dot(bs.wo, m) > 0.f;

    Float<ad> alpha_u = m_alpha_u.eval<ad>(its.uv);
    Float<ad> alpha_v = m_alpha_u.eval<ad>(its.uv);
    GGXDistribution distr(alpha_u, alpha_v);
    Float<ad> result = distr.eval<ad>(m) * distr.smith_g1<ad>(its.wi, m) /
                       (4.f * cos_theta_i);
    Spectrum<ad> fresnel = __FersnelDi<ad>(1.0f, m_eta.eval<ad>(its.uv), cos_theta_i);
    Float<ad> F = fresnel.x();
    return result * F;
}

template <bool ad>
Float<ad> VaeSub::__pdf_sub(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const {
    Spectrum<ad> sigma_s = m_sigma_t.eval<ad>(its.uv) * m_albedo.eval<ad>(its.uv);
    Spectrum<ad> sigma_a = m_sigma_t.eval<ad>(its.uv) - sigma_s;

    Spectrum<ad> miu_s_p = (1.0f - m_g.eval<ad>(its.uv)) * sigma_s;
    // better diphole
    // Spectrum<ad> D = (2.0f * sigma_a + miu_s_p) / (3.0f * (sigma_a + miu_s_p) * (sigma_a + miu_s_p));
    Spectrum<ad> D = 1.0f / (3.0f * (miu_s_p + sigma_a));

    Spectrum<ad> miu_tr = sqrt(sigma_a / D);

    Float<ad> d = norm(its.p - bs.po.p);


    // FloatC denominator;
    // if constexpr(ad){
    //     denominator = detach(miu_tr).x() + detach(mi
    //     .u_tr).y() + detach(miu_tr).z();
    // } else {
    //     denominator = miu_tr.x() + miu_tr.y() + miu_tr.z();
    // }


    Vector3f<ad> dv = its.p - bs.po.p;
    Vector3f<ad> dLocal(dot(its.sh_frame.s, dv), dot(its.sh_frame.t, dv), dot(its.sh_frame.n, dv));
    // Vector3f<ad> nLocal(dot(its.sh_frame.s, bs.po.n), dot(its.sh_frame.t, bs.po.n), dot(its.sh_frame.n, bs.po.n));
    Vector3f<ad> nLocal(dot(its.sh_frame.s, bs.po.n), dot(its.sh_frame.t, bs.po.n), dot(its.sh_frame.n, bs.po.n));
    // should I use geometry normal here?

    Vector3f<ad> rProj(sqrt(dLocal.y() * dLocal.y() + dLocal.z() * dLocal.z()), 
                        sqrt(dLocal.x() * dLocal.x() + dLocal.z() * dLocal.z()),
                        sqrt(dLocal.y() * dLocal.y() + dLocal.x() * dLocal.x()));

    Float<ad> pdf = 0.0f;
    // Joon: i for RGB channel
    for (int i = 0 ; i < 3; i++){
        // Float<ad> pp = miu_tr[i] / denominator;
        // miu_tr[i]
        pdf += __pdf_sr<ad>(miu_tr[i], rProj.x(), active) * 0.25f * abs(nLocal.x()) / 3.0f;
        pdf += __pdf_sr<ad>(miu_tr[i], rProj.y(), active) * 0.25f * abs(nLocal.y()) / 3.0f;
        pdf += __pdf_sr<ad>(miu_tr[i], rProj.z(), active) * 0.5f * abs(nLocal.z()) / 3.0f;
    }
    // FIXME: tmp setting
    pdf = 1.0f / 3.0f;

    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
    Spectrum<ad> Fersnelterm = __FersnelDi<ad>(1.0f, m_eta.eval<ad>(its.uv), cos_theta_i);
    Float<ad> F = Fersnelterm.x();

    Float<ad> value = (1.0f - F) * pdf / bs.po.num;

    return value;
}

template <bool ad>
Float<ad> VaeSub::__pdf_sr(const Float<ad> &miu_tr, const Float<ad> &x, Mask<ad> active) const {
    Float<ad> value = miu_tr * exp(-miu_tr * x) / (x * 2 * Pi);
    return value & active;
}

template <bool ad>
Float<ad> VaeSub::__sample_sr(const Float<ad> &miu_t, const Float<ad> &x) const {
    Float<ad> value = -log(1.0f - x) / miu_t;
    return value;
}

template <typename T, int C, int R>
using Matrix = enoki::Array<Array<T,C>,R>;

template <typename Array>
void print_nested_array(const Array& arr) {
    for (size_t i = 0; i < arr.size(); ++i) {
        std::cout << "Array[" << i << "]: " << std::to_string(arr[i]) << std::endl;
    }
}
template <typename Array>
void printArray(const Array& arr) {
    for (size_t i = 0; i < arr.size(); ++i) {
        if constexpr (enoki::is_array_v<typename Array::Value>) {
            // If the element is also an array, recurse
            std::cout << "[" << i <<"]" << " ";
            printArray(arr[i]);
        } else {
            // Base case: print the element
            std::cout << arr[i] << " ";
        }
    }
    std::cout << std::endl; // Newline for readability
}

template <typename Arr>
Arr transposeArray(const Arr& arr) {
    const int rows = arr.size();
    const int cols = arr[0].size();

    Array<Array<float,rows>,cols> newArray;
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            newArray[j][i] = arr[i][j];
        }
    }
    // std::cout << std::endl; // Newline for readability
    return newArray;
}

template <bool ad,size_t size>
Spectrum<ad> VaeSub::_run(Array<Float<ad>,size> x,Array<Float<ad>,4> latent) const {
    // TODO: add absoprtion
        auto features = max(shared_preproc_mlp_2_shapemlp_fcn_0_weights * x + shared_preproc_mlp_2_shapemlp_fcn_0_biases,0.0f);
        features = max(shared_preproc_mlp_2_shapemlp_fcn_1_weights * features + shared_preproc_mlp_2_shapemlp_fcn_1_biases,0.0f);
        features = max(shared_preproc_mlp_2_shapemlp_fcn_2_weights * features + shared_preproc_mlp_2_shapemlp_fcn_2_biases,0.0f);

        auto abs = max(absorption_mlp_fcn_0_weights * features + absorption_mlp_fcn_0_biases,0.0f);
        abs = absorption_dense_kernel * abs + absorption_dense_bias[0];
        //TODO: check vectorization here
        // Apply sigmoid function
        auto absorption = 1.0f/(1.0f + exp(-abs[0]));

        // Gaussian noise
        latent = float(M_SQRT2) * erfinv(2.f*latent - 1.f);
        // std::cout<< "gaussian latent: "<<hmean(latent)<< std::endl;
        
        auto featLatent = concat(latent,features);
        auto y = head<64,Array<Float<ad>,68>>(scatter_decoder_fcn_fcn_0_weights * featLatent);
        y = max(y + scatter_decoder_fcn_fcn_0_biases,0.0f); 
        y = max(scatter_decoder_fcn_fcn_1_weights * y + scatter_decoder_fcn_fcn_1_biases,0.0f); 
        y = max(scatter_decoder_fcn_fcn_2_weights * y + scatter_decoder_fcn_fcn_2_biases,0.0f); 
        auto outPos = head<3,Array<Float<ad>,64>>(scatter_dense_2_kernel * y);
        outPos = outPos + scatter_dense_2_bias;
        
        return outPos;
}
template <bool ad,size_t size>
// Array<Float<ad>,size> VaeSub::_preprocessFeatures(const Intersection<ad>&its, int ch_idx, bool isPlane) const {
Array<Float<ad>,size> VaeSub::_preprocessFeatures(const Intersection<ad>&its, Int<ad> ch_idx, bool isPlane) const {
    // float scale = 1.0f;
    float scale = 25.0f;
    Spectrum<ad> albedo = m_albedo.eval<ad>(its.uv);
    Spectrum<ad> sigma_s = scale * m_sigma_t.eval<ad>(its.uv) * albedo;
    Spectrum<ad> sigma_a = scale * m_sigma_t.eval<ad>(its.uv) - sigma_s;
    Spectrum<ad> miu_s_p = (1.0f - m_g.eval<ad>(its.uv)) * sigma_s;
    Spectrum<ad> miu_t_p = miu_s_p + sigma_a;
    Spectrum<ad> alpha_p = miu_s_p / miu_t_p;

    Spectrum<ad> g = m_g.eval<ad>(its.uv);
    Spectrum<ad> eta = m_eta.eval<ad>(its.uv);

    auto effectiveAlbedo = -log(1.0f-alpha_p * (1.0f - exp(-8.0f))) / 8.0f;
    
    auto albedoNorm = (effectiveAlbedo - m_albedoMean) * m_albedoStdInv;
    auto gNorm = (g-m_gMean) * m_gStdInv;
    auto iorNorm = 2.0f * (eta - 1.25f);

    auto shapeFeatures = zero<Array<Float<ad>,20>>();
    if(isPlane)
        shapeFeatures[3] = 1.0;

    auto shapeFeaturesNorm = (shapeFeatures - m_shapeFeatMean)*m_shapeFeatStdInv;
    
    // auto b = concat(shapeFeaturesNorm,albedoNorm);
    // b = concat(b,gNorm);
    // b = concat(b,iorNorm);
    // Array<Float<ad>,size> x;
    Array<Float<ad>,64> x = full<Array<Float<ad>,64>>(0.0f); // = zero<Array<Float<ad>,64>>();
    for (int i=0;i<20;i++){
        x[i] = shapeFeaturesNorm[i];
    }

    // Choose RGB channel according to ch_idx
    x[20] = select(ch_idx == 0, albedoNorm.x(),albedoNorm.y());
    x[20] = select(ch_idx == 2, albedoNorm.z(),x[20]);
    x[21] = select(ch_idx == 0, gNorm.x(),gNorm.y());
    x[21] = select(ch_idx == 2, gNorm.z(),x[21]);
    x[22] = select(ch_idx == 0, iorNorm.x(),iorNorm.y());
    x[22] = select(ch_idx == 2, iorNorm.z(),x[22]);
        
    return x;
}
template <bool ad>
Float<ad> VaeSub::getKernelEps(const Intersection<ad>& its,Int<ad> idx) const {
// Float<ad> VaeSub::getKernelEps(const Intersection<ad>& its,int idx) const {
    
    Spectrum<ad> albedo = m_albedo.eval<ad>(its.uv);
    Spectrum<ad> sigma_s = m_sigma_t.eval<ad>(its.uv) * albedo;
    Spectrum<ad> sigma_a = m_sigma_t.eval<ad>(its.uv) - sigma_s;
    Spectrum<ad> miu_s_p = (1.0f - m_g.eval<ad>(its.uv)) * sigma_s;
    Spectrum<ad> miu_t_p = miu_s_p + sigma_a;
    Spectrum<ad> alpha_p = miu_s_p / miu_t_p;

    Spectrum<ad> g = m_g.eval<ad>(its.uv);
    Spectrum<ad> eta = m_eta.eval<ad>(its.uv);

    Spectrum<ad> effectiveAlbedo = -log(1.0f-alpha_p * (1.0f - exp(-8.0f))) / 8.0f;
    // std::cout << "alpha_p" << alpha_p << std::endl;
    // std::cout << "effectiveAlbedo" << effectiveAlbedo << std::endl;
    
    Spectrum<ad> val = 0.25f * g + 0.25f * alpha_p + 1.0f * effectiveAlbedo;
    // std::cout << "val" << val << std::endl;
    auto res = 4.0f * val * val / (miu_t_p * miu_t_p);
    // std::cout << "res" << res << std::endl;

    auto res_ch = select(idx == 0, res.x(), res.y());
    res_ch = select(idx == 2, res.z(), res_ch);
    return res_ch;
}

template <bool ad>
std::pair<Intersection<ad>,Int<ad>> VaeSub::__sample_sp(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Float<ad> &pdf, Mask<ad> active) const {        
    
        // Select RGB channel
        Float<ad> rnd = sample[5];
        Array<Int<ad>,3> index;
        for(int i=0;i<3;i++)
            index[i] = i;
        auto ch = select(rnd < (1.0f / 3.0f), index.x(),index.y());
        ch = select(rnd > 2.0f / 3.0f, index.z(), ch);
        
        auto x = _preprocessFeatures<ad,64>(its,ch,true);
        // auto x = zero<Array<Float<ad>,64>>();
        auto fitScaleFactor = 1.0f / sqrt(getKernelEps<ad>(its, ch));
        // std::cout << fitScaleFactor << std::endl;
        
        // Network Inference: computationally heavy from here...
        rnd = sample.y();
        Array<Float<ad>,4> latent(sample[2],sample[3],sample[6],sample[7]);
        Spectrum<ad> outPos = _run<ad,64>(x,latent);
        // std::cout<< "latent: "<<latent<< std::endl;
        // std::cout << "outPos" << outPos << std::endl;

        auto inNormal = -its.wi;
        auto inPos = its.p;
        auto n = inNormal;

        // Construct orthogonal frame
        Spectrum<ad> tangent1, tangent2;
        NetworkHelpers::onb<ad>(n, tangent1, tangent2);

        // Local to World coordinates 
        // FIXME disabled tmp 
        outPos = inPos + outPos.x() * tangent1 + outPos.y() * tangent2 + outPos.z() * inNormal;
        outPos = inPos + (outPos - inPos) / fitScaleFactor;
        // std::cout<<"outPos after transform" << outPos<<std::endl;
        // TMP setting for debugging
        // outPos = inPos + (outPos - inPos) / fitScaleFactor / 10.0f;
        
        // // Ray<ad> ray2(its.p, vz, l);
        // //NOTE: original code
        Vector3f<ad> vx = its.sh_frame.s;
        Vector3f<ad> vy = its.sh_frame.t;
        // TODO: choose projection direction
        Vector3f<ad> vz = its.sh_frame.n;

        pdf = 1.0f;
        
        // vx = select(rnd > 0.5f, its.sh_frame.t, vx);
        // vy = select(rnd > 0.5f, its.sh_frame.n, vy);
        // vz = select(rnd > 0.5f, its.sh_frame.s, vz);

        // vx = select(rnd > 0.75f, its.sh_frame.n, vx);
        // vy = select(rnd > 0.75f, its.sh_frame.s, vy);
        // vz = select(rnd > 0.75f, its.sh_frame.t, vz);
        
        Spectrum<ad> sigma_s = m_sigma_t.eval<ad>(its.uv) * m_albedo.eval<ad>(its.uv);
        Spectrum<ad> sigma_a = m_sigma_t.eval<ad>(its.uv) - sigma_s;

        Spectrum<ad> miu_s_p = (1.0f - m_g.eval<ad>(its.uv)) * sigma_s;
        Spectrum<ad> D = 1.0f / (3.0f * (sigma_a + miu_s_p));

        Spectrum<ad> miu_tr = sqrt(sigma_a / D);
        
        // // FloatC denominator, p1, p2;
        
        // // if constexpr (ad){
        // //     denominator =  (detach(miu_tr).x() + detach(miu_tr).y() + detach(miu_tr).z());
        // //     p1 = detach(miu_tr).x() / denominator;
        // //     p2 = (detach(miu_tr).x() + detach(miu_tr).y()) / denominator;
        // // } else {
        // //     denominator =  (miu_tr.x() + miu_tr.y() + miu_tr.z());
        // //     p1 = miu_tr.x() / denominator;
        // //     p2 = (miu_tr.x() + miu_tr.y()) / denominator;
        // // }
        
        Float<ad> miu_tr_0 = 0.5f;

        rnd = sample[5];
        miu_tr_0 = select(rnd < 1.0f / 3.0f, miu_tr.x(), miu_tr.y());
        miu_tr_0 = select(rnd > 2.0f / 3.0f, miu_tr.z(), miu_tr_0);
        
        Float<ad> r = __sample_sr<ad>(miu_tr_0, sample.z());
        Float<ad> phi = 2.0f * Pi * sample.w();
        Float<ad> rmax = __sample_sr<ad>(miu_tr_0, 0.999999f);

        Float<ad> l = 2.0f * sqrt(rmax * rmax - r * r);

        auto start = its.p + r * (vx * cos(phi) + vy * sin(phi)) - vz * 0.5f * l;
        // Ray<ad> ray2(start, vz, l);
        // std::cout<<"l" << l<<std::endl;
        // std::cout<<"start" << start <<std::endl;
        //TODO: projection
        float tmax = 40.0f;
        // Ray<ad> ray2(its.p + outPos / fitScaleFactor, vz, tmax);
        Ray<ad> ray2(outPos, vz, tmax);

        // std::cout << "ref:" << hmax(start) << " ours: " << hmax(outPos) << std::endl;
        // std::cout << "isnan" << any(any(isnan(outPos))) << std::endl;
        // std::cout << "isinf" << any(any(isinf(outPos))) << std::endl;
        // std::cout<<its.p<<std::endl;
        
        Intersection<ad> its2 = scene->ray_all_intersect<ad, ad>(ray2, active, sample, 5);

        return std::pair<Intersection<ad>,Int<ad>>(its2,ch);
    
} // namespace psdr

} // namespace psdr
