#include <psdr/core/bitmap.h>
#include <psdr/core/warp.h>
#include <psdr/core/ray.h>
#include <psdr/bsdf/vaesub.h>
#include<psdr/bsdf/polynomials.h>
#include <psdr/scene/scene.h>
#include <psdr/utils.h>
#include <psdr/bsdf/ggx.h>

#include <enoki/cuda.h>
#include <enoki/autodiff.h>
#include <enoki/special.h>
#include <enoki/random.h>

#include <sdf/sdf.hpp>
// #include <sdf/sdf.h>


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
        
        NetworkHelpers::loadMat(variablePath + "/scatter_decoder_fcn_fcn_0_weights.bin",(float*) &m_scatter_decoder_fcn_fcn_0_weights);
        NetworkHelpers::loadMat(variablePath + "/scatter_decoder_fcn_fcn_1_weights.bin",(float*) &m_scatter_decoder_fcn_fcn_1_weights);
        NetworkHelpers::loadMat(variablePath + "/scatter_decoder_fcn_fcn_2_weights.bin",(float*) &m_scatter_decoder_fcn_fcn_2_weights);
        NetworkHelpers::loadVector(variablePath + "/scatter_decoder_fcn_fcn_0_biases.bin",(float*) &m_scatter_decoder_fcn_fcn_0_biases);
        NetworkHelpers::loadVector(variablePath + "/scatter_decoder_fcn_fcn_1_biases.bin",(float*) &m_scatter_decoder_fcn_fcn_1_biases);
        NetworkHelpers::loadVector(variablePath + "/scatter_decoder_fcn_fcn_2_biases.bin",(float*) &m_scatter_decoder_fcn_fcn_2_biases);
        NetworkHelpers::loadMat(variablePath + "/scatter_dense_2_kernel.bin", (float*) &m_scatter_dense_2_kernel,3);
        NetworkHelpers::loadVector(variablePath + "/scatter_dense_2_bias.bin",(float*) &m_scatter_dense_2_bias);
        // NetworkHelpers::loadMat(variablePath + "/scatter_decoder_fcn_fcn_0_weights.bin",(float*) &scatter_decoder_fcn_fcn_0_weights,68);
        // NetworkHelpers::loadMat(variablePath + "/scatter_decoder_fcn_fcn_1_weights.bin",(float*) &scatter_decoder_fcn_fcn_1_weights);
        // NetworkHelpers::loadMat(variablePath + "/scatter_decoder_fcn_fcn_2_weights.bin",(float*) &scatter_decoder_fcn_fcn_2_weights);
        // NetworkHelpers::loadVector(variablePath + "/scatter_decoder_fcn_fcn_0_biases.bin",(float*) &scatter_decoder_fcn_fcn_0_biases);
        // NetworkHelpers::loadVector(variablePath + "/scatter_decoder_fcn_fcn_1_biases.bin",(float*) &scatter_decoder_fcn_fcn_1_biases);
        // NetworkHelpers::loadVector(variablePath + "/scatter_decoder_fcn_fcn_2_biases.bin",(float*) &scatter_decoder_fcn_fcn_2_biases);
        // NetworkHelpers::loadMat(variablePath + "/scatter_dense_2_kernel.bin", (float*) &scatter_dense_2_kernel);
        // NetworkHelpers::loadVector(variablePath + "/scatter_dense_2_bias.bin",(float*) &scatter_dense_2_bias);
        

        NetworkHelpers::loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_weights.bin",(float*) &m_shared_preproc_mlp_2_shapemlp_fcn_0_weights);
        NetworkHelpers::loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_weights.bin",(float*) &m_shared_preproc_mlp_2_shapemlp_fcn_1_weights);
        NetworkHelpers::loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_weights.bin",(float*) &m_shared_preproc_mlp_2_shapemlp_fcn_2_weights);
        NetworkHelpers::loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_biases.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_0_biases);
        NetworkHelpers::loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_biases.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_1_biases);
        NetworkHelpers::loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_biases.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_2_biases);
        // NetworkHelpers::loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_weights.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_0_weights);
        // NetworkHelpers::loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_weights.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_1_weights);
        // NetworkHelpers::loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_weights.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_2_weights);
        

        NetworkHelpers::loadMat(variablePath + "/absorption_dense_kernel.bin",(float*) &m_absorption_dense_kernel,1);
        NetworkHelpers::loadMat(variablePath + "/absorption_mlp_fcn_0_weights.bin",(float*) &m_absorption_mlp_fcn_0_weights,32); //[32,64]
        NetworkHelpers::loadVector(variablePath + "/absorption_dense_bias.bin", (float*) &m_absorption_dense_bias);
        NetworkHelpers::loadVector(variablePath + "/absorption_mlp_fcn_0_biases.bin", (float*) &m_absorption_mlp_fcn_0_biases);
        // NetworkHelpers::loadMat(variablePath + "/absorption_dense_kernel.bin",(float*) &absorption_dense_kernel);
        // NetworkHelpers::loadMat(variablePath + "/absorption_mlp_fcn_0_weights.bin",(float*) &absorption_mlp_fcn_0_weights);
        // NetworkHelpers::loadVector(variablePath + "/absorption_dense_bias.bin", (float*) &absorption_dense_bias);
        // NetworkHelpers::loadVector(variablePath + "/absorption_mlp_fcn_0_biases.bin", (float*) &absorption_mlp_fcn_0_biases);

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
    // Spectrum<ad> r = zero<Spectrum<ad>>();
    // Spectrum<ad> g = zero<Spectrum<ad>>();
    // Spectrum<ad> b = zero<Spectrum<ad>>();
    // r[0] = 1.0f;
    // g[1] = 1.0f;
    // b[2] = 1.0f;

    // FIXME: testing monochromatic
    Spectrum<ad> r = full<Spectrum<ad>>(1.0f);
    Spectrum<ad> g = full<Spectrum<ad>>(1.0f);
    Spectrum<ad> b = full<Spectrum<ad>>(1.0f);
    
    // std::cout << "bs.its.prob" << bs.po.abs_prob << std::endl;
    // std::cout << "bs.rgb_rv" << bs.rgb_rv << std::endl;
    auto sp = select(bs.rgb_rv < (1.0f/3.0f),r,g);
    sp = select(bs.rgb_rv > (2.0f/3.0f),b,sp);
    sp = sp * (1-bs.po.abs_prob) ;
    // sp = sp * 2;
    // Spectrum<ad> sp = sqrt(__Sp<ad>(bs.po, its)) * sqrt(__Sp<ad>(its, bs.po));
    // sp = select(bs.rgb == 2, b, sp); 
    // std::cout << "sp " << sp << std::endl;
    Spectrum<ad> sw =  __Sw<ad>(bs.po, bs.wo, active);
    
    // FIXME : testing kernel idea
    sp = __Sp<ad>(bs.po, its) * sp *0.1f;
    
    Spectrum<ad> value = sp* (1.0f - F) *sw * cos_theta_o;
    if constexpr ( ad ) {
        value = value * bs.po.J;    
    }
    
    // std::cout << "value " << value << std::endl;
    // std::cout << "cos_theta_o" << cos_theta_o<< std::endl;
    // std::cout << "F" << F<< std::endl;
    return value & active;
}

BSDFSampleC VaeSub::sample(const Scene *scene, const IntersectionC &its, const Vector8fC &rand, MaskC active) const {
    return __sample<false>(scene, its, rand, active);
}


BSDFSampleD VaeSub::sample(const Scene *scene, const IntersectionD &its, const Vector8fD &rand, MaskD active) const {

    return __sample<true>(scene, its,rand, active);
}


template <bool ad>
BSDFSample<ad> VaeSub::__sample(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Mask<ad> active) const {
    BSDFSample<ad> bs = __sample_sub<ad>(scene, its, sample, active);
    // std::cout << "[DEBUG] __sample: " << bs.po.abs_prob << std::endl;
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
    bs.po.abs_prob = select(sample.x() > prob, bs.po.abs_prob, bsdf_bs.po.abs_prob);
    // std::cout << "[DEBUG] __sample after select: " << bs.po.abs_prob << std::endl;

    // Joon added
    // std::cout << "sample.x() > prob " << (sample.x() > prob) << std::endl;
    bs.rgb_rv = select(sample.x() > prob, bs.rgb_rv, bsdf_bs.rgb_rv);
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
    bs.rgb_rv = full<Float<ad>>(-1.0f);
    bs.po.abs_prob = full<Float<ad>>(0.0f);
    return bs;
}


template <bool ad>
BSDFSample<ad> VaeSub::__sample_sub(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Mask<ad> active) const {
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
    BSDFSample<ad> bs;
    Float<ad> pdf_po = Frame<ad>::cos_theta(its.wi);
    Vector3f<ad> dummy;
    std::tie(bs.po, bs.rgb_rv,dummy,dummy) = __sample_sp<ad>(scene, its, sample, pdf_po, active);

    // std::tie(bs.po, bs.rgb_rv,dummy,dummy) = __sample_sp<ad>(scene, its, sample, pdf_po, active);
    // std::cout << "[DEBUG] __sample_sub" << bs.po.abs_prob << std::endl;
    bs.wo = warp::square_to_cosine_hemisphere<ad>(tail<2>(sample));
    bs.pdf = pdf_po;
    // TODO: add and check
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
    // // better diphole
    // // Spectrum<ad> D = (2.0f * sigma_a + miu_s_p) / (3.0f * (sigma_a + miu_s_p) * (sigma_a + miu_s_p));
    Spectrum<ad> D = 1.0f / (3.0f * (miu_s_p + sigma_a));

    Spectrum<ad> miu_tr = sqrt(sigma_a / D);

    // Float<ad> d = norm(its.p - bs.po.p);


    // FloatC denominator;
    // if constexpr(ad){
    //     denominator = detach(miu_tr).x() + detach(mi
    //     .u_tr).y() + detach(miu_tr).z();
    // } else {
    //     denominator = miu_tr.x() + miu_tr.y() + miu_tr.z();
    // }


    Vector3f<ad> dv = its.p - bs.po.p;
    Vector3f<ad> dLocal(dot(its.sh_frame.s, dv), dot(its.sh_frame.t, dv), dot(its.sh_frame.n, dv));
    Vector3f<ad> nLocal(dot(its.sh_frame.s, bs.po.n), dot(its.sh_frame.t, bs.po.n), dot(its.sh_frame.n, bs.po.n));
    // Vector3f<ad> nLocal(dot(its.sh_frame.s, bs.po.n), dot(its.sh_frame.t, bs.po.n), dot(its.sh_frame.n, bs.po.n));
    // // // should I use geometry normal here?

    Vector3f<ad> rProj(sqrt(dLocal.y() * dLocal.y() + dLocal.z() * dLocal.z()), 
                        sqrt(dLocal.x() * dLocal.x() + dLocal.z() * dLocal.z()),
                        sqrt(dLocal.y() * dLocal.y() + dLocal.x() * dLocal.x()));

    Float<ad> pdf = 0.0f;
    // Joon: i for RGB channel
    for (int i = 0 ; i < 3; i++){
        // pdf +=  0.25f * abs(nLocal.x()) / 3.0f;
        // pdf +=  0.25f * abs(nLocal.y()) / 3.0f;
        // pdf +=  0.5f * abs(nLocal.z()) / 3.0f;
        // pdf += __pdf_sr<ad>(miu_tr[i], rProj.x(), active) * 0.25f * abs(nLocal.x()) / 3.0f;
        // pdf += __pdf_sr<ad>(miu_tr[i], rProj.y(), active) * 0.25f * abs(nLocal.y()) / 3.0f;
        // pdf += __pdf_sr<ad>(miu_tr[i], rProj.z(), active) * 0.5f * abs(nLocal.z()) / 3.0f;
    }
    // FIXME: tmp setting
    pdf = abs(nLocal.z()) / 3.0f;
    // pdf = 1.0f;
    // pdf = 1 / 3.0f; 

    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
    Spectrum<ad> Fersnelterm = __FersnelDi<ad>(1.0f, m_eta.eval<ad>(its.uv), cos_theta_i);
    Float<ad> F = Fersnelterm.x();

    Float<ad> value = (1.0f - F) * pdf;
    // Float<ad> value = pdf;
    // Float<ad> value = (1.0f - F) * pdf / bs.po.num;

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

template <int C, int R,bool ad>
Array<Float<ad>,R> matmul(Array<Array<float,R>,C> mat, Array<Float<ad>,C> vec) {
    using EVector = Array<Float<ad>,R>;
    Array<Float<ad>,R> sum = mat.coeff(0) * EVector::full_(vec[0],1); //[64] * [1,N]
    
    for (int i=1;i<C;i++){
        sum = fmadd(mat.coeff(i),EVector::full_(vec.coeff(i),1),sum);
    }
    return sum;
}
template <bool ad>
std::pair<Vector3f<ad>,Float<ad>> VaeSub::_run(Array<Float<ad>,23> x,Array<Float<ad>,23> x2,Array<Float<ad>,4> latent) const {
        
        // auto feat = max(shared_preproc_mlp_2_shapemlp_fcn_0_weights * x + shared_preproc_mlp_2_shapemlp_fcn_0_biases,0.0f);
        // feat = max(shared_preproc_mlp_2_shapemlp_fcn_1_weights * feat + shared_preproc_mlp_2_shapemlp_fcn_1_biases,0.0f);
        // feat = max(shared_preproc_mlp_2_shapemlp_fcn_2_weights * feat + shared_preproc_mlp_2_shapemlp_fcn_2_biases,0.0f);
        // std::cout << "feat" << feat[0] << std::endl;
        // features = max(shared_preproc_mlp_2_shapemlp_fcn_1_weights * features + shared_preproc_mlp_2_shapemlp_fcn_1_biases,0.0f);
        // features = max(shared_preproc_mlp_2_shapemlp_fcn_2_weights * features + shared_preproc_mlp_2_shapemlp_fcn_2_biases,0.0f);
        auto features = max(matmul<23,64,ad>(m_shared_preproc_mlp_2_shapemlp_fcn_0_weights,x) + shared_preproc_mlp_2_shapemlp_fcn_0_biases,0.0f);
        features = max(matmul<64,64,ad>(m_shared_preproc_mlp_2_shapemlp_fcn_1_weights,features) + shared_preproc_mlp_2_shapemlp_fcn_1_biases,0.0f);
        features = max(matmul<64,64,ad>(m_shared_preproc_mlp_2_shapemlp_fcn_2_weights,features) + shared_preproc_mlp_2_shapemlp_fcn_2_biases,0.0f);
        

        // auto abs = head<32,Array<Float<ad>,64>>(max(absorption_mlp_fcn_0_weights * features + absorption_mlp_fcn_0_biases,0.0f));
        // Represents 32 dim vector although shape is 64 (invalid regions are set to zero)
        // Array<Float<ad>,64> abs = max(absorption_mlp_fcn_0_weights * features + absorption_mlp_fcn_0_biases,0.0f); // Vector 32
        // abs = absorption_dense_kernel * abs + absorption_dense_bias[0];
        // abs = abs + absorption_dense_bias[0];
        // auto absorption = NetworkHelpers::sigmoid<ad>(abs.x());
        auto features_LS = max(matmul<23,64,ad>(m_shared_preproc_mlp_2_shapemlp_fcn_0_weights,x2) + shared_preproc_mlp_2_shapemlp_fcn_0_biases,0.0f);
        features_LS = max(matmul<64,64,ad>(m_shared_preproc_mlp_2_shapemlp_fcn_1_weights,features_LS) + shared_preproc_mlp_2_shapemlp_fcn_1_biases,0.0f);
        features_LS = max(matmul<64,64,ad>(m_shared_preproc_mlp_2_shapemlp_fcn_2_weights,features_LS) + shared_preproc_mlp_2_shapemlp_fcn_2_biases,0.0f);
        Array<Float<ad>,32> abs1 = max(matmul<64,32,ad>(m_absorption_mlp_fcn_0_weights, features_LS) + m_absorption_mlp_fcn_0_biases,0.0f); // Vector 32
                                                                                                                                         //
        auto abs2 = matmul<32,1,ad>(m_absorption_dense_kernel, abs1) + m_absorption_dense_bias[0];
        auto absorption = NetworkHelpers::sigmoid<ad>(abs2.x());


        // Gaussian noise
        latent = float(M_SQRT2) * erfinv(2.f*latent - 1.f);
        auto featLatent = concat(latent,features);
        
        // auto y = head<64,Array<Float<ad>,68>>(scatter_decoder_fcn_fcn_0_weights * featLatent);
        // y = max(y + scatter_decoder_fcn_fcn_0_biases,0.0f); 
        // y = max(scatter_decoder_fcn_fcn_1_weights * y + scatter_decoder_fcn_fcn_1_biases,0.0f); 
        // y = max(scatter_decoder_fcn_fcn_2_weights * y + scatter_decoder_fcn_fcn_2_biases,0.0f); 
        // auto outPos = head<3,Array<Float<ad>,64>>(scatter_dense_2_kernel * y);
        // outPos = outPos + scatter_dense_2_bias;
        auto y2 = max(matmul<68,64,ad>(m_scatter_decoder_fcn_fcn_0_weights,featLatent)+m_scatter_decoder_fcn_fcn_0_biases,0.0f);
        y2 = max(matmul<64,64,ad>(m_scatter_decoder_fcn_fcn_1_weights,y2)+m_scatter_decoder_fcn_fcn_1_biases,0.0f);
        y2 = max(matmul<64,64,ad>(m_scatter_decoder_fcn_fcn_2_weights,y2)+m_scatter_decoder_fcn_fcn_2_biases,0.0f);
        auto outPos2 = matmul<64,3,ad>(m_scatter_dense_2_kernel,y2) + m_scatter_dense_2_bias;

        return std::pair(outPos2,absorption);
}

template <bool ad>
std::pair<Vector3f<ad>,Float<ad>> VaeSub::_run(Array<Float<ad>,23> x,Array<Float<ad>,4> latent) const {
        
        // auto feat = max(shared_preproc_mlp_2_shapemlp_fcn_0_weights * x + shared_preproc_mlp_2_shapemlp_fcn_0_biases,0.0f);
        // feat = max(shared_preproc_mlp_2_shapemlp_fcn_1_weights * feat + shared_preproc_mlp_2_shapemlp_fcn_1_biases,0.0f);
        // feat = max(shared_preproc_mlp_2_shapemlp_fcn_2_weights * feat + shared_preproc_mlp_2_shapemlp_fcn_2_biases,0.0f);
        // std::cout << "feat" << feat[0] << std::endl;
        // features = max(shared_preproc_mlp_2_shapemlp_fcn_1_weights * features + shared_preproc_mlp_2_shapemlp_fcn_1_biases,0.0f);
        // features = max(shared_preproc_mlp_2_shapemlp_fcn_2_weights * features + shared_preproc_mlp_2_shapemlp_fcn_2_biases,0.0f);
        auto features = max(matmul<23,64,ad>(m_shared_preproc_mlp_2_shapemlp_fcn_0_weights,x) + shared_preproc_mlp_2_shapemlp_fcn_0_biases,0.0f);
        features = max(matmul<64,64,ad>(m_shared_preproc_mlp_2_shapemlp_fcn_1_weights,features) + shared_preproc_mlp_2_shapemlp_fcn_1_biases,0.0f);
        features = max(matmul<64,64,ad>(m_shared_preproc_mlp_2_shapemlp_fcn_2_weights,features) + shared_preproc_mlp_2_shapemlp_fcn_2_biases,0.0f);
        

        // auto abs = head<32,Array<Float<ad>,64>>(max(absorption_mlp_fcn_0_weights * features + absorption_mlp_fcn_0_biases,0.0f));
        // Represents 32 dim vector although shape is 64 (invalid regions are set to zero)
        // Array<Float<ad>,64> abs = max(absorption_mlp_fcn_0_weights * features + absorption_mlp_fcn_0_biases,0.0f); // Vector 32
        // abs = absorption_dense_kernel * abs + absorption_dense_bias[0];
        // abs = abs + absorption_dense_bias[0];
        // auto absorption = NetworkHelpers::sigmoid<ad>(abs.x());
        Array<Float<ad>,32> abs1 = max(matmul<64,32,ad>(m_absorption_mlp_fcn_0_weights, features) + m_absorption_mlp_fcn_0_biases,0.0f); // Vector 32
        auto abs2 = matmul<32,1,ad>(m_absorption_dense_kernel, abs1) + m_absorption_dense_bias[0];
        auto absorption = NetworkHelpers::sigmoid<ad>(abs2.x());


        // Gaussian noise
        latent = float(M_SQRT2) * erfinv(2.f*latent - 1.f);
        auto featLatent = concat(latent,features);
        
        // auto y = head<64,Array<Float<ad>,68>>(scatter_decoder_fcn_fcn_0_weights * featLatent);
        // y = max(y + scatter_decoder_fcn_fcn_0_biases,0.0f); 
        // y = max(scatter_decoder_fcn_fcn_1_weights * y + scatter_decoder_fcn_fcn_1_biases,0.0f); 
        // y = max(scatter_decoder_fcn_fcn_2_weights * y + scatter_decoder_fcn_fcn_2_biases,0.0f); 
        // auto outPos = head<3,Array<Float<ad>,64>>(scatter_dense_2_kernel * y);
        // outPos = outPos + scatter_dense_2_bias;
        auto y2 = max(matmul<68,64,ad>(m_scatter_decoder_fcn_fcn_0_weights,featLatent)+m_scatter_decoder_fcn_fcn_0_biases,0.0f);
        y2 = max(matmul<64,64,ad>(m_scatter_decoder_fcn_fcn_1_weights,y2)+m_scatter_decoder_fcn_fcn_1_biases,0.0f);
        y2 = max(matmul<64,64,ad>(m_scatter_decoder_fcn_fcn_2_weights,y2)+m_scatter_decoder_fcn_fcn_2_biases,0.0f);
        auto outPos2 = matmul<64,3,ad>(m_scatter_dense_2_kernel,y2) + m_scatter_dense_2_bias;

        return std::pair(outPos2,absorption);
}
template <bool ad>
// Array<Float<ad>,size> VaeSub::_preprocessFeatures(const Intersection<ad>&its, int ch_idx, bool isPlane) const {
Array<Float<ad>,23> VaeSub::_preprocessFeatures(const Intersection<ad>&its, Float<ad> rnd, bool isPlane,
        Array<Float<ad>,3> wi, bool lightSpace
            ) const {
    float scale = 1.0f;
    Spectrum<ad> albedo = m_albedo.eval<ad>(its.uv);
    Spectrum<ad> sigma_s = scale * m_sigma_t.eval<ad>(its.uv) * albedo;
    Spectrum<ad> sigma_a = scale * m_sigma_t.eval<ad>(its.uv) - sigma_s;
    Spectrum<ad> miu_s_p = (1.0f - m_g.eval<ad>(its.uv)) * sigma_s;
    Spectrum<ad> miu_t_p = miu_s_p + sigma_a;
    Spectrum<ad> alpha_p = miu_s_p / miu_t_p;

    Spectrum<ad> g = m_g.eval<ad>(its.uv);
    Spectrum<ad> eta = m_eta.eval<ad>(its.uv);

    std::cout << "m_sigma_t" << m_sigma_t.eval<ad>(its.uv)  << std::endl;
    auto effectiveAlbedo = -log(1.0f-alpha_p * (1.0f - exp(-8.0f))) / 8.0f;
    
    auto albedoNorm = (effectiveAlbedo - m_albedoMean) * m_albedoStdInv;
    auto gNorm = (g-m_gMean) * m_gStdInv;
    auto iorNorm = 2.0f * (eta - 1.25f);

    auto shapeFeatures = zero<Array<Float<ad>,20>>();

    if(isPlane)
        shapeFeatures[3] = 1.0;
    else{
        shapeFeatures = select(rnd < (1.0f / 3.0f), its.poly_coeff.x(), its.poly_coeff.y());
        shapeFeatures = select(rnd > (2.0f / 3.0f), its.poly_coeff.z(), shapeFeatures);
    }
    
    // std::cout << "shapeFeatures[3]" << shapeFeatures[3] << std::endl;
    // std::cout << "shapeFeatures " << shapeFeatures << std::endl;

    if(lightSpace){
    // if(false){ //lightSpace){
        // TODO: Debugging: TS -> LS
        Vector<ad> s,t,n;
        n = -wi;
        onb<ad>(n, s, t);
        shapeFeatures = rotatePolynomial<3,ad>(shapeFeatures,s,t,n);
    }
    
    auto shapeFeaturesNorm = (shapeFeatures - m_shapeFeatMean)*m_shapeFeatStdInv;
    
    // auto b = concat(shapeFeaturesNorm,albedoNorm);
    // b = concat(b,gNorm);
    // b = concat(b,iorNorm);
    // Array<Float<ad>,size> x;
    using Vector23f = Array<Float<ad>,23>;
    Vector23f x = full<Vector23f>(0.0f); // = zero<Array<Float<ad>,23>>();
    for (int i=0;i<20;i++){
        x[i] = shapeFeaturesNorm[i];
    }

    // Choose RGB channel according to rnd
    x[20] = select(rnd < (1.0f / 3.0f), albedoNorm.x(),albedoNorm.y());
    x[20] = select(rnd > (2.0f / 3.0f), albedoNorm.z(),x[20]);
    x[21] = select(rnd < (1.0f / 3.0f), gNorm.x(),gNorm.y());
    x[21] = select(rnd > (2.0f / 3.0f), gNorm.z(),x[21]);
    x[22] = select(rnd < (1.0f / 3.0f), iorNorm.x(),iorNorm.y());
    x[22] = select(rnd > (2.0f / 3.0f), iorNorm.z(),x[22]);
    // std::cout << "x[20]" << x[20] << std::endl;
    // std::cout << "rnd" << rnd << std::endl;
        
    return x;
}

template <bool ad>
Float<ad> VaeSub::getKernelEps(const Intersection<ad>& its,Float<ad> rnd) const {
// Float<ad> VaeSub::getKernelEps(const Intersection<ad>& its,int idx) const {
    
    Spectrum<ad> albedo = m_albedo.eval<ad>(its.uv);
    Float<ad> albedo_ch = select(rnd < (1.0f / 3.0f), albedo.x(),albedo.y());
    albedo_ch = select(rnd > (2.0f / 3.0f), albedo.z(),albedo_ch);
    Spectrum<ad> sigma_t = m_sigma_t.eval<ad>(its.uv);
    auto sigma_t_ch = select(rnd < (1.0f / 3.0f), sigma_t.x(),sigma_t.y());
    sigma_t_ch = select(rnd > (2.0f / 3.0f), sigma_t.z(),sigma_t_ch);
    Spectrum<ad> g = m_g.eval<ad>(its.uv);
    auto g_ch = select(rnd < (1.0f / 3.0f), g.x(),g.y());
    g_ch = select(rnd > (2.0f / 3.0f), g.z(),g_ch);

    // FIXME: tmp
    // albedo_ch = albedo.x()
    // sigma_t_ch = sigma_t.x()
    // g_ch = g.x()

    auto sigma_s = sigma_t_ch * albedo_ch;
    auto sigma_a = sigma_t_ch - sigma_s;

    auto miu_s_p = (1.0f - g_ch) * sigma_s;
    auto miu_t_p = miu_s_p + sigma_a;
    auto alpha_p = miu_s_p / miu_t_p;

    auto effectiveAlbedo = -log(1.0f-alpha_p * (1.0f - exp(-8.0f))) / 8.0f;
    
    auto val = 0.25f * g_ch + 0.25f * alpha_p + 1.0f * effectiveAlbedo;
    auto res = 4.0f * val * val / (miu_t_p * miu_t_p);
    // std::cout << "res" << res << std::endl;

    return res;
}
void testRotatePolynomial() {
    const bool ad = false;
    std::cout << "#########################" << std::endl;
    std::cout << "rotatePolynomial Test" << std::endl;
    Vector<ad> s,t,n;
    // n = -wi;
    n = Vector<ad>(0.0f,1.0f,0.0f);
    onb<ad>(n, s, t);
    auto test_shapeFeatures = zero<Array<Float<ad>,20>>();
    test_shapeFeatures[3] = 1.0f;
    auto test1 = rotatePolynomial<3,ad>(test_shapeFeatures,s,t,n);
    std::cout << "Test 1: X-rotation 90 -> y" << std::endl;
    std::cout << "test1 " << test1 << std::endl;

    n = Vector<ad>(1.0f,0.0f,0.0f);
    onb<ad>(n, s, t);
    test_shapeFeatures = zero<Array<Float<ad>,20>>();
    test_shapeFeatures[3] = 1.0f;
    auto test2= rotatePolynomial<3,ad>(test_shapeFeatures,s,t,n);
    std::cout << "[Test 2]: Y-rotation 90 -> x" << std::endl;
    std::cout << "test2 " << test2 << std::endl;
    
    std::cout << "[Test 3]: Transform back to original coords" << std::endl;
    // Construct orthogonal frame
    Spectrum<ad> tangent1, tangent2;
    auto inNormal = Vector3f<ad>(0.2f,0.4f,0.5f);
    normalize(inNormal);
    onb<ad>(inNormal, tangent1, tangent2);
    Vector3f<ad> outPos( 
        Float<ad>(1.0f,0.5f,0.0f,0.2f),
        Float<ad>(0.5f,0.5f,1.0f,0.6f),
        Float<ad>(0.0f,0.0f,0.0f,0.0f)
    ); 
    
    auto Xformed = outPos.x() * tangent1 + outPos.y() * tangent2 + outPos.z() * inNormal;
    std::cout << "Xformed" << Xformed << std::endl;
    
    auto ortho = dot(inNormal,Xformed);
    std::cout << "ortho " << ortho << std::endl;
    inNormal = -inNormal;
    onb<ad>(inNormal, tangent1, tangent2);
    auto XformedBack = outPos.x() * tangent1 + outPos.y() * tangent2 + outPos.z() * inNormal;
    std::cout << "XformedBack " << XformedBack << std::endl;
    std::cout << "outPos " << outPos << std::endl;

    // PSDR_ASSERT_MSG(any(Xformed.z() == 0.0f), "[TEST 3] Failed");
}

template <bool ad>
std::tuple<Intersection<ad>,Float<ad>,Vector3f<ad>,Vector3f<ad>> VaeSub::__sample_sp(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Float<ad> &pdf, Mask<ad> active) const {        
// template <bool ad>
// std::pair<Intersection<ad>,Float<ad>> VaeSub::__sample_sp(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Float<ad> &pdf, Mask<ad> active) const {        
    
        Float<ad> rnd = sample[5];
        // rnd = full<Float<ad>>(0.00f);
        // rnd = full<Float<ad>>(0.50f);
        // rnd = full<Float<ad>>(0.99f);

        float is_plane = false; //false; 
        float is_light_space = false; //true; 

        // its.wi = -its.sh_frame.n;

        auto x = _preprocessFeatures<ad>(its,rnd,is_plane,its.wi,is_light_space);
        // auto x2 = _preprocessFeatures<ad>(its,rnd,is_plane,its.wi,true);
        auto kernelEps = getKernelEps<ad>(its,rnd);
        // std::cout << "kernelEps" << kernelEps << std::endl;
        auto fitScaleFactor = 1.0f / sqrt(kernelEps);
        
        // Network Inference: computationally heavy from here...
        rnd = sample.y();
        Array<Float<ad>,4> latent(sample[2],sample[3],sample[6],sample[7]);
        Spectrum<ad> outPos,outPos2;
        Float<ad> absorption,absorption2;
        // std::tie(outPos,absorption)= _run<ad>(x,x2,latent);
        std::tie(outPos,absorption)= _run<ad>(x,latent);
        
        // Planar Sampling as in [Deng 2022]
        Vector3f<ad> vx = its.sh_frame.s;
        Vector3f<ad> vy = its.sh_frame.t;
        Vector3f<ad> vz = its.sh_frame.n;

        rnd = sample.y();
        pdf = 1.0f;
        
        Float<ad> tmax;
        auto inNormal = vz; // -its.wi;
        auto projDir = vz;
        Float<ad> polyVal;
        // if(true) {
        if(false) {
        // if(is_plane){ // plane
            vx = select(rnd > 0.5f, its.sh_frame.t, vx);
            vy = select(rnd > 0.5f, its.sh_frame.n, vy);
            vz = select(rnd > 0.5f, its.sh_frame.s, vz);

            vx = select(rnd > 0.75f, its.sh_frame.n, vx);
            vy = select(rnd > 0.75f, its.sh_frame.s, vy);
            vz = select(rnd > 0.75f, its.sh_frame.t, vz);

            rnd = sample[5];
            Spectrum<ad> sigma_s = m_sigma_t.eval<ad>(its.uv) * m_albedo.eval<ad>(its.uv);
            Spectrum<ad> sigma_a = m_sigma_t.eval<ad>(its.uv) - sigma_s;
            Spectrum<ad> miu_s_p = (1.0f - m_g.eval<ad>(its.uv)) * sigma_s;
            Spectrum<ad> D = 1.0f / (3.0f * (sigma_a + miu_s_p));
            Spectrum<ad> miu_tr = sqrt(sigma_a / D);
            Float<ad> miu_tr_0 = 0.5f;
            miu_tr_0 = select(rnd < 1.0f / 3.0f, miu_tr.x(), miu_tr.y());
            miu_tr_0 = select(rnd > 2.0f / 3.0f, miu_tr.z(), miu_tr_0);
            // Float<ad> r = __sample_sr<ad>(miu_tr_0, sample.z());
            Vector3f<ad> dv = outPos - its.p;
            Vector3f<ad> dLocal(dot(vx,dv),dot(vy,dv),dot(vz,dv));
            Float<ad> r = sqrt(dLocal.x()*dLocal.x() + dLocal.y()*dLocal.y());
            Float<ad> rmax = __sample_sr<ad>(miu_tr_0, 0.999999f);
            Float<ad> l = select(rmax > r,2.0f * sqrt(rmax * rmax - r * r),0.01f);

            tmax = l;
            inNormal = vz;
            projDir = vz;
            // Construct orthogonal frame
            Spectrum<ad> tangent1, tangent2;
            onb<ad>(inNormal, tangent1, tangent2);

            // Local to World coordinates 
            auto inPos = its.p;
            outPos = inPos + outPos.x() * tangent1 + outPos.y() * tangent2 + outPos.z() * inNormal;
            outPos = inPos + (outPos - inPos) / fitScaleFactor;

            auto shapeFeatures = select(rnd < (1.0f / 3.0f), its.poly_coeff.x(), its.poly_coeff.y());
            shapeFeatures = select(rnd > (2.0f / 3.0f), its.poly_coeff.z(), shapeFeatures);
            
            // projDir = -evalGradient<ad>(detach(outPos),shapeFeatures,its.p,3,detach(fitScaleFactor),true,vz);
            // projDir = select(sample[2] <0.5f,projDir,vz);
        }
        else {
            rnd = sample[5];
            Spectrum<ad> sigma_s = m_sigma_t.eval<ad>(its.uv) * m_albedo.eval<ad>(its.uv);
            Spectrum<ad> sigma_a = m_sigma_t.eval<ad>(its.uv) - sigma_s;
            Spectrum<ad> miu_s_p = (1.0f - m_g.eval<ad>(its.uv)) * sigma_s;
            Spectrum<ad> D = 1.0f / (3.0f * (sigma_a + miu_s_p));
            Spectrum<ad> miu_tr = sqrt(sigma_a / D);
            Float<ad> miu_tr_0 = 0.5f;
            miu_tr_0 = select(rnd < 1.0f / 3.0f, miu_tr.x(), miu_tr.y());
            miu_tr_0 = select(rnd > 2.0f / 3.0f, miu_tr.z(), miu_tr_0);
            // TODO: implementing polyVal regularization50// 
            // Float<ad> r = __sample_sr<ad>(miu_tr_0, sample.z());
            Vector3f<ad> dv = outPos - its.p;
            Vector3f<ad> dLocal(dot(vx,dv),dot(vy,dv),dot(vz,dv));

            Float<ad> r = sqrt(dLocal.x()*dLocal.x() + dLocal.y()*dLocal.y());
            Float<ad> rmax = __sample_sr<ad>(miu_tr_0, 0.999999f);
            Float<ad> l = select(rmax > r,2.0f * sqrt(rmax * rmax - r * r),0.01f);
            // tmax = 2*kernelEps0;
            // std::cout << "tmax" <<  tmax << std::endl;
            tmax = l;
            
            if(is_light_space){
                inNormal = -its.wi;
                // inNormal = vz;
            }
            else {
                inNormal = vz;
            }
            
            auto shapeFeatures = select(rnd < (1.0f / 3.0f), its.poly_coeff.x(), its.poly_coeff.y());
            shapeFeatures = select(rnd > (2.0f / 3.0f), its.poly_coeff.z(), shapeFeatures);
            
            // if (false) { //is_plane){
            if (is_plane){
                shapeFeatures = zero<Array<Float<ad>,20>>();
                shapeFeatures[3] = 1.0;
                // shapeFeatures[4] = -1.0;
                // shapeFeatures[7] = -1.0;
            }

            // Construct orthogonal frame
            Spectrum<ad> tangent1, tangent2;
            onb<ad>(inNormal, tangent1, tangent2);

            auto inPos = its.p;
            outPos = inPos + outPos.x() * tangent1 + outPos.y() * tangent2 + outPos.z() * inNormal;
            outPos = inPos + (outPos - inPos) / fitScaleFactor;

            // NOTE: evalGradient accepts world space coordinates
            std::tie(polyVal,projDir) = evalGradient<ad>(detach(outPos),shapeFeatures,its.p,3,detach(fitScaleFactor),true,vz);
            projDir = -sign(polyVal) * projDir;
        }
        
        std::cout << "polyVal" << hmean(abs(detach(polyVal))) << std::endl;
        tmax = 2*sqrt(kernelEps);
        projDir = normalize(projDir);
        // std::cout << "projDir " << projDir << std::endl;
        Ray<ad> ray3(outPos - 0.5f*tmax*projDir,projDir, tmax);
        // TODO: implementing polyVal regularization
        // active = active*(abs(polyVal) < 50.0);
        // Ray<ad> ray3(outPos - 0.5f*tmax*projDir, projDir, tmax);
        // Ray<ad> ray2(outPos, vz, tmax);
        
        // Ray_all_intersect selects one of numIntersection samples uniformly.
        // We'll just use ray_intesect
        Intersection<ad> its3 = scene->ray_intersect<ad, ad>(ray3, active);
        // std::cout << "its3.t" << its3.t << std::endl;
        // Choose on of all intersections along ray with equal probability5
        // Intersection<ad> its3 = scene->ray_all_intersect<ad, ad>(ray3, active,sample,5);
        // std::cout << "active " << active << std::endl;

        // Intersection<ad> its3 = scene->ray_intersect<ad, ad>(Ray<ad>(outPos,projDir), active); //,sample,5);
        // Intersection<ad> its3 = scene->ray_intersect<ad, ad>(ray3, active); //,sample,5);
        its3.abs_prob = absorption;
        // if(true) {
        // // if(!is_plane){
        //     projDir = -projDir;
        //     Ray<ad> ray2(outPos - 0.5f*tmax*projDir, projDir, tmax);
        //     // Intersection<ad> its2 = scene->ray_all_intersect<ad, ad>(ray2, active,sample,5);
        //     Intersection<ad> its2 = scene->ray_intersect<ad, ad>(Ray<ad>(outPos,projDir),active);
        //     // Intersection<ad> its2 = scene->ray_intersect<ad, ad>(Ray<ad>(outPos,projDir),active);
        //     auto mask_ = its2.is_valid();
        //     float validity = 100*count(detach(mask_)) / slices(detach(mask_));
        //     std::cout << "validity " << validity << std::endl;

        //     auto msk = its2.is_valid() && (its2.t < its3.t);
        //     // std::cout << "count(msk) " << count(msk) << std::endl;
            
        //     std::cout << its2.t<< std::endl;
        //     std::cout << its3.t<<std::endl;

        //     its3.n = select(!((its2.t<its3.t) && its2.is_valid()),its3.n,its2.n);
        //     // its3.sh_frame = select(its2.is_valid(),its3.sh_frame,its2.sh_frame);
        //     its3.uv = select(!((its2.t<its3.t) && its2.is_valid()),its3.uv,its2.uv);
        //     its3.J = select(!((its2.t<its3.t) && its2.is_valid()),its3.J,its2.J);
        //     its3.num = select(!((its2.t<its3.t) && its2.is_valid()),its3.num,its2.num);
        //     its3.wi = select(!((its2.t<its3.t) && its2.is_valid()),its3.wi,its2.wi);
        //     its3.p = select(!((its2.t<its3.t) && its2.is_valid()),its3.p,its2.p);
        //     its3.t = select(!((its2.t<its3.t) && its2.is_valid()),its3.t,its2.t);
        //     its3.shape = select(!((its2.t<its3.t) && its2.is_valid()),its3.shape,its2.shape);
        // }
            
        // std::cout << "tmax " << tmax << std::endl;
        auto mask = its3.is_valid();
        float validity = 100*count(detach(mask)) / slices(detach(mask));
        std::cout << "validity " << validity << std::endl;
        // std::cout << "its3.t" << its3.t << std::endl;

        rnd = sample[5];
        // rnd = full<Float<ad>>(0.00f);
        // rnd = full<Float<ad>>(0.50f);
        // rnd = full<Float<ad>>(0.99f);

        return std::tuple<Intersection<ad>,Float<ad>,Vector3f<ad>,Vector3f<ad>>(its3,rnd,outPos,projDir);
    
} // namespace psdr

} // namespace psdr
