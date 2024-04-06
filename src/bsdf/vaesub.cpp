#include <psdr/core/bitmap.h>
#include <psdr/core/warp.h>
#include <psdr/core/ray.h>
#include <psdr/core/sampler.h>
#include <psdr/bsdf/vaesub.h>
#include<psdr/bsdf/polynomials.h>
#include <psdr/scene/scene.h>
#include <psdr/utils.h>
#include <psdr/bsdf/ggx.h>

#include <enoki/cuda.h>
#include <enoki/autodiff.h>
#include <enoki/special.h>
#include <enoki/random.h>

// #include <sdf/sdf.hpp>
// #include <sdf/sdf.h>
#include <chrono>
#include <nvml.h>


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

        NetworkHelpers::loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_weights.bin",(float*) &m_shared_preproc_mlp_2_shapemlp_fcn_0_weights);
        NetworkHelpers::loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_weights.bin",(float*) &m_shared_preproc_mlp_2_shapemlp_fcn_1_weights);
        NetworkHelpers::loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_weights.bin",(float*) &m_shared_preproc_mlp_2_shapemlp_fcn_2_weights);
        NetworkHelpers::loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_biases.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_0_biases);
        NetworkHelpers::loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_biases.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_1_biases);
        NetworkHelpers::loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_biases.bin",(float*) &shared_preproc_mlp_2_shapemlp_fcn_2_biases);

        NetworkHelpers::loadMat(variablePath + "/absorption_dense_kernel.bin",(float*) &m_absorption_dense_kernel,1);
        NetworkHelpers::loadMat(variablePath + "/absorption_mlp_fcn_0_weights.bin",(float*) &m_absorption_mlp_fcn_0_weights,32); //[32,64]
        NetworkHelpers::loadVector(variablePath + "/absorption_dense_bias.bin", (float*) &m_absorption_dense_bias);
        NetworkHelpers::loadVector(variablePath + "/absorption_mlp_fcn_0_biases.bin", (float*) &m_absorption_mlp_fcn_0_biases);

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
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    auto res = select(bs.is_sub, __eval_sub<ad>(its, bs, active), __eval_bsdf<ad>(its, bs, active));

    auto end_time = high_resolution_clock::now();
    std::cout << "[Time] __eval" <<  duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";
    return res;
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
    // std::cout << "active " << active << std::endl;

    auto sp = full<Spectrum<ad>>(1.0f);
    if(!m_monochrome) {
        Spectrum<ad> r = zero<Spectrum<ad>>();
        Spectrum<ad> g = zero<Spectrum<ad>>();
        Spectrum<ad> b = zero<Spectrum<ad>>();
        r[0] = 1.0f;
        g[1] = 1.0f;
        b[2] = 1.0f;
        sp = select(bs.rgb_rv < (1.0f/3.0f),r,g);
        sp = select(bs.rgb_rv > (2.0f/3.0f),b,sp);
    }

    sp = sp * (1-bs.po.abs_prob);

    Spectrum<ad> sw =  __Sw<ad>(bs.po, bs.wo, active);

    Spectrum<ad> value = sp* (1.0f - F) *sw * cos_theta_o;
    if constexpr ( ad ) {
        value = value * bs.po.J;    
    }
    
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
    // NOTE: Mask active is 100% True
    BSDFSample<ad> bs = __sample_sub<ad>(scene, its, sample, active);
    BSDFSample<ad> bsdf_bs =  __sample_bsdf<ad>(its, sample, active);

    FloatC cos_theta_i = Frame<false>::cos_theta(detach(its.wi));
    SpectrumC probv = __FersnelDi<false>(1.0f, m_eta.eval<false>(detach(its.uv)), cos_theta_i);
    FloatC prob = probv.x();
    
    bs.wo = select(sample.x() > prob, bs.wo, bsdf_bs.wo);
    bs.is_sub = select(sample.x() > prob, bs.is_sub, bsdf_bs.is_sub);
    bs.pdf = select(sample.x() > prob, bs.pdf, bsdf_bs.pdf);
    bs.is_valid = select(sample.x() > prob, bs.is_valid, bsdf_bs.is_valid);

    // std::cout << "[sample] bsdf_bs.po.p " << bsdf_bs.po.p << std::endl;

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
    bs.po.poly_coeff = select(sample.x() > prob, bs.po.poly_coeff, bsdf_bs.po.poly_coeff);
    // std::cout << "[sample] bs.po.poly_coeff " << bs.po.poly_coeff << std::endl;

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
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
    BSDFSample<ad> bs;
    Float<ad> pdf_po = Frame<ad>::cos_theta(its.wi);
    Vector3f<ad> dummy;
    std::tie(bs.po, bs.rgb_rv,dummy,dummy) = __sample_sp<ad>(scene, its, sample, pdf_po, active);
    // std::cout << "[sample_sub] bs.po.p " << bs.po.p << std::endl;

    // std::tie(bs.po, bs.rgb_rv,dummy,dummy) = __sample_sp<ad>(scene, its, sample, pdf_po, active);
    // std::cout << "[DEBUG] __sample_sub" << bs.po.abs_prob << std::endl;
    bs.wo = warp::square_to_cosine_hemisphere<ad>(tail<2>(sample));
    bs.pdf = pdf_po;
    // TODO: add and check
    bs.is_valid = active && (cos_theta_i > 0.f) && bs.po.is_valid();// && ( == its.shape->m_id);&& 
    bs.is_sub = true;
    
    pdf_po = __pdf_sub<ad>(its, bs, active) * warp::square_to_cosine_hemisphere_pdf<ad>(bs.wo);
    bs.pdf = pdf_po;
    auto end_time = high_resolution_clock::now();
    std::cout << "[Time] sample_sub" <<  duration_cast<duration<double>>(end_time - start_time).count() << " seconds." <<std::endl;
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


    Float<ad> pdf = 0.0f;
    if(m_monochrome)
        pdf = 1.0f;
    else
        pdf = 1/3.0f;

    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
    Spectrum<ad> Fersnelterm = __FersnelDi<ad>(1.0f, m_eta.eval<ad>(its.uv), cos_theta_i);
    Float<ad> F = Fersnelterm.x();

    Float<ad> value = (1.0f - F) * pdf;

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
std::pair<Vector3f<ad>,Float<ad>> VaeSub::_run(Array<Float<ad>,23> x,Array<Float<ad>,4> latent) const {
        
        auto tmp = matmul<23,64,ad>(m_shared_preproc_mlp_2_shapemlp_fcn_0_weights,x);
        auto features = max(matmul<23,64,ad>(m_shared_preproc_mlp_2_shapemlp_fcn_0_weights,x) + shared_preproc_mlp_2_shapemlp_fcn_0_biases,0.0f);
        features = max(matmul<64,64,ad>(m_shared_preproc_mlp_2_shapemlp_fcn_1_weights,features) + shared_preproc_mlp_2_shapemlp_fcn_1_biases,0.0f);
        features = max(matmul<64,64,ad>(m_shared_preproc_mlp_2_shapemlp_fcn_2_weights,features) + shared_preproc_mlp_2_shapemlp_fcn_2_biases,0.0f);
        
        Array<Float<ad>,32> abs1 = max(matmul<64,32,ad>(m_absorption_mlp_fcn_0_weights, features) + m_absorption_mlp_fcn_0_biases,0.0f); // Vector 32
        auto abs2 = matmul<32,1,ad>(m_absorption_dense_kernel, abs1) + m_absorption_dense_bias[0];
        auto absorption = NetworkHelpers::sigmoid<ad>(abs2.x());

        // Gaussian noise
        latent = float(M_SQRT2) * erfinv(2.f*latent - 1.f);
        auto featLatent = concat(latent,features);
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
    std::cout << "m_albedo" << m_albedo.eval<ad>(its.uv)  << std::endl;
    
    // NOTE: Modified
    auto effectiveAlbedo = -log(1.0f-albedo * (1.0f - exp(-8.0f))) / 8.0f;
    // auto effectiveAlbedo = -log(1.0f-alpha_p * (1.0f - exp(-8.0f))) / 8.0f;
    
    auto albedoNorm = (effectiveAlbedo - m_albedoMean) * m_albedoStdInv;
    auto gNorm = (g-m_gMean) * m_gStdInv;
    auto iorNorm = 2.0f * (eta - 1.25f);

    auto shapeFeatures = zero<Array<Float<ad>,20>>();

    if(isPlane)
    {
        shapeFeatures[3] = 1.0;
    }
    else{
        shapeFeatures = select(rnd < (1.0f / 3.0f), its.poly_coeff.x(), its.poly_coeff.y());
        shapeFeatures = select(rnd > (2.0f / 3.0f), its.poly_coeff.z(), shapeFeatures);
    }
    
    // std::cout << "shapeFeatures[3]" << shapeFeatures[3] << std::endl;
    // std::cout << "shapeFeatures " << shapeFeatures << std::endl;

    if(lightSpace){
        // TODO: Debugging: TS -> LS
        Vector<ad> s,t,n;
        // n = -normalize(wi);
        n = normalize(wi);
        onb<ad>(n, s, t);
        shapeFeatures = rotatePolynomial<3,ad>(shapeFeatures,-s,-t,n);
    }
    // std::cout << "shapeFeatures " << shapeFeatures << std::endl;

    auto shapeFeaturesNorm = (shapeFeatures - m_shapeFeatMean)*m_shapeFeatStdInv;
    
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
template <bool ad>
std::tuple<Intersection<ad>,Float<ad>,Vector3f<ad>,Vector3f<ad>> VaeSub::__sample_sp(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Float<ad> &pdf, Mask<ad> active) const {        
        using namespace std::chrono;
    
        Float<ad> rnd = sample[5];

        float is_plane = true;
        // float is_plane = false;
        // FIXME: detach detection
        ///////////////////////////////
        // float is_light_space = false;
        float is_light_space = true;


        // auto start_time = high_resolution_clock::now();
        auto x = _preprocessFeatures<ad>(its,rnd,is_plane,its.wi,is_light_space);
        // auto end_time = high_resolution_clock::now();
        // std::cout << "[Time] preprocess" <<  duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";
        auto kernelEps = getKernelEps<ad>(its,rnd);
        auto fitScaleFactor = 1.0f / sqrt(kernelEps);
        
        Array<Float<ad>,4> latent(sample[2],sample[3],sample[6],sample[7]);
        Spectrum<ad> outPos;
        Float<ad> absorption;;

        // start_time = high_resolution_clock::now();
        outPos = its.p;
        absorption = 1.0f;
        if constexpr(ad){
            std::tie(outPos,absorption)= _run<ad>(x,latent);
        }
        else
            std::tie(outPos,absorption)= _run<ad>(x,latent);
        // end_time = high_resolution_clock::now();
        // std::cout << "[Time] _run" <<  duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";

        // Vector3f<ad> vx = its.sh_frame.s;
        // Vector3f<ad> vy = its.sh_frame.t;
        Vector3f<ad> vz = its.sh_frame.n;

        pdf = 1.0f;
        
        Float<ad> tmax;
        auto inNormal = vz; // -its.wi;
        auto projDir = vz;
        Float<ad> polyVal;

        rnd = sample[5];
        
        // Construct orthogonal frame
        Spectrum<ad> tangent1, tangent2;
        auto inPos = its.p;
        Frame<ad> local_frame(inNormal);
        outPos = inPos + local_frame.to_world(outPos);
        outPos = inPos + (outPos - inPos) / fitScaleFactor;

        if(is_light_space){
            inNormal = local_frame.to_world(its.wi);
        }
        else {
            inNormal = vz;
        }
        
        // std::cout << "inNormal" << inNormal << std::endl;

        auto shapeFeatures = select(rnd < (1.0f / 3.0f), its.poly_coeff.x(), its.poly_coeff.y());
        shapeFeatures = select(rnd > (2.0f / 3.0f), its.poly_coeff.z(), shapeFeatures);
        
        if (is_plane){
            shapeFeatures = zero<Array<Float<ad>,20>>();
            shapeFeatures[3] = 1.0f;
        }

        // NOTE: useLocalDir should be TRUE when using TS shape features
        // outPos should be in WORLD coords
        // FIXME: detach detection
        // TODO: Evaluate for each unique shapeFeatures instead of each input
        
        // auto st_t = high_resolution_clock::now();
        if (ad) {
            /////////////////////////////
            // polyVal = full<Float<ad>>(1.0f);
            // projDir = vz;
            /////////////////////////////
            std::tie(polyVal,projDir) = evalGradient<ad>(detach(its.p),detach(shapeFeatures),detach(outPos),3,detach(fitScaleFactor),true,vz);
        }
        else{
            std::tie(polyVal,projDir) = evalGradient<ad>(detach(its.p),detach(shapeFeatures),detach(outPos),3,detach(fitScaleFactor),true,vz);
        }
        // ///////////////////////////
        // polyVal = full<Float<ad>>(1.0f);
        // projDir = vz;
        // ///////////////////////////
        // auto end_t = high_resolution_clock::now();
        // std::cout << "[Time] evalGradient evaluation" << duration_cast<duration<double>>(end_t - st_t).count() << std::endl;
        projDir = normalize(projDir);
        projDir = -sign(polyVal) * projDir;
        
        auto rnd_var = sample[2];

        // FIXME
        // active &= (absorption < rnd_var);
        
        tmax = 1.0f;
        // tmax = 0.05f; //1.0f;
        // tmax = fitScaleFactor;
        // tmax = kernelEps;
        // std::cout << "tmax " << tmax << std::endl;

        Intersection<ad> its3;
        float eps = 0.05f;
        Mask<ad> msk;
        // std::cout << "[DEBUG] Projection count" << hsum(active) <<std::endl;
        // Spectrum<ad> sigma_s = m_sigma_t.eval<ad>(its.uv) * m_albedo.eval<ad>(its.uv);
        // Spectrum<ad> sigma_a = m_sigma_t.eval<ad>(its.uv) - sigma_s;

        // Spectrum<ad> miu_s_p = (1.0f - m_g.eval<ad>(its.uv)) * sigma_s;
        // Spectrum<ad> D = 1.0f / (3.0f * (sigma_a + miu_s_p));

        // Spectrum<ad> miu_tr = sqrt(sigma_a / D);
        // Float<ad> miu_tr_0 = 0.5f;

        // rnd = sample[5];
        // miu_tr_0 = select(rnd < 1.0f / 3.0f, miu_tr.x(), miu_tr.y());
        // miu_tr_0 = select(rnd > 2.0f / 3.0f, miu_tr.z(), miu_tr_0);
        
        // Float<ad> r = __sample_sr<ad>(miu_tr_0, sample.z());
        // Float<ad> phi = 2.0f * Pi * sample.w();
        // Float<ad> rmax = __sample_sr<ad>(miu_tr_0, 0.999999f);

        // Float<ad> l = 2.0f * sqrt(rmax * rmax - r * r);
        // tmax = l;
        // start_time = high_resolution_clock::now();
        // if(false) {
        if(true){
            // Ray<ad> ray3(detach(outPos) - eps*detach(projDir), projDir,detach(tmax)); //1.0f); //- 0.5f*tmax*projDir,projDir, tmax);
            Ray<ad> ray3(outPos- eps*projDir, projDir); //,tmax);
            Intersection<ad> its3_front = scene->ray_intersect<ad, ad>(ray3, active);

            Ray<ad> ray_back(outPos+ eps*projDir,-projDir); //tmax);
            // Ray<ad> ray_back(outPos+ eps*projDir,-projDir);
            Intersection<ad> its3_back = scene->ray_intersect<ad, ad>(ray_back,active);

            msk = !(its3_back.is_valid() && (its3_front.t > its3_back.t));
            active &= (its3_front.is_valid() | its3_back.is_valid());

            its3.n = select(msk,its3_front.n,its3_back.n);
            its3.sh_frame.s = select(msk,its3_front.sh_frame.s,its3_back.sh_frame.s);
            its3.sh_frame.t = select(msk,its3_front.sh_frame.t,its3_back.sh_frame.t);
            its3.sh_frame.n = select(msk,its3_front.sh_frame.n,its3_back.sh_frame.n);
            its3.uv = select(msk,its3_front.uv,its3_back.uv);
            its3.J = select(msk,its3_front.J,its3_back.J);
            its3.num = select(msk,its3_front.num,its3_back.num);
            its3.wi = select(msk,its3_front.wi,its3_back.wi);
            its3.p = select(msk,its3_front.p,its3_back.p);
            its3.t = select(msk,its3_front.t,its3_back.t);
            its3.shape = select(msk,its3_front.shape,its3_back.shape);
            its3.poly_coeff = shapeFeatures;
        }
        else{
            // Ray<ad> ray3(its.p + r * (vx * cos(phi) + vy * sin(phi)) - vz * 0.5f * l, vz, l);
            Ray<ad> ray3(detach(outPos) - eps*detach(projDir), projDir,detach(tmax)); //1.0f); //- 0.5f*tmax*projDir,projDir, tmax);
            its3 = scene->ray_intersect<ad, ad>(ray3, active);
            msk = true;
        }

        // end_time = high_resolution_clock::now();
        // std::cout << "[Time] projection" <<  duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";

        its3.abs_prob = absorption;
        if constexpr(ad){
            // Reparameterization to allow gradient flow
            auto ray_dir = select(msk,projDir,-projDir);
            its3.p = outPos-eps*detach(ray_dir) + detach(its3.t) * detach(ray_dir);
        }
        else{
            auto ray_dir = select(msk,projDir,-projDir);
            its3.p = outPos-eps*ray_dir + its3.t * ray_dir;
        }


        // std::cout << "[sample_sp] its.poly_coeff " << its.poly_coeff << std::endl;
            
        // auto mask = its3.is_valid();
        // float validity = 100*count(detach(mask)) / slices(detach(mask));
        // std::cout << "validity " << validity << std::endl;

        rnd = sample[5];
        // std::cout << "[sample_sp] its3.p " << its3.p << std::endl;

        return std::tuple<Intersection<ad>,Float<ad>,Vector3f<ad>,Vector3f<ad>>(its3,rnd,outPos,projDir);
    
} // namespace psdr

} // namespace psdr
