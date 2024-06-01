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
    // return __pdf_sub<false>(its,bs,active) & active;

    FloatC cos_theta_i = Frame<false>::cos_theta(detach(its.wi));
    SpectrumC Fersnelterm = __FersnelDi<false>(1.0f, m_eta.eval<false>(detach(its.uv)), cos_theta_i);
    FloatC F = Fersnelterm.x();
    FloatC value = select(bs.is_sub,  __pdf_sub<false>(its, bs, active), F);
    return value & active;
}

FloatD VaeSub::pdfpoint(const IntersectionD &its, const BSDFSampleD &bs, MaskD active) const {
    return __pdf_sub<true>(its,bs,active) & active;

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
    // std::cout << "hsum(hsum(its.uv)) " << hsum(hsum(its.uv)) << std::endl;
    // std::cout << "hsum(hsum(c)) " << hsum(hsum(c)) << std::endl;
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
    // return __eval_sub<ad>(its,bs,active);
    auto res = select(bs.is_sub, __eval_sub<ad>(its, bs, active), __eval_bsdf<ad>(its, bs, active));
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
    // std::cout << "bs.wo " << bs.wo << std::endl;
    // std::cout << "its.wi " << its.wi << std::endl;
    // std::cout << "ends bsdf " << std::endl;
    
    Spectrum<ad> specular_reflectance = m_specular_reflectance.eval<ad>(its.uv);

    return (result * specular_reflectance * cos_theta_o) & active;
}

template <bool ad>
Spectrum<ad> VaeSub::__eval_sub(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const {
    // Incident angle of camera ray 

    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
    // Outgoing angle of light ray
    Float<ad> cos_theta_o = Frame<ad>::cos_theta(bs.wo);
    active &= (cos_theta_i > 0.f);
    active &= (cos_theta_o > 0.f);
    // std::cout << "hsum(cos_theta_o) " << hsum(cos_theta_o) << std::endl;
    // std::cout << "count(hsum(bs.wo) != 0.0f) " << count(hsum(bs.wo) != 0.0f) << std::endl;

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

    // std::cout << "count(active) " << count(active) << std::endl;
    // std::cout << "hsum(hsum(sw)) " << hsum(hsum(sw)) << std::endl;
    
    // ============================= 
    // No Fresnel
    // Spectrum<ad> value = sp*sw * cos_theta_o;
    Spectrum<ad> value = sp* (1.0f - F) *sw * cos_theta_o;
    // ============================= 
    // std::cout << "1-F " << 1-F << std::endl;
    // std::cout << "hmean(F) " << hmean(F) << std::endl;
    // Spectrum<ad> value = sp* sw * cos_theta_o;
    // std::cout << "cos_theta_o " << cos_theta_o << std::endl;
    // std::cout << "sw " << sw << std::endl;
    // std::cout << "sp " << sp << std::endl;
    // std::cout << "F " << F << std::endl;
    // std::cout << "bs.po.abs_prob " << bs.po.abs_prob << std::endl;
    // std::cout << "bs.po.J " << bs.po.J << std::endl;
    // std::cout << "active " << active << std::endl;
    
    // if constexpr ( ad ) {
    //     value = value * bs.po.J;    
    // }
    // std::cout << "hsum(hsum(sw) " << hsum(hsum(sw)) << std::endl;
    // std::cout << "hsum(hsum(value) " << hsum(hsum(value)) << std::endl;
    std::cout << "count(!isnan(value)) " << count((~isnan(hsum(value))) & (hsum(value) != 0.0f))<< std::endl;
    
    
    return value & active;
}
// 
BSDFSampleC VaeSub::sample(const Scene *scene, const IntersectionC &its, const Vector8fC &rand, MaskC active) const {
    return __sample<false>(scene, its, rand, active);
}

BSDFSampleD VaeSub::sample(const Scene *scene, const IntersectionD &its, const Vector8fD &rand, MaskD active) const {
    return __sample<true>(scene, its,rand, active);
}

std::pair<BSDFSampleD,FloatD> VaeSub::sample_v2(const Scene *scene, const IntersectionD &its, const Vector8fD &rand, MaskD active) const {
    auto kernelEps = getKernelEps<true>(its,rand[5]);
    auto result = __sample<true>(scene, its,rand, active);
    // return std::tie(result,kernelEps);
    return std::pair(result,kernelEps); //outPos2,absorption);
}

template <bool ad>
BSDFSample<ad> VaeSub::__sample(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Mask<ad> active) const {
    // NOTE: Mask active is 100% True
    
    BSDFSample<ad> bs = __sample_sub<ad>(scene, its, sample, active);
    // Joon added; 100% BSSRDF samples
    // return bs;
    BSDFSample<ad> bsdf_bs =  __sample_bsdf<ad>(its, sample, active);

    FloatC cos_theta_i = Frame<false>::cos_theta(detach(its.wi));
    SpectrumC probv = __FersnelDi<false>(1.0f, m_eta.eval<false>(detach(its.uv)), cos_theta_i);
    FloatC prob = probv.x();
    // std::cout << "prob " << prob << std::endl;
    
    bs.wo = select(sample.x() > prob, bs.wo, bsdf_bs.wo);
    bs.is_sub = select(sample.x() > prob, bs.is_sub, bsdf_bs.is_sub);
    bs.pdf = select(sample.x() > prob, bs.pdf, bsdf_bs.pdf);
    bs.is_valid = select(sample.x() > prob, bs.is_valid, bsdf_bs.is_valid);

    // std::cout << "count(bs.is_sub) " << count(bs.is_sub) << std::endl;
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
    bs.maxDist = select(sample.x() > prob, bs.maxDist, bsdf_bs.maxDist);
    bs.velocity = select(sample.x() > prob, bs.velocity, bsdf_bs.velocity);

    if constexpr(ad){

        bs.pair.num = select(sample.x() > prob, bs.pair.num, bsdf_bs.pair.num);
        bs.pair.n = select(sample.x() > prob, bs.pair.n, bsdf_bs.pair.n);
        bs.pair.p = select(sample.x() > prob, bs.pair.p, bsdf_bs.pair.p);
        bs.pair.J = select(sample.x() > prob, bs.pair.J, bsdf_bs.pair.J);
        bs.pair.uv = select(sample.x() > prob, bs.pair.uv, bsdf_bs.pair.uv);
        bs.pair.shape = select(sample.x() > prob, bs.pair.shape, bsdf_bs.pair.shape);
        bs.pair.t = select(sample.x() > prob, bs.pair.t, bsdf_bs.pair.t);
        bs.pair.wi = select(sample.x() > prob, bs.pair.wi, bsdf_bs.pair.wi);
        bs.pair.sh_frame.s = select(sample.x() > prob, bs.pair.sh_frame.s, bsdf_bs.pair.sh_frame.s);
        bs.pair.sh_frame.t = select(sample.x() > prob, bs.pair.sh_frame.t, bsdf_bs.pair.sh_frame.t);
        bs.pair.sh_frame.n = select(sample.x() > prob, bs.pair.sh_frame.n, bsdf_bs.pair.sh_frame.n);
        bs.pair.abs_prob = select(sample.x() > prob, bs.pair.abs_prob, bsdf_bs.pair.abs_prob);
        bs.pair.poly_coeff = select(sample.x() > prob, bs.pair.poly_coeff, bsdf_bs.pair.poly_coeff);
       
        bs.pair2.num = select(sample.x() > prob, bs.pair2.num, bsdf_bs.pair2.num);
        bs.pair2.n = select(sample.x() > prob, bs.pair2.n, bsdf_bs.pair2.n);
        bs.pair2.p = select(sample.x() > prob, bs.pair2.p, bsdf_bs.pair2.p);
        bs.pair2.J = select(sample.x() > prob, bs.pair2.J, bsdf_bs.pair2.J);
        bs.pair2.uv = select(sample.x() > prob, bs.pair2.uv, bsdf_bs.pair2.uv);
        bs.pair2.shape = select(sample.x() > prob, bs.pair2.shape, bsdf_bs.pair2.shape);
        bs.pair2.t = select(sample.x() > prob, bs.pair2.t, bsdf_bs.pair2.t);
        bs.pair2.wi = select(sample.x() > prob, bs.pair2.wi, bsdf_bs.pair2.wi);
        bs.pair2.sh_frame.s = select(sample.x() > prob, bs.pair2.sh_frame.s, bsdf_bs.pair2.sh_frame.s);
        bs.pair2.sh_frame.t = select(sample.x() > prob, bs.pair2.sh_frame.t, bsdf_bs.pair2.sh_frame.t);
        bs.pair2.sh_frame.n = select(sample.x() > prob, bs.pair2.sh_frame.n, bsdf_bs.pair2.sh_frame.n);
        bs.pair2.abs_prob = select(sample.x() > prob, bs.pair2.abs_prob, bsdf_bs.pair2.abs_prob);
        bs.pair2.poly_coeff = select(sample.x() > prob, bs.pair2.poly_coeff, bsdf_bs.pair2.poly_coeff);
    }
    else{
        bs.pair = bs.po;
        bs.pair2 = bs.po;
    }

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
    bs.maxDist = full<Float<ad>>(-1.0f);
    bs.velocity = full<Vector3f<ad>>(0.0f);
    bs.pair = its;
    bs.pair2 = its;
    bs.po.abs_prob = full<Float<ad>>(0.0f);
    return bs;
}


template <bool ad>
BSDFSample<ad> VaeSub::__sample_sub(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Mask<ad> active) const {
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
    BSDFSample<ad> bs;
    Float<ad> pdf_po = Frame<ad>::cos_theta(its.wi);
    Vector3f<ad> dummy;
    std::tie(bs.po, bs.rgb_rv,dummy,bs.velocity,bs.maxDist,bs.pair,bs.pair2) = __sample_sp<ad>(scene, its, sample, pdf_po, active);
    bs.wo = warp::square_to_cosine_hemisphere<ad>(tail<2>(sample));
    bs.pdf = pdf_po;
    bs.is_valid = active && (cos_theta_i > 0.f) && bs.po.is_valid();// && ( == its.shape->m_id);&& 
    bs.is_sub = true;
    
    // Joon addedThis won't be used with point lights
    // Use pdfpoint instead
    // bs.pdf = __pdf_sub<ad>(its, bs, active) * warp::square_to_cosine_hemisphere_pdf<ad>(bs.wo);
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
    // [var61] no fresnel
    return pdf;

    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
    Spectrum<ad> Fersnelterm = __FersnelDi<ad>(1.0f, m_eta.eval<ad>(its.uv), cos_theta_i);
    Float<ad> F = Fersnelterm.x();

    // if constexpr(!ad){

    // std::cout << "pdf_sub F term hmax" << hmax(F) << std::endl;
    // std::cout << "pdf_sub F term hmin" << hmin(F) << std::endl;
    // }
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
    Spectrum<ad> g = m_g.eval<ad>(its.uv);
    Spectrum<ad> eta = m_eta.eval<ad>(its.uv);
    Spectrum<ad> sigma_s = scale * m_sigma_t.eval<ad>(its.uv) * albedo;
    Spectrum<ad> sigma_a = scale * m_sigma_t.eval<ad>(its.uv) - sigma_s;
    Spectrum<ad> miu_s_p = (1.0f - g) * sigma_s;
    Spectrum<ad> miu_t_p = miu_s_p + sigma_a;
    Spectrum<ad> alpha_p = miu_s_p / miu_t_p;
    auto sigma_t = m_sigma_t.eval<ad>(its.uv);
    auto sigma_t_ch = select(rnd < (1.0f / 3.0f), sigma_t.x(), sigma_t.y());
    sigma_t_ch = select(rnd > (2.0f / 3.0f),sigma_t.z(),sigma_t_ch);

    std::cout << "m_sigma_t" << m_sigma_t.eval<ad>(its.uv)  << std::endl;
    std::cout << "m_albedo" << m_albedo.eval<ad>(its.uv)  << std::endl;
    // std::cout << "g " << g << std::endl;
    
    // NOTE: Modified
    // auto effectiveAlbedo = -log(1.0f-albedo * (1.0f - exp(-8.0f))) / 8.0f;
    auto effectiveAlbedo = -log(1.0f-alpha_p * (1.0f - exp(-8.0f))) / 8.0f;
    // std::cout << "alpha_p " << alpha_p << std::endl;
    
    auto albedoNorm = (effectiveAlbedo - m_albedoMean) * m_albedoStdInv;
    auto gNorm = (g-m_gMean) * m_gStdInv;
    auto iorNorm = 2.0f * (eta - 1.25f);

    auto shapeFeatures = zero<Array<Float<ad>,20>>();

    if(isPlane)
    {
        shapeFeatures[3] = 1.0;
    }
    else{
        auto shape_coeff = its.poly_coeff;
        shapeFeatures = select(rnd < (1.0f / 3.0f), shape_coeff.x(), shape_coeff.y());
        shapeFeatures = select(rnd > (2.0f / 3.0f), shape_coeff.z(), shapeFeatures);
    }

    if(lightSpace){
        Vector<ad> s,t,n;
        n = normalize(wi);
        onb<ad>(n, s, t);
        shapeFeatures = rotatePolynomial<3,ad>(shapeFeatures,-s,-t,n);
    }
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

FloatD VaeSub::getKernelEps(const IntersectionD& its,FloatD idx) const
{
    return getKernelEps<true>(its,idx);
}

FloatC VaeSub::getKernelEps(const IntersectionC& its,FloatC idx) const
{
auto res = getKernelEps<false>(its,idx);
return res;
}
template <bool ad>
Float<ad> VaeSub::getKernelEps(const Intersection<ad>& its,Float<ad> rnd) const {
    // std::cout << "rnd" << rnd <<std::endl;
    
    Spectrum<ad> albedo = m_albedo.eval<ad>(its.uv);
    Float<ad> albedo_ch = select(rnd < (1.0f / 3.0f), albedo.x(),albedo.y());
    albedo_ch = select(rnd > (2.0f / 3.0f), albedo.z(),albedo_ch);
    Spectrum<ad> sigma_t = m_sigma_t.eval<ad>(its.uv);
    // if constexpr(ad){
    //     set_requires_gradient(sigma_t.x()); //sigma_t_ch);
    //     // set_requires_gradient(albedo.x()); //sigma_t_ch);
    // }
    auto sigma_t_ch = select(rnd < (1.0f / 3.0f), sigma_t.x(),sigma_t.y());
    sigma_t_ch = select(rnd > (2.0f / 3.0f), sigma_t.z(),sigma_t_ch);
    Spectrum<ad> g = m_g.eval<ad>(its.uv);
    auto g_ch = select(rnd < (1.0f / 3.0f), g.x(),g.y());
    g_ch = select(rnd > (2.0f / 3.0f), g.z(),g_ch);
    // std::cout << "albedo_ch" << albedo_ch <<std::endl;
    // std::cout << "mean albedo_ch" << hmean(albedo_ch) <<std::endl;
    // std::cout << "sigma_t_ch" << sigma_t_ch <<std::endl;
    // std::cout << "mean sigma_t_ch" << hmean(sigma_t_ch) <<std::endl;
    // std::cout << "g_ch" << g_ch <<std::endl;
    // std::cout << "mean g_ch" << hmean(g_ch) <<std::endl;


    Float<ad> sigma_s = sigma_t_ch * albedo_ch;
    auto sigma_a = sigma_t_ch - sigma_s;

    auto miu_s_p = (1.0f - g_ch) * sigma_s;
    auto miu_t_p = miu_s_p + sigma_a;
    auto alpha_p = miu_s_p / miu_t_p;

    auto effectiveAlbedo = -log(1.0f-alpha_p * (1.0f - exp(-8.0f))) / 8.0f;
    auto val = 0.25f * g_ch + 0.25f * alpha_p + 1.0f * effectiveAlbedo;

    Float<ad> res = 4.0f * val * val / (miu_t_p * miu_t_p);
    // if constexpr(ad){
    // backward(res);
    // std::cout << "gradient(sigma_t_ch) " << gradient(sigma_t.x()) << std::endl;
    // // std::cout << "gradient(sigma_t_ch) " << gradient(albedo.x()) << std::endl;
    // }
    
    return res;
}

template <bool ad>
Float<ad> VaeSub::getFitScaleFactor(const Intersection<ad> &its, Float<ad> sigma_t_ch, Float<ad> albedo_ch, Float<ad> g_ch) const
    {
        float scaleFactor = 1.0f;
        // float scaleFactor = 4.0f;
        // Spectrum<ad> albedo = m_albedo.eval<ad>(its.uv);
        // Float<ad> albedo_ch = select(rnd < (1.0f / 3.0f), albedo.x(),albedo.y());
        // albedo_ch = select(rnd > (2.0f / 3.0f), albedo.z(),albedo_ch);
        // Spectrum<ad> sigma_t = m_sigma_t.eval<ad>(its.uv);
        // auto sigma_t_ch = select(rnd < (1.0f / 3.0f), sigma_t.x(),sigma_t.y());
        // sigma_t_ch = select(rnd > (2.0f / 3.0f), sigma_t.z(),sigma_t_ch);
        // Spectrum<ad> g = m_g.eval<ad>(its.uv);
        // auto g_ch = select(rnd < (1.0f / 3.0f), g.x(),g.y());
        // g_ch = select(rnd > (2.0f / 3.0f), g.z(),g_ch);

        Float<ad> sigma_s = sigma_t_ch * albedo_ch;
        auto sigma_a = sigma_t_ch - sigma_s;

        auto miu_s_p = (1.0f - g_ch) * sigma_s;
        auto miu_t_p = miu_s_p + sigma_a;
        auto alpha_p = miu_s_p / miu_t_p;

        auto effectiveAlbedo = -log(1.0f-alpha_p * (1.0f - exp(-8.0f))) / 8.0f;
        auto val = 0.25f * g_ch + 0.25f * alpha_p + 1.0f * effectiveAlbedo;

        Float<ad> kernelEps = scaleFactor*4.0f * val * val / (miu_t_p * miu_t_p);
        return 1.0f/sqrt(kernelEps);
    }

template<bool ad>
Intersection<ad> VaeSub::projectPointToSurface(const Scene* scene, const Intersection<ad> &its, Vector3f<ad> outPos, Float<ad> fitScaleFactor, Vectorf<20,ad> shapeFeatures,Vector3f<ad> projDir, Mask<ad> active,  bool isPlane) const{

    // Projection algorithm parameters
    float eps = 0.05f;
    float maxDistScale = 10.0f;
    // float eps = 0.01f;
    // float maxDistScale = 2.0f;
    // float maxDistScale = 10.0f;

    Intersection<ad> its3;
    Float<ad> polyVal;
    auto vz = its.sh_frame.n;
        
    auto maxDist = maxDistScale/fitScaleFactor + eps;

    Ray<ad> ray3(outPos- eps*projDir, projDir,maxDist); //,tmax);
    Intersection<ad> its3_front = scene->ray_intersect<ad, ad>(ray3, active);
    its3 = its3_front;
    return its3;
    
    Ray<ad> ray_back(outPos+ eps*projDir,-projDir,maxDist); //tmax);
    Intersection<ad> its3_back = scene->ray_intersect<ad, ad>(ray_back,active);
    auto msk = !(its3_back.is_valid() && (its3_front.t > its3_back.t)); 

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

    return its3;
}
template <bool ad>
VaeSub::RetTypeSampleSp<ad> VaeSub::__sample_sp(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Float<ad> &pdf, Mask<ad> active) const {        
        using namespace std::chrono;
        Float<ad> rnd = sample[5];
        if(scene->m_opts.rgb != 0){
            rnd = rnd * 0.0f + (scene->m_opts.rgb / 3.0f) - 0.1f;
            std::cout << "Testing RGB single channel" << rnd << std::endl;
        }

        float is_plane = false;
        float is_light_space = true;
        // float is_plane = true;
        // float is_light_space = true; //false;

        auto x = _preprocessFeatures<ad>(its,rnd,is_plane,its.wi,is_light_space);
        Spectrum<ad> albedo = m_albedo.eval<ad>(its.uv);
        Float<ad> albedo_ch = select(rnd < (1.0f / 3.0f), albedo.x(),albedo.y());
        albedo_ch = select(rnd > (2.0f / 3.0f), albedo.z(),albedo_ch);
        Spectrum<ad> sigma_t = m_sigma_t.eval<ad>(its.uv);
        auto sigma_t_ch = select(rnd < (1.0f / 3.0f), sigma_t.x(),sigma_t.y());
        sigma_t_ch = select(rnd > (2.0f / 3.0f), sigma_t.z(),sigma_t_ch);
        Spectrum<ad> g = m_g.eval<ad>(its.uv);
        auto g_ch = select(rnd < (1.0f / 3.0f), g.x(),g.y());
        g_ch = select(rnd > (2.0f / 3.0f), g.z(),g_ch);

        // Float<ad> sigma_s = sigma_t_ch * albedo_ch;
        // auto sigma_a = sigma_t_ch - sigma_s;

        // auto miu_s_p = (1.0f - g_ch) * sigma_s;
        // auto miu_t_p = miu_s_p + sigma_a;
        // auto alpha_p = miu_s_p / miu_t_p;

        // auto effectiveAlbedo = -log(1.0f-alpha_p * (1.0f - exp(-8.0f))) / 8.0f;
        // auto val = 0.25f * g_ch + 0.25f * alpha_p + 1.0f * effectiveAlbedo;

        // Float<ad> kernelEps = 4.0f * val * val / (miu_t_p * miu_t_p);
        // auto kernelEps = getKernelEps<ad>(its,rnd);
        // auto fitScaleFactor = 1.0f / sqrt(kernelEps);
        auto fitScaleFactor = getFitScaleFactor<ad>(its,sigma_t_ch,albedo_ch,g_ch);
        
        Array<Float<ad>,4> latent(sample[2],sample[3],sample[6],sample[7]);
        Spectrum<ad> outPos,outPosVae;
        Float<ad> absorption;;
        std::tie(outPosVae,absorption)= _run<ad>(x,latent);
    
        Vector3f<ad> vz = its.sh_frame.n;
        pdf = 1.0f;
        
        Float<ad> tmax;
        auto inNormal = vz; // -its.wi;
        // auto projDir = vz;
        
        // Construct orthogonal frame
        Spectrum<ad> tangent1, tangent2;
        auto inPos = its.p;
        Frame<ad> local_frame(inNormal);

        outPos = inPos + local_frame.to_world(outPosVae);
        outPos = inPos + (outPos - inPos) / fitScaleFactor;
        
        
        // std::ofstream outfile0("mate_.txt");
        // auto tmp0 = outPos[1];
        // for (int i=0;i<slices(outPos);i++){
        //     outfile0 << tmp0[i] << std::endl;;
        // }
        // outfile0.close();
        // std::ofstream outfile1("mate_plus.txt");
        // auto tmp1 = matePos[1];
        // for (int i=0;i<slices(outPos);i++){
        //     outfile1 << tmp1[i] << std::endl;;
        // }
        // outfile1.close();
        // std::ofstream outfile2("mate_minus.txt");
        // auto tmp2 = matePos2[1];
        // for (int i=0;i<slices(outPos);i++){
        //     outfile2 << tmp2[i] << std::endl;;
        // }
        // outfile2.close();


        if(is_light_space)
            inNormal = local_frame.to_world(its.wi);
        else 
            inNormal = vz;

        auto shapeFeatures = select(rnd < (1.0f / 3.0f), its.poly_coeff.x(), its.poly_coeff.y());
        shapeFeatures = select(rnd > (2.0f / 3.0f), its.poly_coeff.z(), shapeFeatures);
        if (is_plane){
            shapeFeatures = zero<Array<Float<ad>,20>>();
            shapeFeatures[3] = 1.0f;
        }

        
        // if (ad)
        //     std::tie(polyVal,projDir) = evalGradient<ad>(its.p,shapeFeatures,outPos,3,detach(fitScaleFactor),true,vz);
        // else
        //     std::tie(polyVal,projDir) = evalGradient<ad>(detach(its.p),detach(shapeFeatures),detach(outPos),3,detach(fitScaleFactor),true,vz);

        
        // projDir = normalize(detach(projDir));
        // projDir = -sign(polyVal) * projDir;
        
        Vector3f<ad> projDir = -vz;
        Float<ad> polyVal;
        if (!is_plane){
            std::tie(polyVal,projDir) = evalGradient<ad>(its.p,shapeFeatures,outPos,3,fitScaleFactor,true,vz);
            projDir = normalize(detach(projDir));
            projDir = -sign(polyVal) * projDir;
        }

        Intersection<ad> its3 = projectPointToSurface<ad>(scene,its,outPos,fitScaleFactor,shapeFeatures,projDir,active,is_plane);
        its3.abs_prob = absorption;
        
        if constexpr(ad){
            // // Idea #3. Tangential Direction (w.o. Normalization)
            // auto relVector = detach(its3.p) - detach(its.p);
            // auto s = cross(Vector3fD(relVector),its3.n);
            // auto t = cross(its3.n,s); 
            // its3.p = t / fitScaleFactor - t/detach(fitScaleFactor) + detach(its3.p);

            // Idea #4. Tangential Direction (w. Normalization)
            // auto relVector = detach(its3.p) - detach(its.p);
            // auto s = cross(Vector3fD(relVector),its3.n);
            // auto t = normalize(cross(its3.n,s)); 
            // its3.p = t/fitScaleFactor - t/detach(fitScaleFactor) + detach(its3.p);

            // // Idea #5. Tangential direction. cosine weighted
            // auto relVector = detach(its3.p) - detach(its.p);
            // auto s = cross(Vector3fD(relVector),its3.n);
            // auto t = normalize(cross(its3.n,s)); 
            // its3.p = t/fitScaleFactor - t/detach(fitScaleFactor) + detach(its3.p);
        }

        Vector3f<ad> velocity(0.0f); 
        if constexpr(ad){
            // // Idea #3. Tangential velocity, w. Normalization
            // auto s = cross(Vector3fD(relVector),its3.n);
            // auto t = cross(its3.n,s); 
            // t = normalize(t);
            // velocity =(-1.0f/detach(fitScaleFactor)/detach(fitScaleFactor)) * t * dfit;
            
            // Idea#4. var224
            // auto relVector = detach(its3.p) - detach(its.p);
            // auto s = cross(Vector3fD(relVector),its3.n);
            // auto t = normalize(cross(its3.n,s)); 
            // auto tmp = dot(t,relVector) * detach(fitScaleFactor);
            // velocity = Vector3fD(detach(its3.p)) + tmp/fitScaleFactor - tmp/detach(fitScaleFactor); 
            velocity = its3.p;
        }
        Intersection<ad> its3M, its3M2;
        if constexpr(ad){
            float epsM = 5.0f;
            auto fitScaleFactor1 = getFitScaleFactor<ad>(its,sigma_t_ch+epsM,albedo_ch,g_ch);
            auto fitScaleFactor2 = getFitScaleFactor<ad>(its,sigma_t_ch-epsM,albedo_ch,g_ch);

            auto matepos = inPos + local_frame.to_world(outPosVae);
            matepos = inPos + (matepos - inPos) / fitScaleFactor1;
            auto matepos2 = inPos + local_frame.to_world(outPosVae);
            matepos2 = inPos + (matepos2 - inPos) / fitScaleFactor2;
            Vector3fC matePos = detach(matepos);
            Vector3fC matePos2 = detach(matepos2);
            its3M = projectPointToSurface<ad>(scene,its,matePos,fitScaleFactor1,shapeFeatures,projDir,active,is_plane);
            its3M2 = projectPointToSurface<ad>(scene,its,matePos2,fitScaleFactor2,shapeFeatures,projDir,active,is_plane);
        }
        else{
            its3M = its3;
            its3M2 = its3;
        }
        // Spectrum<ad> sigma_t = m_sigma_t.eval<ad>(its.uv);
        // auto _sigma_t_ch = select(rnd < (1.0f / 3.0f), sigma_t.x(),sigma_t.y());
        // _sigma_t_ch = select(rnd > (2.0f / 3.0f), sigma_t.z(),_sigma_t_ch);
        return RetTypeSampleSp<ad>(its3,rnd,outPos,velocity,sigma_t_ch,its3M,its3M2);
        // return RetTypeSampleSp<ad>(its3,sample[5],outPos,velocity,sigma_t[2],its3M,its3M2);
}
    
} // namespace psdr