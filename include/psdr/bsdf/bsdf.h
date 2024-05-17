#pragma once

#include <psdr/psdr.h>
#include <psdr/core/bitmap.h>
#include <psdr/core/intersection.h>
#include <psdr/core/records.h>
#include <psdr/scene/scene.h>

namespace psdr
{

template <typename Float_>
struct BSDFSample_ : public SampleRecord_<Float_> {
    PSDR_IMPORT_BASE(SampleRecord_<Float_>, ad, pdf, is_valid)

    Vector3f<ad> wo;
    // Xi Deng added to support bssrdf
    Intersection<ad> po;
    Mask<ad> is_sub;

    // Joon added
    Float<ad> rgb_rv; //= full<Int<ad>>(1);
    Float<ad> maxDist;
    Vector3f<ad> velocity; // for debugging autograd
    // Bitmap1fD m_alpha_u, m_alpha_v; // surface roughness
    // Bitmap1fD m_eta, m_g;
    // Bitmap3fD m_albedo, m_sigma_t; // medium features
    // Bitmap3fD m_specular_reflectance; // reflectance

    ENOKI_DERIVED_STRUCT(BSDFSample_, Base,
        ENOKI_BASE_FIELDS(pdf, is_valid),
        ENOKI_DERIVED_FIELDS(wo, po, is_sub, rgb_rv,maxDist,velocity) //m_alpha_u,m_alpha_v,m_eta,m_g,m_albedo,m_sigma_t,m_specular_reflectance)
    )
};


PSDR_CLASS_DECL_BEGIN(BSDF,, Object)
public:
    virtual ~BSDF() override {}
    BSDF() : 
    m_alpha_u(0.1f), m_alpha_v(0.1f), m_eta(0.8f), 
    m_albedo(1.0f), m_sigma_t(0.5f),
    m_specular_reflectance(1.0f), m_g(0.0f) {}

    BSDF(const Bitmap1fD&alpha, const Bitmap1fD &eta, const Bitmap3fD &reflectance, const Bitmap3fD &sigma_t, const Bitmap3fD &albedo)
        : 
    m_alpha_u(alpha), m_alpha_v(alpha), m_eta(eta), m_specular_reflectance(1.0f), 
    m_albedo(albedo), m_sigma_t(sigma_t), m_g(0.0f) {} 
        // {   
        //     m_anisotropic = false;
        // loadNetwork();
        //  }

    virtual SpectrumC eval(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const = 0;
    virtual SpectrumD eval(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const = 0;

    // virtual BSDFSampleC sample(const IntersectionC &its, const Vector3fC &sample, MaskC active = true) const = 0;
    // virtual BSDFSampleD sample(const IntersectionD &its, const Vector3fD &sample, MaskD active = true) const = 0;

    virtual FloatC pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active) const = 0;
    virtual FloatD pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active) const = 0;
    virtual bool anisotropic() const = 0;

    // Xi Deng added to support bssrdf
    virtual SpectrumC eval(const IntersectionC &its, const BSDFSampleC &bs, MaskC active = true) const = 0;
    virtual SpectrumD eval(const IntersectionD &its, const BSDFSampleD &bs, MaskD active = true) const = 0;
    // virtual SpectrumC eval(const IntersectionC &its, const BSDFSampleC &bs, MaskC active = true) const = 0;
    // virtual SpectrumD eval(const IntersectionD &its, const BSDFSampleD &bs, MaskD active = true) const = 0;

    virtual BSDFSampleC sample(const Scene *scene, const IntersectionC &its, const Vector8fC &sample, MaskC active = true) const = 0;
    // virtual std::tuple<Vector3fC,Vector3fC, Vector3fC> sample_vae(const Scene *scene, const IntersectionC &its, const Vector8fC &sample, MaskC active = true) const ;
    // virtual std::tuple<float,float, float> sample_vae(const Scene *scene, const IntersectionC &its, const Vector8fC &sample, MaskC active = true) const ;
    virtual BSDFSampleD sample(const Scene *scene, const IntersectionD &its, const Vector8fD &sample, MaskD active = true) const = 0;
    // Unused
    virtual std::pair<BSDFSampleD,FloatD> sample_v2(const Scene *scene, const IntersectionD &its, const Vector8fD &sample, MaskD active = true) const {
        BSDFSampleD sam;
        FloatD kEps;
        return std::pair(sam,kEps);
    }

    virtual FloatC pdf(const IntersectionC &its, const BSDFSampleC &wo, MaskC active) const = 0;
    virtual FloatD pdf(const IntersectionD &its, const BSDFSampleD &wo, MaskD active) const = 0;
    
    virtual FloatC pdfpoint(const IntersectionC &its, const BSDFSampleC &bs, MaskC active) const = 0;
    virtual FloatD pdfpoint(const IntersectionD &its, const BSDFSampleD &bs, MaskD active) const = 0;

    
    // template <bool ad>
    // Float<ad> getKernelEps(const Intersection<ad>& its,Float<ad> idx) const;
    virtual FloatC getKernelEps(const IntersectionC& its,FloatC rnd) const {
        return full<FloatC>(1.0f);
    }
    virtual FloatD getKernelEps(const IntersectionD& its,FloatD idx) const { 
        return full<FloatD>(1.0f);
    }
    virtual FloatC getFresnel(const IntersectionC& its) const {
        return full<FloatC>(1.0f);
    }
    virtual FloatD getFresnel(const IntersectionD& its) const { 
        return full<FloatD>(1.0f);
    }
    // virtual FloatC absorption(const IntersectionC& its,FloatC rnd) const {
    //     return full<FloatC>(1.0f);
    // }
    virtual FloatD absorption(const IntersectionD& its,Vector8fD x) const { 
        return full<FloatD>(1.0f);
    }
    virtual bool hasbssdf() const = 0;
    
    virtual Vector3fC getSigmaT(const IntersectionC& its) const { 
        return detach(m_sigma_t.m_data); //return m_sigma_t.eval<false>(its.uv);
    }
    virtual Vector3fD getSigmaT(const IntersectionD& its) const { 
        return m_sigma_t.m_data;
        // return m_sigma_t.eval<true>(its.uv);
    }
    virtual Vector3fC getAlbedo(const IntersectionC& its) const { 
        return m_albedo.eval<false>(its.uv);
    }
    virtual Vector3fD getAlbedo(const IntersectionD& its) const { 
        return m_albedo.eval<true>(its.uv);
    }

    Bitmap1fD m_alpha_u, m_alpha_v; // surface roughness
    Bitmap1fD m_eta, m_g;
    Bitmap3fD m_albedo, m_sigma_t; // medium features
    Bitmap3fD m_specular_reflectance; // reflectance

PSDR_CLASS_DECL_END(BSDF)

} // namespace psdr

ENOKI_STRUCT_SUPPORT(psdr::BSDFSample_, pdf, is_valid, wo, po, is_sub,rgb_rv,maxDist,velocity) 

ENOKI_CALL_SUPPORT_BEGIN(psdr::BSDF)
    ENOKI_CALL_SUPPORT_METHOD(eval)
    ENOKI_CALL_SUPPORT_METHOD(sample)
    ENOKI_CALL_SUPPORT_METHOD(sample_v2)
    // ENOKI_CALL_SUPPORT_METHOD(sample_vae)
    ENOKI_CALL_SUPPORT_METHOD(pdf)
    ENOKI_CALL_SUPPORT_METHOD(pdfpoint)
    ENOKI_CALL_SUPPORT_METHOD(anisotropic)
    // ENOKI_CALL_SUPPORT_METHOD(getKernelEps<true>)
    ENOKI_CALL_SUPPORT_METHOD(getKernelEps)
    ENOKI_CALL_SUPPORT_METHOD(getFresnel)
    ENOKI_CALL_SUPPORT_METHOD(getAlbedo)
    ENOKI_CALL_SUPPORT_METHOD(getSigmaT)
    ENOKI_CALL_SUPPORT_METHOD(absorption)
    ENOKI_CALL_SUPPORT_METHOD(hasbssdf)
ENOKI_CALL_SUPPORT_END(psdr::BSDF)
