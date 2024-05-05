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
    // Bitmap1fD m_alpha_u, m_alpha_v; // surface roughness
    // Bitmap1fD m_eta, m_g;
    // Bitmap3fD m_albedo, m_sigma_t; // medium features
    // Bitmap3fD m_specular_reflectance; // reflectance

    ENOKI_DERIVED_STRUCT(BSDFSample_, Base,
        ENOKI_BASE_FIELDS(pdf, is_valid),
        ENOKI_DERIVED_FIELDS(wo, po, is_sub, rgb_rv) //m_alpha_u,m_alpha_v,m_eta,m_g,m_albedo,m_sigma_t,m_specular_reflectance)
    )
};


PSDR_CLASS_DECL_BEGIN(BSDF,, Object)
public:
    virtual ~BSDF() override {}

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
    // virtual FloatC absorption(const IntersectionC& its,FloatC rnd) const {
    //     return full<FloatC>(1.0f);
    // }
    virtual FloatD absorption(const IntersectionD& its,Vector8fD x) const { 
        return full<FloatD>(1.0f);
    }
    virtual bool hasbssdf() const = 0;
PSDR_CLASS_DECL_END(BSDF)

} // namespace psdr

ENOKI_STRUCT_SUPPORT(psdr::BSDFSample_, pdf, is_valid, wo, po, is_sub,rgb_rv) //,m_alpha_u,m_alpha_v,m_eta,m_g,m_albedo,m_sigma_t,m_specular_reflectance)

ENOKI_CALL_SUPPORT_BEGIN(psdr::BSDF)
    ENOKI_CALL_SUPPORT_METHOD(eval)
    ENOKI_CALL_SUPPORT_METHOD(sample)
    // ENOKI_CALL_SUPPORT_METHOD(sample_vae)
    ENOKI_CALL_SUPPORT_METHOD(pdf)
    ENOKI_CALL_SUPPORT_METHOD(pdfpoint)
    ENOKI_CALL_SUPPORT_METHOD(anisotropic)
    // ENOKI_CALL_SUPPORT_METHOD(getKernelEps<true>)
    ENOKI_CALL_SUPPORT_METHOD(getKernelEps)
    ENOKI_CALL_SUPPORT_METHOD(absorption)
    ENOKI_CALL_SUPPORT_METHOD(hasbssdf)
ENOKI_CALL_SUPPORT_END(psdr::BSDF)
