#pragma once

#include <string>
#include <psdr/psdr.h>
#include <enoki/array.h>

namespace psdr
{

PSDR_CLASS_DECL_BEGIN(Integrator,, Object)
public:
    virtual ~Integrator() {}

    SpectrumC renderC(const Scene &scene, int sensor_id = 0) const;
    SpectrumD renderD(const Scene &scene, int sensor_id = 0) const;
    SpectrumC renderC_shape(const Scene &scene, const IntersectionC &its, int sensor_id) const; 
    // Vector3f<false> sample_sub(const Scene &scene, Vector3f<false> pts, Vector3f<false> dir);
    // std::tuple<Vector3fC,Vector3fC,Vector3fC,Array<Array<Float<false>,20>,3>> sample_sub(const Scene &scene, Vector3f<false> pts, Vector3f<false> dir);
    // std::tuple<Vector3fC,Vector3fC,Vector3fC,Array<Array<float,20>,3>> sample_sub(const Scene &scene, Vector3f<false> pts, Vector3f<false> dir);
    std::tuple<Vector3fC,Vector3fC,Vector3fC,Vector20fC,Float<false>> sample_sub(const Scene &scene, Vector3f<false> pts, Vector3f<false> dir);
    // std::tuple<Vector3fC,Vector3fC,Vector3fC,Array<Array<float,20>,3>> sample_sub(const Scene &scene, Vector3f<false> pts, Vector3f<false> dir);
    // Vector3f<false> sample_sub(const Scene &scene, Vector3f<false> pts, Vector3f<false> dir);
    
    // NOTE: Joon added
    IntersectionC getIntersection(const Scene &scene, int sensor_id=0) const;
    // IntersectionD getIntersection(const Scene &scene, int sensor_id=0) const;

    virtual void preprocess_secondary_edges(const Scene &scene, int sensor_id, const ScalarVector4i &reso, int nrounds = 1) {}

protected:
    virtual SpectrumC Li(const Scene &scene, Sampler &sampler, const RayC &ray, MaskC active = true, int sensor_id=0) const = 0;
    virtual SpectrumD Li(const Scene &scene, Sampler &sampler, const RayD &ray, MaskD active = true, int sensor_id=0) const = 0;
    virtual SpectrumC Li_shape(const Scene &scene, Sampler &sampler, const IntersectionC &its, MaskC active = true, int sensor_id=0) const {
        return SpectrumC(0.0f);
    }
    virtual SpectrumD Li_shape(const Scene &scene, Sampler &sampler, const IntersectionD &its, MaskD active = true, int sensor_id=0) const {
        return SpectrumD(0.0f);
    }

    virtual void render_primary_edges(const Scene &scene, int sensor_id, SpectrumD &result) const;

    virtual void render_secondary_edges(const Scene &scene, int sensor_id, SpectrumD &result) const {}

    template <bool ad>
    Spectrum<ad> __render(const Scene &scene, int sensor_id) const;
    // NOTE: Joon added
    template <bool ad>
    Spectrum<ad> __render_shape(const Scene &scene, const Intersection<ad> &its, int sensor_id) const;
PSDR_CLASS_DECL_END(SamplingIntegrator)

} // namespace psdr
