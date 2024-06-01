#pragma once

#include "integrator.h"

namespace psdr
{

PSDR_CLASS_DECL_BEGIN(DirectIntegrator, final, Integrator)
public:
    DirectIntegrator(int bsdf_samples = 1, int light_samples = 1);
    virtual ~DirectIntegrator();

    void preprocess_secondary_edges(const Scene &scene, int sensor_id, const ScalarVector4i &reso, int nrounds = 1) override;

    bool m_hide_emitters = false;

protected:
    SpectrumC Li(const Scene &scene, Sampler &sampler, const RayC &ray, MaskC active = true, int sensor_id=0) const override;
    SpectrumD Li(const Scene &scene, Sampler &sampler, const RayD &ray, MaskD active = true, int sensor_id=0) const override;

    // NOTE: Joon added
    SpectrumC Li_shape(const Scene &scene, Sampler &sampler, const IntersectionC&its, MaskC active, int sensor_id) const override;
    SpectrumD Li_shape(const Scene &scene, Sampler &sampler, const IntersectionD&its, MaskD active, int sensor_id) const override;
    template <bool ad>
    Spectrum<ad> __Li_shape(const Scene &scene, Sampler &sampler, const Intersection<ad> &its, Mask<ad> active) const;
    
    IntC _compress(IntC input, MaskC mask) const;
    template <bool ad>
    Spectrum<ad> getContribution(const Scene &scene,BSDFArray<ad> bsdf_array, Intersection<ad> its, BSDFSample<ad> bs, Mask<ad> active) const;

    template <bool ad>
    Spectrum<ad> __Li(const Scene &scene, Sampler &sampler, const Ray<ad> &ray, Mask<ad> active) const;

    void __render_boundary(const Scene &scene, int sensor_id, SpectrumD &result) const override;
    void render_secondary_edges(const Scene &scene, int sensor_id, SpectrumD &result) const override;
    void render_secondary_edgesC(const Scene &scene, int sensor_id, SpectrumC &result) const override;
    std::tuple<Vector3fD,Vector3fD,FloatD,Vector3fD,Vector3fD,Vector3fD> _sample_boundary_3(const Scene &scene, RayC camera_ray, IntC idx, bool debug=false) const override;

    template <bool ad>
    std::pair<IntC, Spectrum<ad>> eval_secondary_edge(const Scene &scene, const Sensor &sensor, const Vector3fC &sample3) const;
    template <bool ad>
    std::pair<IntC, Spectrum<ad>> eval_boundary_edge(const Scene &scene, const Sensor &sensor, const Vector3fC &sample3) const;

    // std::tuple<Vector3f<true>,IntC,Vector3f<true>,IntC,Vector3f<true>,Float<true>,Float<true>,Float<true>> _eval_boundary_edge(const Scene &scene, const Sensor &sensor, const Vector3fC &sample3, BoundarySegSampleDirect bss, SecondaryEdgeInfo edge_info) const override;
    std::tuple<Vector3f<true>,IntC,Vector3f<true>,IntC,
    Vector3f<true>,Float<true>,Float<true>,SpectrumD, //Float<true>,
    Vector3fC,Vector3fC,Vector3fC,Vector3fC> _eval_boundary_edge(const Scene &scene, const Sensor &sensor, const Vector3fC &sample3,
BoundarySegSampleDirect bss, SecondaryEdgeInfo sec_edge_info) const override;
    // std::tuple<Vector3f<ad>,IntC,Vector3fD,IntC,Vectof3fD,FloatD,FloatD,FloatD> _eval_boundary_edge(const Scene &scene, const Sensor &sensor, const Vector3fC &sample3) const;

    template <bool ad>
    void eval_secondary_edge_bssrdf(const Scene &scene, const IntersectionC &its, const Sensor &sensor, const Vector3fC &sample3, SpectrumD &result) const;

    int m_bsdf_samples, m_light_samples;
    std::vector<HyperCubeDistribution3f*> m_warpper;
PSDR_CLASS_DECL_END(DirectIntegrator)

} // namespace psdr
