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
    SpectrumC renderC_shape(const Scene &scene, const IntersectionC &its, int sensor_id) const {} 
    SpectrumD renderD_shape(const Scene &scene, const IntersectionD &its, int sensor_id) const {}
    // Vector3f<false> sample_sub(const Scene &scene, Vector3f<false> pts, Vector3f<false> dir);
    // std::tuple<Vector3fC,Vector3fC,Vector3fC,Array<Array<Float<false>,20>,3>> sample_sub(const Scene &scene, Vector3f<false> pts, Vector3f<false> dir);
    // std::tuple<Vector3fC,Vector3fC,Vector3fC,Array<Array<float,20>,3>> sample_sub(const Scene &scene, Vector3f<false> pts, Vector3f<false> dir);
    std::tuple<Vector3fC,Vector3fC,Vector3fC,Vector20fC,Float<false>> sample_sub(const Scene &scene, Vector3f<false> pts, Vector3f<false> dir);
    template <bool ad>
    // Deprecated
    std::tuple<BSDFSampleC,BSDFSampleC,Float<ad>> sample_boundary(const Scene &scene, const Ray<ad> &camera_ray) const;
    IntersectionC sample_boundary_(const Scene &scene, const Vector3fC &pts, const Vector3fC &dir); 

    std::tuple<Vector3fC,Vector3fC,Vector3fC,Vector3fC,Vector3fC,Vector3fC> sample_boundary_2(Scene &scene, int edge_idx);
    std::tuple<Vector3fD, Vector3fD, FloatD,Vector3fD,Vector3fD,Vector3fD> sample_boundary_3(const Scene &scene, const Vector3fC &pts, const Vector3fC &dir); 
    // std::tuple<Vector3fD,IntC,Vector3fD,IntC,Vector3fD,FloatD,FloatD,FloatD> sample_boundary_4(const Scene &scene, int idx);
    std::tuple<Vector3fD,IntC,Vector3fD,IntC,Vector3fD,FloatD,FloatD,SpectrumD,Vector3fC,Vector3fC,Vector3fC,Vector3fC> sample_boundary_4(const Scene &scene, int idx);
    // std::tuple<Vector3f<true>,IntC,Vector3f<true>,IntC,Vector3f<true>,Float<true>,Float<true>,Float<true>> sample_boundary_4(const Scene &scene, const Vector3fC &pts, const Vector3fC &dir);
// template FloatC Scene::emitter_position_pdf<false>(const Vector3fC&, const IntersectionC&, MaskC) const;
// template FloatD Scene::emitter_position_pdf<true >(const Vector3fD&, const IntersectionD&, MaskD) const;
    // std::tuple<BSDFSampleC,BSDFSampleC,FloatC> sample_boundary(const Scene &scene, const RayC &camera_ray) const;
    // std::tuple<BSDFSampleD,BSDFSampleD,FloatD> sample_boundary(const Scene &scene, const RayD &camera_ray)const ;
    // std::tuple<Vector3fC,Vector3fC,Vector3fC,Array<Array<float,20>,3>> sample_sub(const Scene &scene, Vector3f<false> pts, Vector3f<false> dir);
    // Vector3f<false> sample_sub(const Scene &scene, Vector3f<false> pts, Vector3f<false> dir);
    
    IntersectionC getIntersectionC(const Scene &scene, int sensor_id=0) const;
    IntersectionD getIntersectionD(const Scene &scene, int sensor_id=0) const;

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

    virtual void __render_boundary(const Scene &scene, int sensor_id, SpectrumD &result) const {}
    virtual void render_secondary_edges(const Scene &scene, int sensor_id, SpectrumD &result) const {}
    virtual void render_secondary_edgesC(const Scene &scene, int sensor_id, SpectrumC &result) const {}
    virtual std::tuple<Vector3fD ,Vector3fD,FloatD,Vector3fD,Vector3fD,Vector3fD> _sample_boundary_3(const Scene &scene, RayC camera_ray, IntC idx, bool debug=false) const {}
    // virtual std::tuple<Vector3f<ad>,IntC,Vector3f<ad>,IntC,Vector3f<ad>,Float<ad>,Float<ad>,Float<ad>> _eval_boundary_edge(const Scene &scene, const Sensor &sensor, const Vector3fC &sample3) const {}
    // template <bool ad>
    // virtual std::tuple<Vector3f<true>,IntC,Vector3f<true>,IntC,Vector3f<true>,Float<true>,Float<true>,Float<true>> _eval_boundary_edge(const Scene &scene, const Sensor &sensor, const Vector3fC &sample3,
    // BoundarySegSampleDirect bss, SecondaryEdgeInfo sec_edge_info) const {}
    virtual std::tuple<Vector3f<true>,IntC,Vector3f<true>,IntC,
    Vector3f<true>,Float<true>,Float<true>,SpectrumD, //Float<true>,
    Vector3fC,Vector3fC,Vector3fC,Vector3fC> _eval_boundary_edge(const Scene &scene, const Sensor &sensor, const Vector3fC &sample3,
    BoundarySegSampleDirect bss, SecondaryEdgeInfo sec_edge_info) const {}

    template <bool ad>
    Spectrum<ad> __render(const Scene &scene, int sensor_id) const;
    // NOTE: Joon added
    template <bool ad>
    Spectrum<ad> __render_shape(const Scene &scene, const Intersection<ad> &its, int sensor_id) const;
    // NOTE: Joon added
    template <bool ad>
    Intersection<ad> getIntersection(const Scene &scene, int sensor_id=0) const;
PSDR_CLASS_DECL_END(SamplingIntegrator)

} // namespace psdr
