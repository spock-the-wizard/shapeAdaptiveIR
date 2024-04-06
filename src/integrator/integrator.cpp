#include <chrono>
#include <misc/Exception.h>
#include <psdr/core/ray.h>
#include <psdr/core/intersection.h>
#include <psdr/core/sampler.h>
#include <psdr/scene/scene.h>
#include <psdr/sensor/sensor.h>
#include <psdr/integrator/integrator.h>

#include <psdr/core/bitmap.h>
#include <psdr/core/warp.h>
#include <psdr/core/ray.h>
#include <psdr/shape/mesh.h>
#include <psdr/scene/scene.h>
#include <psdr/utils.h>
#include <psdr/bsdf/ggx.h>

#include <psdr/bsdf/vaesub.h>
#include <psdr/bsdf/hetersub.h>
// #include <psdr/bsdf/polynomials.h>

#include <enoki/cuda.h>
#include <enoki/autodiff.h>
#include <enoki/special.h>
#include <enoki/random.h>

namespace psdr
{

IntersectionC Integrator::getIntersectionC(const Scene &scene, int sensor_id) const {
    return getIntersection<false>(scene,sensor_id);
}

IntersectionD Integrator::getIntersectionD(const Scene &scene, int sensor_id) const {
    return getIntersection<true>(scene,sensor_id);
}

template <bool ad>
Intersection<ad> Integrator::getIntersection(const Scene &scene, int sensor_id) const {
// IntersectionC Integrator::getIntersection(const Scene &scene, int sensor_id) const {
    using namespace std::chrono;
    // auto start_time = high_resolution_clock::now();

    PSDR_ASSERT_MSG(scene.is_ready(), "Input scene must be configured!");
    PSDR_ASSERT_MSG(sensor_id >= 0 && sensor_id < scene.m_num_sensors, "Invalid sensor id!");

    const RenderOption &opts = scene.m_opts;
    const int num_pixels = opts.cropwidth*opts.cropheight;
    // const bool ad = false;
    Intersection<ad> its;
    if ( likely(opts.spp > 0) ) {
        int64_t num_samples = static_cast<int64_t>(num_pixels)*opts.spp;
        PSDR_ASSERT(num_samples <= std::numeric_limits<int>::max());

        Int<ad> idx = arange<Int<ad>>(num_samples);
        if ( likely(opts.spp > 1) ) idx /= opts.spp;
        // std::cout<<"idx: "<<slices(idx)<<std::endl;
        Vector2f<ad> samples_base = gather<Vector2f<ad>>(meshgrid(arange<Float<ad>>(opts.cropwidth),
                                                        arange<Float<ad>>(opts.cropheight)),
                                                        idx);
        // std::cout << "samples_base " << samples_base << std::endl;


        Vector2f<ad> samples = (samples_base + scene.m_samplers[0].next_2d<ad>())
                                / ScalarVector2f(opts.cropwidth, opts.cropheight);
        // std::cout<<"sampled: "<<slices(samples)<<std::endl;
        Ray<ad> camera_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);
        
        // Compute ray intersection
        its = scene.ray_intersect<ad>(camera_ray, true);

    }

    cuda_eval(); cuda_sync();

    // auto end_time = high_resolution_clock::now();
    // if ( scene.m_opts.log_level ) {
    //     std::stringstream oss;
    //     oss << "Rendered in " << duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";
    //     LOG(oss.str().c_str());
    // }
    return its;
}

SpectrumC Integrator::renderC_shape(const Scene &scene, const IntersectionC &its, int sensor_id) const {
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    // std::cout << "[renderC_shape]  its.poly_coeff " << its.poly_coeff << std::endl;
    SpectrumC result = __render_shape<false>(scene, its, sensor_id);

    cuda_eval(); cuda_sync();

    auto end_time = high_resolution_clock::now();
    if ( scene.m_opts.log_level ) {
        std::stringstream oss;
        oss << "Rendered in " << duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";
        LOG(oss.str().c_str());
    }

    return result;
}

SpectrumC Integrator::renderC(const Scene &scene, int sensor_id) const {
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    SpectrumC result = __render<false>(scene, sensor_id);

    cuda_eval(); cuda_sync();

    auto end_time = high_resolution_clock::now();
    if ( scene.m_opts.log_level ) {
        std::stringstream oss;
        oss << "Rendered in " << duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";
        LOG(oss.str().c_str());
    }

    return result;
}

std::tuple<Vector3fC, Vector3fC, Vector3fC, Vector20fC, FloatC> Integrator::sample_sub(const Scene &scene, Vector3f<false> pts, Vector3f<false> dir) {
    constexpr bool ad = false;
    auto active = full<Mask<ad>>(true); //~isnan(pts[0]);

    // TODO: do not set tmax... fu... use RayC instead (tmax set to inf)
    // auto tmax = full<Float<ad>>(3000.0f);
    // Ray<ad> ray(pts,dir,tmax);
    Intersection<ad> its = scene.ray_intersect<ad>(RayC(pts,dir), true);
    // its.poly_coeff = zero<decltype(its.poly_coeff)>();
    for (int i=0;i<3;i++){
    //     // Case 1. Plane f = z
        // its.poly_coeff[i][3] = 1.0;

        // Case 2. f = -z + x^2 + y^2
        // its.poly_coeff[i][3] = 1.0f;
        // its.poly_coeff[i][4] = -1.0f;
        // its.poly_coeff[i][7] = -1.0f;
    }
    
    Float<ad> pdf;
    Sampler sampler;
    sampler.seed(arange<UInt64C>(slices(pts)));

    // Type<HeterSub*,ad> vaesub_array = its.shape->bsdf(true);
    Type<VaeSub*,ad> vaesub_array = its.shape->bsdf(true);
    auto res = vaesub_array[0] -> __sample_sp<ad>(&scene, its, (sampler.next_nd<8, false>()),pdf, active);
    auto res1 = vaesub_array[0] -> __sample_sub<ad>(&scene, its, (sampler.next_nd<8, false>()), active);
    auto res2 = vaesub_array[0] -> __eval_sub<ad>(its, res1, active);
    // std::cout << "res1 " << res1 << std::endl;
    std::cout << "res1.pdf " << res1.pdf << std::endl;
    // std::cout << "res2 " << res2.x() << std::endl;
    auto weight = res2.x() / res1.pdf;
    // auto weight = res2.x();
    std::cout << "weight " << weight << std::endl;
    // poly_coeffs = its.poly_coeffs
    Intersection<ad> its_sub;
    Float<ad> second;
    Vector3f<ad> outPos;
    Vector3f<ad> projDir;
    std::tie(its_sub,second,outPos,projDir) = res;
    its_sub = res1.po;


    Array<Float<ad>,20> poly_coeffs = its.poly_coeff.x();
    Array<Float<ad>,3> s,t,n;

    // TODO: change this to its
    n = -its.wi;
    // n = its.wi;
    // n = its.sh_frame.n;
    // n = -its_debug.wi;
    // onb<ad>(n, s, t);
    Vector3f<ad> forward = zero<Vector3f<ad>>();
    Vector3f<ad> up = zero<Vector3f<ad>>();
    Vector3f<ad> right = zero<Vector3f<ad>>();

    forward.x() = 1.0f;
    up.y() = 1.0f;
    right.z() = 1.0f;
    
    auto active_sub = its_sub.is_valid();
    Float<ad> color = weight; 

    return std::make_tuple(its_sub.p,outPos,projDir,poly_coeffs,color);
}
SpectrumD Integrator::renderD_shape(const Scene &scene, const IntersectionD &its, int sensor_id) const {
    // Interior integral

    // std::cout << "its.poly_coeff " << its.poly_coeff << std::endl;
    SpectrumD result = __render_shape<true>(scene, its,sensor_id);

    // Boundary integral
    if ( likely(scene.m_opts.sppe > 0) ) {
        render_primary_edges(scene, sensor_id, result);
    }

    if ( likely(scene.m_opts.sppse > 0) ) {
        // std::cout<<" secondary edge "<<std::endl;
        render_secondary_edges(scene, sensor_id, result);
    }
    
    cuda_eval(); cuda_sync();

    if ( scene.m_opts.log_level ) {
        std::stringstream oss;
        // oss << "Rendered in " << duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";
        // LOG(oss.str().c_str());
    }

    return result;
}

SpectrumD Integrator::renderD(const Scene &scene, int sensor_id) const {
    using namespace std::chrono;
    // Interior integral

    SpectrumD result = __render<true>(scene, sensor_id);

    // Boundary integral
    if ( likely(scene.m_opts.sppe > 0) ) {
        render_primary_edges(scene, sensor_id, result);
    }

    if ( likely(scene.m_opts.sppse > 0) ) {
        // std::cout<<" secondary edge "<<std::endl;
        render_secondary_edges(scene, sensor_id, result);
    }
    
    cuda_eval(); cuda_sync();

    if ( scene.m_opts.log_level ) {
        std::stringstream oss;
        // oss << "Rendered in " << duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";
        // LOG(oss.str().c_str());
    }

    return result;
}

template <bool ad>
Spectrum<ad> Integrator::__render_shape(const Scene &scene, const Intersection<ad> &its, int sensor_id) const {
    PSDR_ASSERT_MSG(scene.is_ready(), "Input scene must be configured!");
    PSDR_ASSERT_MSG(sensor_id >= 0 && sensor_id < scene.m_num_sensors, "Invalid sensor id!");

    const RenderOption &opts = scene.m_opts;
    const int num_pixels = opts.cropwidth*opts.cropheight;
    // std::cout<<"pixel number: "<<num_pixels<<std::endl;
    Spectrum<ad> result = zero<Spectrum<ad>>(num_pixels);
    if ( likely(opts.spp > 0) ) {
        int64_t num_samples = static_cast<int64_t>(num_pixels)*opts.spp;
        PSDR_ASSERT(num_samples <= std::numeric_limits<int>::max());

        Int<ad> idx = arange<Int<ad>>(num_samples);
        if ( likely(opts.spp > 1) ) idx /= opts.spp;
        Spectrum<ad> value = Li_shape(scene, scene.m_samplers[0], its, true, sensor_id);
        masked(value, ~enoki::isfinite<Spectrum<ad>>(value)) = 0.f;
        scatter_add(result, value, idx);
        if ( likely(opts.spp > 1) ) {
            result /= static_cast<float>(opts.spp);
        }
    }

    return result;
}


template <bool ad>
Spectrum<ad> Integrator::__render(const Scene &scene, int sensor_id) const {
    PSDR_ASSERT_MSG(scene.is_ready(), "Input scene must be configured!");
    PSDR_ASSERT_MSG(sensor_id >= 0 && sensor_id < scene.m_num_sensors, "Invalid sensor id!");

    const RenderOption &opts = scene.m_opts;
    const int num_pixels = opts.cropwidth*opts.cropheight;
    // std::cout<<"pixel number: "<<num_pixels<<std::endl;
    Spectrum<ad> result = zero<Spectrum<ad>>(num_pixels);
    if ( likely(opts.spp > 0) ) {
        int64_t num_samples = static_cast<int64_t>(num_pixels)*opts.spp;
        PSDR_ASSERT(num_samples <= std::numeric_limits<int>::max());
        // std::cout<<"sample number values: "<<num_samples<<std::endl;

        Int<ad> idx = arange<Int<ad>>(num_samples);
        if ( likely(opts.spp > 1) ) idx /= opts.spp;
        // std::cout<<"idx: "<<slices(idx)<<std::endl;
        Vector2f<ad> samples_base = gather<Vector2f<ad>>(meshgrid(arange<Float<ad>>(opts.cropwidth),
                                                        arange<Float<ad>>(opts.cropheight)),
                                                        idx);
        // std::cout << "samples_base " << samples_base << std::endl;


        Vector2f<ad> samples = (samples_base + scene.m_samplers[0].next_2d<ad>())
                                / ScalarVector2f(opts.cropwidth, opts.cropheight);
        // std::cout<<"sampled: "<<slices(samples)<<std::endl;
        Ray<ad> camera_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);
        // std::cout<<"camera_ray: "<<slices(camera_ray)<<std::endl;
        Spectrum<ad> value = Li(scene, scene.m_samplers[0], camera_ray, true, sensor_id);
        masked(value, ~enoki::isfinite<Spectrum<ad>>(value)) = 0.f;
        // std::cout<<"Li: "<<slices(value)<<std::endl;
        scatter_add(result, value, idx);
        if ( likely(opts.spp > 1) ) {
            result /= static_cast<float>(opts.spp);
        }
            
        
        // TODO: Joon added
        // for(int i=0;i<opts.spp;i++){
        //     std::cout << "i " << i << std::endl;
        //     // int64_t num_samples_chunk = static_cast<int64_t>(num_pixels);

        //     Int<ad> idx = arange<Int<ad>>(num_pixels);
        //     // if ( likely(opts.spp > 1) ) idx /= opts.spp;
        //     Vector2f<ad> samples_base = gather<Vector2f<ad>>(meshgrid(arange<Float<ad>>(opts.cropwidth),
        //                                                     arange<Float<ad>>(opts.cropheight)),
        //                                                     idx);
        //     // std::cout << "samples_base " << samples_base << std::endl;


        //     Vector2f<ad> samples = (samples_base + scene.m_samplers[0].next_2d<ad>())
        //                             / ScalarVector2f(opts.cropwidth, opts.cropheight);
        //     // std::cout<<"sampled: "<<slices(samples)<<std::endl;
        //     Ray<ad> camera_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);
        //     // std::cout<<"camera_ray: "<<slices(camera_ray)<<std::endl;
        //     Spectrum<ad> value = Li(scene, scene.m_samplers[0], camera_ray, true, sensor_id);
        //     masked(value, ~enoki::isfinite<Spectrum<ad>>(value)) = 0.f;
        //     // std::cout<<"Li: "<<slices(value)<<std::endl;
        //     scatter_add(result, value, idx);
        // }
        // if ( likely(opts.spp > 1) ) {
        //     result /= static_cast<float>(opts.spp);
        // }
    }

    return result;
}


void Integrator::render_primary_edges(const Scene &scene, int sensor_id, SpectrumD &result) const {
    const RenderOption &opts = scene.m_opts;

    const Sensor *sensor = scene.m_sensors[sensor_id];
    if ( sensor->m_enable_edges ) {
        PrimaryEdgeSample edge_samples = sensor->sample_primary_edge(scene.m_samplers[1].next_1d<false>());
        MaskC valid = (edge_samples.idx >= 0);
#ifdef PSDR_PRIMARY_EDGE_VIS_CHECK
        IntersectionC its = scene.ray_intersect<false>(edge_samples.ray_c);
        valid &= ~its.is_valid();
#endif
        SpectrumC delta_L = Li(scene, scene.m_samplers[1], edge_samples.ray_n, valid, sensor_id) - 
                            Li(scene, scene.m_samplers[1], edge_samples.ray_p, valid, sensor_id);
                            
                            // estimate subgradient
        SpectrumD value = edge_samples.x_dot_n*SpectrumD(delta_L/edge_samples.pdf);
        masked(value, ~enoki::isfinite<SpectrumD>(value)) = 0.f;
        if ( likely(opts.sppe > 1) ) {
            value /= static_cast<float>(opts.sppe);
        }
        value -= detach(value);
        scatter_add(result, value, IntD(edge_samples.idx), valid);
    }
}

} // namespace psdr
