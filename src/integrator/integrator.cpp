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

template <bool ad>
std::tuple<BSDFSampleC,BSDFSampleC,Float<ad>> Integrator::sample_boundary(const Scene &scene, const Ray<ad> &camera_ray) const{
    BSDFSampleC bs1,bs2;
    Intersection<ad> its_curve;
    Intersection<ad> its = scene.ray_intersect<ad>(camera_ray, true);
    bs1.po = detach(its);
    Float<ad> maxDist;

    auto active = its.is_valid();
    if (any(active)){
        IntC cameraIdx = arange<IntC>(slices(its));
        IntC validRayIdx= cameraIdx.compress_(active);
        Intersection<ad> masked_its = gather<Intersection<ad>>(its,Int<ad>(validRayIdx));

        // Map value to pixel idx
        Vector8f<ad> sample = gather<Vector8f<ad>>(scene.m_samplers[0].next_nd<8,ad>(),Int<ad>(validRayIdx));
        Vector3f<ad> phi = 2.0f  * Pi * sample.x();

        Vector3f<ad> vx = masked_its.sh_frame.t;
        Vector3f<ad> vy = masked_its.sh_frame.s;
        Vector3f<ad> vz = masked_its.sh_frame.n;

        BSDFArray<ad> bsdf_array = masked_its.shape->bsdf(true);
        BSDFArrayC bsdf_arrayC = detach(bsdf_array);
        auto vaesub_bsdf = reinterpret_cast<VaeSub*>(bsdf_arrayC[0]);
        Float<ad> kernelEps = bsdf_array -> getKernelEps(masked_its,sample[5]);
        // Float<ad> scaleFactor = sqrt(kernelEps);
        maxDist = sqrt(kernelEps) * 2.0f;//* 5.0f;
        // Float<ad> maxDist(scaleFactor);
        std::cout << "maxDist " << maxDist << std::endl;
        // For some reason, gotta do it in this order...
        float eps = 0.01f;
        // Disk sample
        Vector3f<ad> rayO = vx*cos(phi) + vy*sin(phi); //+ vz*eps;
        rayO = rayO * maxDist + vz*eps;
        rayO = rayO + masked_its.p;
        Ray<ad> ray(rayO,-vz,maxDist+eps);
        // TODO: search both ways
        its_curve = scene.ray_intersect<ad, ad>(ray, true);
        std::cout << "its_curve.t " << its_curve.t << std::endl;
        std::cout << "count(its_curve.is_valid()) " << count(its_curve.is_valid()) << std::endl;
        // Ray<ad> rayB(rayO - 2.0f * vz * eps,+vz,maxDist + eps);
        // Intersection<ad> its_curve_ = scene.ray_intersect<ad, ad>(ray, true);

        bs1.po = detach(masked_its);
        bs2.po = detach(its_curve); //.p);
        bs2.is_sub = MaskC(true);
        MaskC valid = bs2.po.is_valid();
        Vector3fC lightpoint = (scene.m_emitters[0]->sample_position(Vector3fC(1.0f), Vector2fC(1.0f), valid)).p;
        Vector3fC wo = lightpoint - bs2.po.p;
        FloatC dist_sqr = squared_norm(wo);
        FloatC dist = safe_sqrt(dist_sqr);
        wo /= dist;
        bs2.wo =  bs2.po.sh_frame.to_local(wo);
        bs2.po.wi = bs2.wo;
        bs2.po.abs_prob = vaesub_bsdf->absorption<false>(bs1.po,detach(sample));
        bs2.rgb_rv = detach(sample.y());
        // TODO: set J
        bs2.po.J = FloatC(1.0f);
    }
    else{
        std::cout << "No Intersections are valid " << std::endl;
    }
    return std::tie(bs1,bs2,maxDist);
}

IntersectionC Integrator::sample_boundary_(const Scene &scene, const Vector3fC &pts, const Vector3fC &dir){
    // IntersectionC tmp;
    // return tmp;
    // auto camera_ray = RayC(pts,dir);
    Ray<false> camera_ray(pts,dir);
    BSDFSampleC bs;
    std::tie(std::ignore,bs,std::ignore) = sample_boundary<false>(scene,camera_ray);
    // std::tie(bs1,bs2,maxDist) = sample_boundary<ad>(scene,camera_ray);

    return bs.po;
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
    
    // FIXME: temp checking herer
    const bool ad = true;
    const RenderOption &opts = scene.m_opts;
    const int num_pixels = opts.cropwidth*opts.cropheight;
    // std::cout<<"pixel number: "<<num_pixels<<std::endl;
    Spectrum<ad> res = zero<Spectrum<ad>>(num_pixels);
    if ( likely(opts.spp > 0) ) {
        int64_t num_samples = static_cast<int64_t>(num_pixels)*opts.spp;
        PSDR_ASSERT(num_samples <= std::numeric_limits<int>::max());

        Int<ad> idx = arange<Int<ad>>(num_samples);
        if ( likely(opts.spp > 1) ) idx /= opts.spp;
        // std::cout<<"idx: "<<slices(idx)<<std::endl;
        Vector2f<ad> samples_base = gather<Vector2f<ad>>(meshgrid(arange<Float<ad>>(opts.cropwidth),
                                                        arange<Float<ad>>(opts.cropheight)),
                                                        idx);
        Vector2f<ad> samples = (samples_base + scene.m_samplers[0].next_2d<ad>())
                                / ScalarVector2f(opts.cropwidth, opts.cropheight);
        Ray<ad> camera_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);
        // TODO: active -> tru
        // Intersection<ad> its = scene.ray_intersect<ad>(camera_ray, true);
        // auto active = its.is_valid();
        // if (any(active)){


        // IntC cameraIdx = arange<IntC>(slices(its));
        // IntC validRayIdx= cameraIdx.compress_(active);
        // Intersection<ad> masked_its = gather<Intersection<ad>>(its,Int<ad>(validRayIdx));

        // // Map value to pixel idx
        // // auto its_curve = scene.bounding_edge<ad>(scene,test,Vectorf<1,ad>(0.05f),Vector8f<ad>(1.0f));

        // Vector8f<ad> sample = gather<Vector8f<ad>>(scene.m_samplers[0].next_nd<8,ad>(),Int<ad>(validRayIdx));
        // Vector3f<ad> phi = 2.0f  * Pi * sample.x();

        // Vector3f<ad> vx = masked_its.sh_frame.t;
        // Vector3f<ad> vy = masked_its.sh_frame.s;
        // Vector3f<ad> vz = masked_its.sh_frame.n;

        // BSDFArray<ad> bsdf_array = masked_its.shape->bsdf(true);
        // BSDFArrayC bsdf_arrayC = detach(bsdf_array);
        // auto vaesub_bsdf = reinterpret_cast<VaeSub*>(bsdf_arrayC[0]);
        // auto kernelEps = bsdf_array -> getKernelEps(masked_its,sample[5]);
        // auto scaleFactor = sqrt(kernelEps);
        // Float<ad> maxDist(scaleFactor);
        // std::cout << "maxDist " << maxDist << std::endl;
        // // For some reason, gotta do it in this order...
        // float eps = 0.03f;
        // Vector3f<ad> rayO = vx*cos(phi) + vy*sin(phi) + vz*eps;
        // rayO = rayO * maxDist;
        // rayO = rayO + masked_its.p;
        // Ray<ad> ray(rayO,-vz,maxDist+eps);  //test.p + rayO,vz,maxDist);
        // Intersection<ad> its_curve;
        // // TODO: search both ways
        // its_curve = scene.ray_intersect<ad, ad>(ray, true);
        // Intersection<ad> its_curve = sample_boundary<ad>(scene,camera_ray);
        BSDFSampleC bs1,bs2;
        Float<ad> maxDist;
        std::tie(bs1,bs2,maxDist) = sample_boundary<ad>(scene,camera_ray);
        if (any(bs1.po.is_valid())){

        BSDFArray<ad> bsdf_array = bs1.po.shape->bsdf(true);
        BSDFArrayC bsdf_arrayC = detach(bsdf_array);
        auto vaesub_bsdf = reinterpret_cast<VaeSub*>(bsdf_arrayC[0]);
        // auto kernelEps = bsdf_array -> getKernelEps(bs1.po,sample[5]);
        // auto scaleFactor = sqrt(kernelEps);
        // Float<ad> maxDist(scaleFactor);

        auto valid = bs2.po.is_valid();
        
        // auto cedge_p = its_curve.p;
        // auto cedge_p = _its_curve.p;
        // Get corresponding pixel of cedge point
        SensorDirectSampleC sds = scene.m_sensors[sensor_id]->sample_direct(detach(bs2.po.p));
        // SensorDirectSampleC sds_d = scene.m_sensors[sensor_id]->sample_direct(detach(test.p));
        // // TODO: do we need the base_v?
        
        //TODO: get rid of masked maybe?
        // bs1 is the center point, bs2 is the xB
        // BSDFSampleC bs1,bs2;
        // bs1.po = detach(masked_its);
        // bs2.po = detach(its_curve); //.p);
        // bs2.is_sub = MaskC(true);
        // MaskC valid = bs2.po.is_valid();
        // Vector3fC lightpoint = (scene.m_emitters[0]->sample_position(Vector3fC(1.0f), Vector2fC(1.0f), valid)).p;
        // Vector3fC wo = lightpoint - bs2.po.p;
        // FloatC dist_sqr = squared_norm(wo);
        // FloatC dist = safe_sqrt(dist_sqr);
        // wo /= dist;
        // bs2.wo =  bs2.po.sh_frame.to_local(wo);
        // bs2.po.wi = bs2.wo;
        // // TODO: set J
        // bs2.po.abs_prob = vaesub_bsdf->absorption<false>(bs1.po,detach(sample));
        // bs2.rgb_rv = detach(sample.y());
        // bs2.po.J = FloatC(1.0f);
        FloatC bss_pdf = 1 / (2.0f * Pi * detach(maxDist));

        SpectrumC bsdf_val = vaesub_bsdf->eval(bs1.po, bs2,valid);
        FloatC pdfpoint = vaesub_bsdf->pdfpoint(bs1.po, bs2, valid);
        SpectrumC value0 = (bsdf_val*scene.m_emitters[0]->eval(bs2.po, valid)*(sds.sensor_val/bss_pdf/pdfpoint)) & valid;

        // Derive normal of bounary curve
        Vector3fC nFace = bs1.po.sh_frame.n;
        Vector3fC l = cross(nFace,bs2.po.p - bs1.po.p);
        Vector3fC nB = cross(normalize(l),nFace);
        nB = normalize(nB);
        PSDR_ASSERT(all(dot(nFace,l)<0.01f));
        PSDR_ASSERT(all(dot(nB,l)<0.01f));
        // std::cout << "nB " << nB << std::endl;
        // Parameterize xi w.r.t parameter
        Vector3fC xvae = (bs2.po.p - bs1.po.p) / detach(maxDist);
        Vector3fD xi = Vector3fD(bs1.po.p) + Vector3fD(xvae) * maxDist;
        SpectrumD resultB = (SpectrumD(value0) * dot(Vector3fD(nB), xi)) & valid;
        scatter_add(result,resultB - detach(resultB),IntD(sds.pixel_idx));
        }

        // Spectrum<ad> red(0.0f),blue(0.0f), white(1.0f);
        // red.x() = 10.0f;
        // blue.z() = 10.0f;
        // // scatter_add(result,red,IntD(sds_d.pixel_idx));
        // if constexpr(ad)
        //     scatter_add(result,white,IntD(sds.pixel_idx));
        // else
        //     scatter_add(result,white,sds.pixel_idx);

        // }
    }
        

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
        // FIXME: tmp setting
        // value = value*0.2f;
        // masked(value, value > 0.0f) = 0.1f;
        scatter_add(result, value, idx);

        if constexpr(ad){

        // TODO: replace with size of neighborhood
        // TODO: reuse rays from Li
        // Intersection<ad> its = scene.ray_intersect<ad>(camera_ray, true);
        // auto active = its.is_valid();

        // IntC cameraIdx = arange<IntC>(slices(its));
        // IntC validRayIdx= cameraIdx.compress_(active);
        // Intersection<ad> masked_its = gather<Intersection<ad>>(its,Int<ad>(validRayIdx));
        // auto test = slice(masked_its,0);
        // Map value to pixel idx
        // std::cout << "scene.m_samples[0].is_ready() " << scene.m_samplers[0].is_ready() << std::endl;
        // auto cedge_p = scene.crossing_edge<ad>(test.p,Vectorf<1,ad>(0.05f),scene.m_samplers[0].next_2d<ad>());
        // Sampler sampler = 
        // scene.m_samplers[0].seed(arange<UInt64C>(100));
        // Vector8f<ad> test_sample(0.5f);
        // set_slices(test_sample,100);
        // auto its_curve = scene.bounding_edge<ad>(scene,test,Vectorf<1,ad>(0.05f),test_sample);
        // std::cout << "count(its_curve.is_valid()) " << count(its_curve.is_valid()) << std::endl;
        // std::cout << "cedge_p " << cedge_p << std::endl;


        // auto sample = scene.m_samplers[0].next_nd<8,ad>();
        // auto phi = 2.0  * Pi * sample.x();
        // auto vx = test.sh_frame.t;
        // auto vy = test.sh_frame.s;
        // auto vz = test.sh_frame.n;

        // // // Make bounding sphere
        // // Float<ad> maxDist(0.05f);
        // Float<ad> maxDist = Vectorf<1,ad>(0.05f)[0];
        // Ray<ad> ray(test.p + maxDist * (cos(phi)*vx + vy*sin(phi)),vz,maxDist);
        // Intersection<ad> its_curve;
        // its_curve = scene.ray_all_intersect<ad, ad>(ray, true, sample, 5);


        // auto cedge_p = its_curve.p;
        // std::cout << "norm(cedge_p - its.p) " << norm(detach(cedge_p)- detach(test.p)) << std::endl;
        // std::cout << "cedge_p " << cedge_p << std::endl;
        // Get corresponding pixel of cedge point
        // SensorDirectSampleC sds = scene.m_sensors[sensor_id]->sample_direct(detach(cedge_p));
        // SensorDirectSampleC sds_d = scene.m_sensors[sensor_id]->sample_direct(detach(test.p));
        // // // // SensorDirectSampleC sds_its = scene.m_sensors[sensor_id]->sample_direct(detach(its.p));
        // // scatter_add(result,Spectrum<ad>(10.0f),IntD(sds_its.pixel_idx));

        // Spectrum<ad> red(0.0f),blue(0.0f), white(1.0f);
        // red.x() = 10.0f;
        // blue.z() = 10.0f;
        // scatter_add(result,red,IntD(sds_d.pixel_idx));
        // // // // // scatter_add(result,Spectrum<ad>(1.0f),IntD(sds_d.pixel_idx-1));
        // // // // // scatter_add(result,Spectrum<ad>(1.0f),IntD(sds_d.pixel_idx+1));
        // if constexpr(ad)
        //     scatter_add(result,white,IntD(sds.pixel_idx));
        // else
        //     scatter_add(result,white,sds.pixel_idx);
        

        }
        }

        if ( likely(opts.spp > 1) ) {
            result /= static_cast<float>(opts.spp);
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

    template std::tuple<BSDFSampleC,BSDFSampleC,Float<true>> Integrator::sample_boundary<true>(const Scene &scene, const Ray<true> &camera_ray) const;
    template std::tuple<BSDFSampleC,BSDFSampleC,Float<false>> Integrator::sample_boundary<false>(const Scene &scene, const Ray<false> &camera_ray) const;
} // namespace psdr
