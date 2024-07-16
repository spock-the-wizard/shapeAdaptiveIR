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
#include <iostream>
#include <fstream>

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

SpectrumC Integrator::renderC(const Scene &scene, int sensor_id) const {
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();
    

    SpectrumC result = __render<false>(scene, sensor_id);
    
    // FIXME: Debugging edge
    // if ( likely(scene.m_opts.sppse > 0) ) {
    //     render_secondary_edgesC(scene, sensor_id, result);
    // }

    cuda_eval(); cuda_sync();

    auto end_time = high_resolution_clock::now();
    if ( scene.m_opts.log_level ) {
        std::stringstream oss;
        oss << "Rendered in " << duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";
        LOG(oss.str().c_str());
    }

    return result;
}

//TODO: visualizer for boundary term
// std::tuple<Vector3fD,Vector3fD,FloatD,Vector3fD,Vector3fD,Vector3fD> Integrator::sample_boundary_4(const Scene &scene, const Vector3fC &pts, const Vector3fC &dir){
//     // IntersectionC tmp;
//     // return tmp;
//     // auto camera_ray = RayC(pts,dir);
//     Ray<false> camera_ray(pts,dir);
//     IntC idx(0);
//     BSDFSampleC bs;
//     FloatD weight;
//     Vector3fD its_p, value, vDis, vDot,d_wLe;
//     std::tie(its_p,value,weight,vDis,d_wLe,vDot) = _sample_boundary_3(scene,camera_ray,idx,true);
//     // std::cout << "weight " << weight << std::endl;

//     return std::tie(its_p,value,weight,vDis,d_wLe,vDot);
// } 

std::tuple<Vector3fC,Vector3fC,Vector3fC,Vector3fC,Vector3fC,Vector3fC> Integrator::sample_boundary_2(Scene &scene, int edge_idx) {
    /*
    * Debug boundary integral computation
    * @param scene
    * @param edge_idx
    * @return xi[Vector3fC] sampled edge point (xi)
    * @return xo[Vector3fC] VAE-sampled camera point (xo)
    * @return its_p,its_n [Vector3fC]
    * @return G- - G+[Vector3fC] discontinuity delta
    * @return grad[Vector3fC] magnitude of gradient
    */
    
    // Sample only input edge
    // scene.setEdgeDistr(edge_idx);
    // scene.configure();
    int nSamples = 128;

    // Vector3fC sample3(0.5f);
    // set_slices(sample3,nSamples);
    scene.m_samplers[2].seed(arange<UInt64C>(nSamples)); //+m_seed);
    Vector3fC sample3 = scene.m_samplers[2].next_nd<3, false>();

    BoundarySegSampleDirect bss;
    SecondaryEdgeInfo sec_edge_info;
    std::tie(bss,sec_edge_info) = scene.sample_boundary_segment_v3(sample3,true,edge_idx);
    Vector3fC _p0    = detach(bss.p0);
    Vector3fC _p2    = bss.p2;
    Vector3fC _dir    = normalize(_p2 - _p0);                    
    MaskC valid = bss.is_valid;
    // std::cout << "valid " << valid << std::endl;
    // std::cout << "hmax(_p0) " << hmax(_p0) << std::endl;
    // std::cout << "hmin(_p0) " << hmin(_p0) << std::endl;




    const bool ad = false; 
    Sensor& sensor = *scene.m_sensors[0];
    // IntersectionC _its2;
    TriangleInfoD tri_info;
    IntersectionC _its1;
    // Light ray intersection on edge
    if constexpr ( ad ) {
        _its1 = scene.ray_intersect<false>(RayC(_p2, -_dir), MaskC(valid), &tri_info);
    } else {
        _its1 = scene.ray_intersect<false>(RayC(_p2, -_dir), valid);
    }
    valid &=(_its1.is_valid());
    valid &= norm(_its1.p - _p0) < 0.02f;
    // std::cout << "[2] valid count" << count(valid) <<std::endl;
    // TODO: diff
    // SensorDirectSampleC sds3= sensor.sample_direct(_p0);
    // RayC cam_ray = sensor.sample_primary_ray(sds2.q);

    BSDFSampleC bs1, bs2;
    BSDFSampleD bs1D;
    BSDFArrayC bsdf_array = _its1.shape->bsdf(valid);
    BSDFArrayD bsdf_arrayD = BSDFArrayD(bsdf_array);
    auto vaesub_bsdf = reinterpret_cast<VaeSub*>(bsdf_arrayD[0]);
    auto bs1_samples = scene.m_samplers[2].next_nd<8, false>();
    // TODO: account for bias with weight
    bs1_samples[0] = 1.1f;
    // auto weight_sub = (1-F);
    // TODO: diff
    SensorDirectSampleC sds2 = sensor.sample_direct(_its1.p);
    RayC cam_ray = sensor.sample_primary_ray(sds2.q);
    _its1.wi = _its1.sh_frame.to_local(-cam_ray.d);

    // Sample camera ray
    bs1D = bsdf_arrayD->sample(&scene, IntersectionD(_its1), Vector8fD(bs1_samples),  MaskD(valid));
    bs1 = detach(bs1D);
    SensorDirectSampleC sds = sensor.sample_direct(bs1.po.p);
    
    // xo is a valid VAE output and seen from camera
    valid &= bs1.po.is_valid();
    // valid &= sds.is_valid;
    // std::cout << "[3] valid count" << count(valid) <<std::endl;

    Ray<ad> camera_ray;
    // Camera its ray
    Intersection<ad> its2;
    if constexpr ( ad ) {
        camera_ray = sensor.sample_primary_ray(Vector2fD(sds.q));
        its2 = scene.ray_intersect<true, false>(camera_ray, valid);
        // valid &= its2.is_valid() && norm(detach(its2.p) - bs1.po.p) < ShadowEpsilon;
    } else {
        camera_ray = sensor.sample_primary_ray(sds.q);
        its2 = scene.ray_intersect<false>(camera_ray, valid);
        // valid &= its2.is_valid() && norm(its2.p - bs1.po.p) < ShadowEpsilon;
    }

    Vector3fC d0;
    if constexpr ( ad ) {
        d0 = -detach(camera_ray.d);
    } else {
        d0 = -camera_ray.d;
    }
    bs1.wo =  bs1.po.sh_frame.to_local(d0);
    bs1.po.wi =  bs1.wo;
    bs2.po = _its1;
    bs2.wo =  bs2.po.sh_frame.to_local(_dir);
    bs2.po.wi =  bs2.wo;
    bs2.rgb_rv = bs1.rgb_rv;
    bs2.po.abs_prob = bs1.po.abs_prob;
    bs2.is_sub = bsdf_array -> hasbssdf();
    // NOTE: don't forget this part...!! 
    // Needed for eval_sub
    bs2.is_valid = _its1.is_valid();
    bs2.pdf = bss.pdf;
    bs2.po.J = FloatC(1.0f);

    SpectrumC bsdf_val = bsdf_array->eval(bs1.po, bs2, valid);
    FloatC pdfpoint = bsdf_array->pdfpoint(bs1.po, bs2, valid);
    SpectrumC value0 = (bsdf_val*scene.m_emitters[0]->eval(bs2.po, valid)*(sds.sensor_val/bss.pdf/pdfpoint)) & valid;

    Vector3fC edge_n0 = normalize(detach(sec_edge_info.n0));
    Vector3fC edge_n1 = normalize(detach(sec_edge_info.n1));
    // // FIXME: should this be dependent on relative position of light ray?
    
    // Vector3fC line_n0 = normalize(cross(edge_n0,bss.edge));
    // Vector3fC line_n1 = normalize(cross(edge_n1,bss.edge));
    // FIXED
    FloatC n0IsLeft = sign(dot(edge_n0,cross(bss.edge,bss.edge2)));
    Vector3fC line_n0 = normalize(cross(edge_n0,bss.edge))*n0IsLeft;
    Vector3fC line_n1 = -normalize(cross(edge_n1,bss.edge))*n0IsLeft;
    
    // Vector3fC line_n0 = normalize(cross(-bss.edge,edge_n0));
    // Vector3fC line_n1 = normalize(cross(-edge_n1,bss.edge));

    // Vector3fC n = line_n0;
    // FIXME: changed
    // Vector3fC n = normalize(cross(_its1.n,bss.edge));
    
    // Compute geometry term for left and right limits
    float eps = 0.05f;
    auto lightpoint = _p2;
    Vector3fC pos_p = bs2.po.p + eps * line_n0;
    Vector3fC pos_n = bs2.po.p + eps * line_n1;
    Vector3fC dir_p = normalize(lightpoint - pos_p);
    Vector3fC dir_n = normalize(lightpoint - pos_n);
    IntersectionC its_p = scene.ray_intersect<false>(RayC(pos_p,dir_p),valid);
    IntersectionC its_n = scene.ray_intersect<false>(RayC(pos_n,dir_n),valid);
    MaskC vis_n = ~its_n.is_valid();
    MaskC vis_p = ~its_p.is_valid();
    // FIXME: tmp disabled 
    // FloatC cos_theta_o = FrameC::cos_theta(bs2.wo);
    // value0 /= cos_theta_o;
    // std::cout << "cos_theta_o " << cos_theta_o << std::endl;
    
    // edge normal
    // deltaG = (G-)-(G+)
    
    FloatC cos_n = dot(edge_n1,_dir);
    FloatC cos_p = dot(edge_n0,_dir);
    FloatC  delta = cos_n & vis_n;
    delta -= cos_p & vis_p;
    

    // FIXME: checking
    auto isN0 = (dot(_its1.n,edge_n0) - dot(_its1.n,edge_n1));
    Vector3fC n = select(isN0 > 0.0f,line_n0,line_n1);
    // Vector3fC n = normalize(cross(_its1.n,bss.edge));
    // FIXME: checking
    delta *= sign(isN0);
    // delta *= -sign(dot(_its1.n,edge_n0)-dot(_its1.n,edge_n1));
    value0 *= delta;
    valid &= (vis_n ^ vis_p);
    // posp = select(isN0 > 0.0f,line_n0,line_n1);





    // delta *= -sign(dot(_its1.n,edge_n0)-dot(_its1.n,edge_n1));
    // value0 *= delta;
    // // FIXME: add only sharp edges
    // // valid &= (vis_n & vis_p); //
    // // valid &= (vis_n ^ vis_p);
    // valid &= (~isnan(edge_n0)) && (~isnan(edge_n1));

    // // Reparameterize xi
    // // xi = xo + x_VAE / scalefactor
    FloatD kernelEps = bsdf_arrayD -> getKernelEps(IntersectionD(_its1),FloatD(bs1_samples[5]));
    FloatD maxDist = sqrt(kernelEps);
    Vector3fC xo = detach(bs1D.po.p);
    Vector3fC xvae = (bs2.po.p - bs1.po.p) / detach(maxDist);
    std::cout << "bs2.po.p " << bs2.po.p << std::endl;
    std::cout << "bs1.po.p " << bs1.po.p << std::endl;
    std::cout << "maxDist " << maxDist << std::endl;
    xvae = normalize(xvae);
    

    Vector3fD xi = Vector3fD(xo) + Vector3fD(xvae) * maxDist;

    // FIXME: testing
    // SpectrumD result = (SpectrumD(value0) * dot(Vector3fD(n), xi)) & valid;

    // NOTE: tmp fix 
    Vector3fC grad = dot(n,xvae);
    std::cout << "grad " << grad << std::endl;
    std::cout << "n " << n << std::endl;
    std::cout << "xvae " << xvae << std::endl;
    // SpectrumC result = grad & valid; //(delta * grad) & valid;
    SpectrumC result = (value0 * grad) & valid;
    // SpectrumC result = (delta * grad) & valid;


    return std::tie(_p0,bs1.po.p,pos_p, pos_n, delta,result);

}

// Deprecated
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
        // TODO: set absorption
        // bs2.po.abs_prob = vaesub_bsdf->absorption<false>(bs1.po,detach(sample));
        bs2.rgb_rv = detach(sample.y());
        // TODO: set J
        bs2.po.J = FloatC(1.0f);
    }
    else{
        std::cout << "No Intersections are valid " << std::endl;
    }
    return std::tie(bs1,bs2,maxDist);
}

// Deprecated
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
// //TODO: visualizer for boundary term
std::tuple<Vector3fD,IntC,Vector3fD,IntC,Vector3fD,FloatD,FloatD,SpectrumD,Vector3fC,Vector3fC,Vector3fC,Vector3fC> Integrator::sample_boundary_4(const Scene &scene, int idx) {
    int sensor_id = 4;

    const RenderOption &opts = scene.m_opts;
    Vector3fC sample3 = scene.m_samplers[2].next_nd<3, false>();

    BoundarySegSampleDirect bss;
    SecondaryEdgeInfo sec_edge_info;
    std::tie(bss,sec_edge_info) = scene.sample_boundary_segment_v2(sample3);


    // scene.m_samplers[2].seed(arange<UInt64C>(slices(bss.p0)));
    
    auto sensor = scene.m_sensors[sensor_id];

    return _eval_boundary_edge(scene,*sensor,sample3,bss,sec_edge_info);
}

//TODO: visualizer for boundary term
std::tuple<Vector3fD,Vector3fD,FloatD,Vector3fD,Vector3fD,Vector3fD> Integrator::sample_boundary_3(const Scene &scene, const Vector3fC &pts, const Vector3fC &dir){
    // IntersectionC tmp;
    //
    // return tmp;
    // auto camera_ray = RayC(pts,dir);
    Ray<false> camera_ray(pts,dir);

    IntC idx(0);
    BSDFSampleC bs;
    FloatD weight;
    Vector3fD its_p, value, vDis, vDot,d_wLe;
    std::tie(its_p,value,weight,vDis,d_wLe,vDot) = _sample_boundary_3(scene,camera_ray,idx,true);
    // std::cout << "weight " << weight << std::endl;

    return std::tie(its_p,value,weight,vDis,d_wLe,vDot);
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
    auto weight = res2.x() / res1.pdf;
    // auto weight = res2.x();
    // poly_coeffs = its.poly_coeffs
    Intersection<ad> its_sub;
    Float<ad> second;
    Vector3f<ad> outPos;
    Vector3f<ad> projDir;
    std::tie(its_sub,second,outPos,projDir,std::ignore,std::ignore,std::ignore) = res;
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

SpectrumD Integrator::renderD(const Scene &scene, int sensor_id) const {
    using namespace std::chrono;

    // Interior integral
    SpectrumD result = __render<true>(scene, sensor_id);
    
    // NOTE: Do not test interior term wehn spp is nonzero.
    // if ( likely(scene.m_opts.sppe > 0) ) {
    //     result = detach(result);
    // }
    
    // TODO: testing WAS-based boundary term
    // __render_boundary(scene,sensor_id,result);

    // Boundary integral
    if ( likely(scene.m_opts.sppe > 0) ) {
        render_primary_edges(scene, sensor_id, result);
    }

    if ( likely(scene.m_opts.sppse > 0) ) {
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

std::pair<SpectrumD,FloatC> Integrator::render_adaptive(const Scene &scene, int sensor_id, IntC idx_,FloatC pdf) const {
    PSDR_ASSERT_MSG(scene.is_ready(), "Input scene must be configured!");
    PSDR_ASSERT_MSG(sensor_id >= 0 && sensor_id < scene.m_num_sensors, "Invalid sensor id!");
    // scene->m_gradient_distrb.sample()

    std::cout << "Entering render_adaptive " << std::endl;
    const RenderOption &opts = scene.m_opts;
    const int num_pixels = opts.cropwidth*opts.cropheight;
    Spectrum<true> result = zero<Spectrum<true>>(num_pixels);
    FloatC weight_map = zero<FloatC>(num_pixels);
    if ( likely(opts.spp > 0) ) {
        int64_t num_samples = static_cast<int64_t>(slices(idx_))*opts.spp;
        PSDR_ASSERT(num_samples <= std::numeric_limits<int>::max());
        // std::cout<<"sample number values: "<<num_samples<<std::endl;
        
        IntC idx = zero<IntC>(num_samples);
        IntC mappingIdx = arange<IntC>(slices(idx_));
        // set_slices(mappingIdx,slices(idx_));

        
        for (int i=0;i<opts.spp;i++){
            scatter_add(idx,idx_,mappingIdx * opts.spp + i); // Map idx_ to sample domain
        }

        // Int<true> idx = arange<Int<true>>(num_samples);
        // if ( likely(opts.spp > 1) ) idx /= opts.spp;
        Vector2f<true> samples_base = gather<Vector2f<true>>(meshgrid(arange<Float<true>>(opts.cropwidth),
                                                        arange<Float<true>>(opts.cropheight)),
                                                        IntD(idx));

        Sampler sampler;
        sampler.seed(arange<IntC>(slices(idx)));
        Vector2f<true> samples;
        samples = (samples_base + sampler.next_2d<true>())
                                / ScalarVector2f(opts.cropwidth, opts.cropheight);
        // if(opts.debug == 1){
        //     samples = (samples_base)
        //                             / ScalarVector2f(opts.cropwidth, opts.cropheight);
        // }
        
        Ray<true> camera_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);
        Spectrum<true> value = Li(scene, sampler, camera_ray, true, sensor_id);
        std::cout << "hmean(hmean(value)) " << hmean(hmean(value)) << std::endl;
        masked(value, ~enoki::isfinite<Spectrum<true>>(value)) = 0.f;
        scatter_add(result, value, IntD(idx));
        scatter_add(weight_map, 1.0f / pdf, idx_);
        }
        if ( likely(opts.spp > 1) ) {
            result /= static_cast<float>(opts.spp);
        }
    return std::make_pair(result,weight_map);
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
        Vector2f<ad> samples_base = gather<Vector2f<ad>>(meshgrid(arange<Float<ad>>(opts.cropwidth),
                                                        arange<Float<ad>>(opts.cropheight)),
                                                        idx);

        Vector2f<ad> samples;
        samples = (samples_base + scene.m_samplers[0].next_2d<ad>())
                                / ScalarVector2f(opts.cropwidth, opts.cropheight);
        // if(opts.debug == 1){
        //     samples = (samples_base)
        //                             / ScalarVector2f(opts.cropwidth, opts.cropheight);
        // }
        
        Ray<ad> camera_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);
        Spectrum<ad> value = Li(scene, scene.m_samplers[0], camera_ray, true, sensor_id);
        masked(value, ~enoki::isfinite<Spectrum<ad>>(value)) = 0.f;
        scatter_add(result, value, idx);
        }
        if ( likely(opts.spp > 1) ) {
            result /= static_cast<float>(opts.spp);
        }
    return result;
}

// void Integrator::render_primary_edges(const Scene &scene, int sensor_id, SpectrumD &result) const {
//     const RenderOption &opts = scene.m_opts;

//     std::cout << "[var135] Render primary visibility edges " << std::endl;
//     const Sensor *sensor = scene.m_emitters_sensors[0];
//     if ( sensor->m_enable_edges ) {
//         PrimaryEdgeSample edge_samples = sensor->sample_primary_edge(scene.m_samplers[1].next_1d<false>());
//         MaskC valid = (edge_samples.idx >= 0);
// #ifdef PSDR_PRIMARY_EDGE_VIS_CHECK
//         IntersectionC its = scene.ray_intersect<false>(edge_samples.ray_c);
//         valid &= ~its.is_valid();
//         std::cout << "hmmm " << std::endl;
// #endif
//         std::cout << "count(valid) " << count(valid) << std::endl;
//         SpectrumC delta_L = Li(scene, scene.m_samplers[1], edge_samples.ray_n, valid, sensor_id) - 
//                             Li(scene, scene.m_samplers[1], edge_samples.ray_p, valid, sensor_id);
                            
//                             // estimate subgradient
//         SpectrumD value = edge_samples.x_dot_n*SpectrumD(delta_L/edge_samples.pdf);
//         masked(value, ~enoki::isfinite<SpectrumD>(value)) = 0.f;
//         if ( likely(opts.sppe > 1) ) {
//             value /= static_cast<float>(opts.sppe);
//         }

//         // TODO: Normal
//         // value -= detach(value);
//         // Debugging silhouette detection
//         value = SpectrumD(1000.0f);

//         scatter_add(result, value, IntD(edge_samples.idx), valid);
//     }
// }

void Integrator::render_primary_edges(const Scene &scene, int sensor_id, SpectrumD &result) const {
    const RenderOption &opts = scene.m_opts;

    const Sensor *sensor = scene.m_sensors[sensor_id];
    if ( sensor->m_enable_edges ) {
        PrimaryEdgeSample edge_samples = sensor->sample_primary_edge(scene.m_samplers[1].next_1d<false>());
        MaskC valid = (edge_samples.idx >= 0);
#ifdef PSDR_PRIMARY_EDGE_VIS_CHECK
        IntersectionC its = scene.ray_intersect<false>(edge_samples.ray_c);
        valid &= ~its.is_valid();
        std::cout << "hmmm " << std::endl;
#endif
        std::cout << "[render_primary_edges] count(valid) " << count(valid) << std::endl;
        SpectrumC delta_L = Li(scene, scene.m_samplers[1], edge_samples.ray_n, valid, sensor_id) - 
                            Li(scene, scene.m_samplers[1], edge_samples.ray_p, valid, sensor_id);
                            
                            // estimate subgradient
        SpectrumD value = edge_samples.x_dot_n*SpectrumD(delta_L/edge_samples.pdf);
        masked(value, ~enoki::isfinite<SpectrumD>(value)) = 0.f;
        if ( likely(opts.sppe > 1) ) {
            value /= static_cast<float>(opts.sppe);
        }

        // TODO: Normal
        value -= detach(value);
        // Debugging silhouette detection
        // value = SpectrumD(1000.0f);

        scatter_add(result, value, IntD(edge_samples.idx), valid);
    }
}

    template std::tuple<BSDFSampleC,BSDFSampleC,Float<true>> Integrator::sample_boundary<true>(const Scene &scene, const Ray<true> &camera_ray) const;
    template std::tuple<BSDFSampleC,BSDFSampleC,Float<false>> Integrator::sample_boundary<false>(const Scene &scene, const Ray<false> &camera_ray) const;
} // namespace psdr
