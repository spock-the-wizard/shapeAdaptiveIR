#include <misc/Exception.h>
#include <psdr/core/cube_distrb.h>
#include <psdr/core/ray.h>
#include <psdr/core/intersection.h>
#include <psdr/core/sampler.h>
#include <psdr/core/transform.h>
#include <psdr/bsdf/bsdf.h>
#include <psdr/emitter/emitter.h>
#include <psdr/scene/scene.h>
#include <psdr/shape/mesh.h>
#include <psdr/sensor/perspective.h>
#include <psdr/integrator/direct.h>
#include <psdr/sensor/sensor.h>

#include<psdr/bsdf/vaesub.h>
#include <enoki/special.h>
#include <enoki/random.h>

#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#   include <cxxabi.h>
#endif
#include <memory>
#include <string>
#include <cstdlib>

#include <iostream>
#include <fstream>

namespace psdr
{

template <bool ad>
static inline Float<ad> mis_weight(const Float<ad> &pdf1, const Float<ad> &pdf2) {
    Float<ad> w1 = sqr(pdf1), w2 = sqr(pdf2);
    return w1/(w1 + w2);
}

template <typename T,bool ad>
void print_masked_view(Array<T> array, Mask<ad> mask,int count = 5){
    std::cout << "Print Masked view " << std::endl;
    for(int i=0;i<slices(array);i++){
        if(mask[i]){
            std::cout << array[i] << std::endl;
            count--;
        }
        if(count==0)
            break;
    }
}

DirectIntegrator::~DirectIntegrator() {
    for ( auto *item : m_warpper ) {
        if ( item != nullptr ) delete item;
    }
}


DirectIntegrator::DirectIntegrator(int bsdf_samples, int light_samples) : m_bsdf_samples(bsdf_samples), m_light_samples(light_samples) {
    PSDR_ASSERT((bsdf_samples >= 0) && (light_samples >= 0) && (bsdf_samples + light_samples > 0));
}


SpectrumC DirectIntegrator::Li(const Scene &scene, Sampler &sampler, const RayC &ray, MaskC active, int sensor_id) const {
    return __Li<false>(scene, sampler, ray, active);
}


SpectrumD DirectIntegrator::Li(const Scene &scene, Sampler &sampler, const RayD &ray, MaskD active, int sensor_id) const {
    return __Li<true>(scene, sampler, ray, active);
}

SpectrumC DirectIntegrator::Li_shape(const Scene &scene, Sampler &sampler, const IntersectionC &its, MaskC active, int sensor_id) const {
    return __Li_shape<false>(scene, sampler, its, active);
}


SpectrumD DirectIntegrator::Li_shape(const Scene &scene, Sampler &sampler, const IntersectionD&its, MaskD active, int sensor_id) const {
    return __Li_shape<true>(scene, sampler, its, active);
}


template <bool ad>
Spectrum<ad> DirectIntegrator::__Li_shape(const Scene &scene, Sampler &sampler, const Intersection<ad> &its, Mask<ad> active) const {
    // std::cout << "its.poly_coeff " << its.poly_coeff << std::endl;
    std::cout << "ad " << ad << std::endl;
    std::cout<<"rendering ... "<<std::endl;

    // Intersection<ad> its = scene.ray_intersect<ad>(ray, active);
    // std::cout<<"intersection ... "<<std::endl;

    active &= its.is_valid();

    Spectrum<ad> result = zero<Spectrum<ad>>();
    BSDFArray<ad> bsdf_array = its.shape->bsdf(active);

    Mask<ad> maskl = Mask<ad>(active);
    Vector3f<ad> lightpoint = (scene.m_emitters[0]->sample_position(Vector3f<ad>(1.0f), Vector2f<ad>(1.0f), maskl)).p;


    BSDFSample<ad> bs;
    bs = bsdf_array->sample(&scene, its, sampler.next_nd<8, ad>(), active);
    Mask<ad> active1 = active && bs.is_valid;

    Vector3f<ad> wo = lightpoint - bs.po.p;
    Float<ad> dist_sqr = squared_norm(wo);
    Float<ad> dist = safe_sqrt(dist_sqr);
    wo /= dist;

    bs.wo = bs.po.sh_frame.to_local(wo);
    bs.po.wi = bs.wo;
    Ray<ad> ray1(bs.po.p, wo, dist);
    Intersection<ad> its1 = scene.ray_intersect<ad, ad>(ray1, active1);

    // Note the negation as we are casting rays TO the light source.
    // Occlusion indicates Invisibility
    active1 &= !its1.is_valid(); 

    Spectrum<ad> bsdf_val;
    if constexpr ( ad ) {
        bsdf_val = bsdf_array->eval(its, bs, active1);
    } else {
        bsdf_val = bsdf_array->eval(its, bs, active1);
    }
    
    Float<ad> pdfpoint = bsdf_array->pdfpoint(its, bs, active1);
    Spectrum<ad> Le;

    if constexpr ( ad ) {
        Le = scene.m_emitters[0]->eval(bs.po, active1);
    }
    else{
        Le = scene.m_emitters[0]->eval(bs.po, active1);
    }

    masked(result, active1) +=  bsdf_val * Le / detach(pdfpoint); //Spectrum<ad>(detach(dd));  //bsdf_val;//detach(Intensity) / 
    return result;
}



template <bool ad>
Spectrum<ad> DirectIntegrator::__Li(const Scene &scene, Sampler &sampler, const Ray<ad> &ray, Mask<ad> active) const {
    std::cout << "ad " << ad << std::endl;
    std::cout<<"rendering ... "<<std::endl;

    auto active_tmp1 = active;
    Intersection<ad> its = scene.ray_intersect<ad>(ray, active);
    std::cout<<"intersection ... "<<std::endl;

    active &= its.is_valid();

    Spectrum<ad> result = zero<Spectrum<ad>>();
    BSDFArray<ad> bsdf_array = its.shape->bsdf(active);

    Mask<ad> maskl = Mask<ad>(active);
    Vector3f<ad> lightpoint = (scene.m_emitters[0]->sample_position(Vector3f<ad>(1.0f), Vector2f<ad>(1.0f), maskl)).p;

    BSDFSample<ad> bs;
    bs = bsdf_array->sample(&scene, its, sampler.next_nd<8, ad>(), active);
    Mask<ad> active1 = active && bs.is_valid;

    Vector3f<ad> wo = lightpoint - bs.po.p;
    Float<ad> dist_sqr = squared_norm(detach(wo));
    Float<ad> dist = safe_sqrt(dist_sqr);
    wo = wo / dist;

    // Directly connect to point light
    bs.wo = bs.po.sh_frame.to_local(wo);
    bs.po.wi = bs.wo;
    Ray<ad> ray1(bs.po.p, wo, dist);
    Intersection<ad> its1 = scene.ray_intersect<ad, ad>(ray1, active1);

    // Note the negation as we are casting rays TO the light source.
    // Occlusion indicates Invisibility
    active1 &= !its1.is_valid(); 

    Spectrum<ad> bsdf_val;
    if constexpr ( ad ) {
        bsdf_val = bsdf_array->eval(its, bs, active1);
    } else {
        bsdf_val = bsdf_array->eval(its, bs, active1);
    }
    
    Float<ad> pdfpoint = bsdf_array->pdfpoint(its, bs, active1);
    Spectrum<ad> Le;
    // Spectrum<ad> Le_inv;

    if constexpr ( ad ) {
        Le = scene.m_emitters[0]->eval(bs.po, active1);
    }
    else{
        Le = scene.m_emitters[0]->eval(bs.po, active1);
    }

    // masked(result, active1) +=  bsdf_val * detach(Le)/ detach(pdfpoint); //Spectrum<ad>(detach(dd));  
    masked(result, active1) +=  bsdf_val * Le / detach(pdfpoint); //Spectrum<ad>(detach(dd));  
    // masked(result, active1) +=  bsdf_val * detach(Le) / pdfpoint;
    return result;
}


void DirectIntegrator::preprocess_secondary_edges(const Scene &scene, int sensor_id, const ScalarVector4i &reso, int nrounds) {
    PSDR_ASSERT(nrounds > 0);
    PSDR_ASSERT_MSG(scene.is_ready(), "Scene needs to be configured!");

    if ( static_cast<int>(m_warpper.size()) != scene.m_num_sensors )
        m_warpper.resize(scene.m_num_sensors, nullptr);

    if ( m_warpper[sensor_id] == nullptr )
        m_warpper[sensor_id] = new HyperCubeDistribution3f();
    auto warpper = m_warpper[sensor_id];

    warpper->set_resolution(head<3>(reso));
    int num_cells = warpper->m_num_cells;
    const int64_t num_samples = static_cast<int64_t>(num_cells)*reso[3];
    PSDR_ASSERT(num_samples <= std::numeric_limits<int>::max());

    IntC idx = divisor<int>(reso[3])(arange<IntC>(num_samples));
    Vector3iC sample_base = gather<Vector3iC>(warpper->m_cells, idx);

    Sampler sampler;
    sampler.seed(arange<UInt64C>(num_samples));

    FloatC result = zero<FloatC>(num_cells);
    for ( int j = 0; j < nrounds; ++j ) {
        SpectrumC value0;
        // std::tie(std::ignore, value0) = eval_secondary_edge<false>(scene, *scene.m_sensors[sensor_id],
        std::cout << "Preprocess edges " << std::endl;
        std::tie(std::ignore, value0) = eval_secondary_edge<false>(scene, *scene.m_sensors[sensor_id],
                                                                   (sample_base + sampler.next_nd<3, false>())*warpper->m_unit);
        masked(value0, ~enoki::isfinite<SpectrumC>(value0)) = 0.f;
        if ( likely(reso[3] > 1) ) {
            value0 /= static_cast<float>(reso[3]);
        }
        //PSDR_ASSERT(all(hmin(value0) > -Epsilon));
        scatter_add(result, hmax(value0), idx);
    }
    if ( nrounds > 1 ) result /= static_cast<float>(nrounds);
    warpper->set_mass(result);

    cuda_eval(); cuda_sync();
}
void DirectIntegrator::render_secondary_edgesC(const Scene &scene, int sensor_id, SpectrumC &result) const {
    
    std::cout << "render_secondary_edges " << std::endl;
    const RenderOption &opts = scene.m_opts;

    Vector3fC sample3 = scene.m_samplers[2].next_nd<3, false>();
    FloatC pdf0 = (m_warpper.empty() || m_warpper[sensor_id] == nullptr) ?
                  1.f : m_warpper[sensor_id]->sample_reuse(sample3);
    // FIXME: eval_boundary_edge
    auto [idx, value] = eval_boundary_edge<false>(scene, *scene.m_sensors[sensor_id], sample3);
    // auto [idx, value] = eval_secondary_edge<false>(scene, *scene.m_sensors[sensor_id], sample3);
    // IntC range = hmax(idx);
    // IntC counts = zero<IntC>(range[0]);
    // auto ones = full<IntC>(1);
    // set_slices(ones,slices(idx));
    // scatter_add(counts,ones,idx,idx >=0);
    // IntC countIdx = gather<IntC>(counts,idx,idx>=0);
    // countIdx[idx<0] = 1;
    // value /= countIdx;
    
    masked(value, ~enoki::isfinite<SpectrumC>(value)) = 0.f;
    masked(value, pdf0 > Epsilon) /= pdf0;
    if ( likely(opts.sppse > 1) ) {
        value /= (static_cast<float>(opts.sppse) / static_cast<float>(opts.cropwidth * opts.cropheight)); /// 
    }
    scatter_add(result, value, idx, idx >= 0);
}


void DirectIntegrator::render_secondary_edges(const Scene &scene, int sensor_id, SpectrumD &result) const {
    
    std::cout << "render_secondary_edges " << std::endl;
    const RenderOption &opts = scene.m_opts;

    Vector3fC sample3 = scene.m_samplers[2].next_nd<3, false>();
    FloatC pdf0 = (m_warpper.empty() || m_warpper[sensor_id] == nullptr) ?
                  1.f : m_warpper[sensor_id]->sample_reuse(sample3);
    auto [idx, value] = eval_boundary_edge<true>(scene, *scene.m_sensors[sensor_id], sample3);
    // auto [idx, value] = eval_secondary_edge<true>(scene, *scene.m_sensors[sensor_id], sample3);
    
    // FIXME: here
    IntC range = hmax(idx);
    if (range[0] >= 0){
        IntC counts = zero<IntC>(range[0]);
        auto ones = full<IntC>(1);
        set_slices(ones,slices(idx));
        scatter_add(counts,ones,idx,idx >=0);
        IntC countIdx = gather<IntC>(counts,idx,idx>=0);
        countIdx[idx<0] = 1;
        value /= countIdx;
    }

    
    masked(value, ~enoki::isfinite<SpectrumD>(value)) = 0.f;
    masked(value, pdf0 > Epsilon) /= pdf0;
    if ( likely(opts.sppse > 1) ) {
        value /= (static_cast<float>(opts.sppse) / static_cast<float>(opts.cropwidth * opts.cropheight)); /// 
    }

    scatter_add(result, value, IntD(idx), idx >= 0);
}

IntC DirectIntegrator::_compress(IntC input, MaskC mask) const {
    IntC output;
    // output.set_slices(slices(input))
    // int *ptr = slice_ptr(output, 0);
    output = input.compress_(mask);

    // uint64_t count = compress(ptr, input, mask);
    // output.set_slices(count);
    // std::cout<<"slices "<<slices(output)<<" before: "<<slices(input)<<std::endl;
    return output;
}

template <bool ad>
void DirectIntegrator::eval_secondary_edge_bssrdf(const Scene &scene, const IntersectionC &its, const Sensor &sensor, const Vector3fC &sample3, SpectrumD &result) const {
    std::cout << "eval_secondary_edge_bssrdf "<< std::endl;
    // sample p0 on edge    
    IntersectionC camera_its = its;
    BoundarySegSampleDirect bss = scene.sample_boundary_segment_direct(sample3);
    MaskC valid = bss.is_valid;

    const Vector3fC &_p0    = detach(bss.p0);
    Vector3fC       _dir    = normalize(bss.p2 - _p0), 
                    _p2     = Vector3fC(0.0f);
    // &_p2    = bss.p2,
    set_slices(_p2, slices(bss.p0));
    _p2 = _p2 + bss.p2;
              
    // check visibility between edge point and light point.
    IntersectionC _its2;
    TriangleInfoD tri_info;
    _its2 = scene.ray_intersect<false>(RayC(_p0, _dir), valid);
    valid &= (~_its2.is_valid());

    // trace in another direction to find shadow edge (record the triangle it intersect)
      
    IntersectionC its1 = scene.ray_intersect<false>(RayC(_p0, -_dir), valid, &tri_info);  //scene.ray_intersect<true, false>(RayD(_p0, normalize(_p0 - bss.p2)), MaskD(valid));
    valid &= its1.is_valid();

    // compute idx for matrix entry
    IntC cameraIdx = arange<IntC>(slices(camera_its));
    IntC edgeIdx = arange<IntC>(slices(valid));

    // remove the invalid entry
    if(!any(camera_its.is_valid()) || ! any(valid)) return;
    IntC validCameraSampleIdx = cameraIdx.compress_(camera_its.is_valid());
    IntC validEdgeSampleIdx = edgeIdx.compress_(valid);

    // std::cout<<"valid edge sample "<<slices(validEdgeSampleIdx)<<std::endl;
    // std::cout<<"valid camera sample "<<slices(validCameraSampleIdx)<<std::endl;


    IntersectionC cameraIts = gather<IntersectionC>(camera_its, validCameraSampleIdx);
    IntersectionC edgeItsC = gather<IntersectionC>(its1, IntC(validEdgeSampleIdx));
    // IntersectionC edgeItsC = detach(edgeItsD);
    TriangleInfoD triangleInfo = gather<TriangleInfoD>(tri_info, IntD(validEdgeSampleIdx));

    Vector3fD p0d = gather<Vector3fD>(bss.p0, IntD(validEdgeSampleIdx));
    Vector3fC p2 = gather<Vector3fC>(_p2, validEdgeSampleIdx);
    std::cout<<"slices p2 "<<slices(p2)<<"slices bss.p2"<<slices(bss.p2)<<"slices p0d"<<slices(p0d)<<std::endl;
    Vector3fC p0 = gather<Vector3fC>(_p0, validEdgeSampleIdx);
    Vector3fC dir = gather<Vector3fC>(_dir, validEdgeSampleIdx);
    Vector3fC bssN = gather<Vector3fC>(bss.n, validEdgeSampleIdx);
    Vector3fC bssE = gather<Vector3fC>(bss.edge, validEdgeSampleIdx);
    Vector3fC bssE2 = gather<Vector3fC>(bss.edge2, validEdgeSampleIdx);
    Vector3fC &p1 = edgeItsC.p;
    FloatC bssPDF = gather<FloatC>(bss.pdf, validEdgeSampleIdx);
    MaskC edgeValid = gather<MaskC>(valid, validEdgeSampleIdx);
    
    
    // compute jacobian that transform from x_b to x_d
    // J = d L(x_d) / d L(x_b) = |x_d - light source| / | x_b - light_source | * sinphi / sinphi2
    FloatC      dist    = norm(p2 - p1),
                dist2   = norm(p0 - p2),
                cos2    = abs(dot(edgeItsC.sh_frame.n, dir));
    Vector3fC   e       = cross(bssE, -dir);
    FloatC      sinphi  = norm(e);
    Vector3fC   proj    = normalize(cross(e, edgeItsC.sh_frame.n));
    FloatC      sinphi2 = norm(cross(-dir, proj));
    FloatC      base_v  =   ( dist / dist2)*(sinphi/sinphi2); //dist2 * dist2 * 
    e = normalize(e);

    edgeValid &= (sinphi > Epsilon) && (sinphi2 > Epsilon);

    // connect to camera all ray
    int64_t numIntersections = slices(cameraIts);
    int64_t numEdgesamples = slices(edgeItsC);
    int64_t num_entry = numIntersections * numEdgesamples;

    IntC cameraSampleMatrixIdx   = arange<IntC>(num_entry);
    IntC sidx   = arange<IntC>(num_entry);

    if (likely(numEdgesamples > 1)){
        cameraSampleMatrixIdx /= numEdgesamples;
        sidx /= numEdgesamples;
        sidx *= numEdgesamples;
    }

    IntC edgeSampleMatrixIdx = arange<IntC>(num_entry);
    edgeSampleMatrixIdx = edgeSampleMatrixIdx - sidx;
    

    IntersectionC   mCameraIts = gather<IntersectionC>(cameraIts, cameraSampleMatrixIdx);
    IntC            rIdx = gather<IntC>(validCameraSampleIdx, cameraSampleMatrixIdx);

    // IntersectionD       mEdgeItsD = gather<IntersectionD>(edgeItsD, IntD(edgeSampleMatrixIdx));
    IntersectionC      mEdgeItsC = gather<IntersectionC>(edgeItsC, edgeSampleMatrixIdx);
    FloatC            mBase_v = gather<FloatC>(base_v, edgeSampleMatrixIdx);
    Vector3fC           projs = gather<Vector3fC>(proj, edgeSampleMatrixIdx);
    Vector3fC              es = gather<Vector3fC>(e, edgeSampleMatrixIdx);
    Vector3fC           _dirs = gather<Vector3fC>(dir, edgeSampleMatrixIdx);
    FloatC              dists = gather<FloatC>(dist, edgeSampleMatrixIdx);
    FloatC               pdfs = gather<FloatC>(bssPDF, edgeSampleMatrixIdx);
    Vector3fD               p0s = gather<Vector3fD>(p0d, IntD(edgeSampleMatrixIdx));
    // std::cout<<" slices p2 "<<slices(p2)<<std::endl;
    Vector3fC               p2s = gather<Vector3fC>(p2, edgeSampleMatrixIdx);
    MaskC        active_entry = gather<MaskC>(edgeValid, edgeSampleMatrixIdx);
    Vector3fC          edge2s = gather<Vector3fC>(bssE2, edgeSampleMatrixIdx);
    Vector3fC              ns = gather<Vector3fC>(edgeItsC.n, edgeSampleMatrixIdx);
    TriangleInfoD triInfos = gather<TriangleInfoD>(triangleInfo, IntD(edgeSampleMatrixIdx));

    // compute bssrdf
    BSDFSampleC bs;
    BSDFArrayC bsdf_array = mEdgeItsC.shape->bsdf(active_entry);
    bs.po = mEdgeItsC;
    bs.wo = mEdgeItsC.wi;
    bs.is_sub = bsdf_array->hasbssdf();
    bs.is_valid = mEdgeItsC.is_valid();
    bs.pdf = FloatC(1.0f);

    active_entry &= mCameraIts.is_valid();
    active_entry &= bs.is_valid;
    SpectrumC bsdf_val = bsdf_array->eval(mCameraIts, bs, active_entry);
    SpectrumC Le = scene.m_emitters[0]->eval(bs.po, active_entry);
    SpectrumC v0 = ( Le * bsdf_val * mBase_v / (pdfs) ) & active_entry;

    if constexpr (ad) {
        Vector3fC nc = normalize(cross(ns, projs));
        v0 *= sign(dot(es, edge2s))*sign(dot(es, nc));
        
        const Vector3fD &v = triInfos.p0,
                        &e1 = triInfos.e1,
                        &e2 = triInfos.e2;

        // std::cout<<"slices p2s"<<slices(p2s)<<std::endl;
        // RayD shadow_ray(Vector3fD(p2s), normalize(-_dirs));
        // Vector2fD material_uv;
        // std::tie(material_uv, std::ignore) = ray_intersect_triangle<true>(v, e1, e2, shadow_ray);
        // Vector3fD point1 = bilinear<true>(detach(v), detach(e1), detach(e2), material_uv);

        RayD shadow_ray2(Vector3fD(p2s), normalize(p0s - p2s));
        Vector2fD material_uv2;
        // FloatD ts;
        std::tie(material_uv2, std::ignore) = ray_intersect_triangle<true>(v, e1, e2, shadow_ray2);
        Vector3fD point2 = bilinear<true>(detach(v), detach(e1), detach(e2), material_uv2);
        
        
        SpectrumD value = (dot(Vector3fD(nc), point2)) * SpectrumD(v0) & active_entry;
        scatter_add(result, value - detach(value), IntD(rIdx));
    }
}
// template <bool ad>
// std::pair<IntC, Spectrum<ad>> DirectIntegrator::eval_secondary_edge(const Scene &scene, const Sensor &sensor, const Vector3fC &sample3) const {
//     std::cout << "eval_secondary_edge" << std::endl;
//     BoundarySegSampleDirect bss = scene.sample_boundary_segment_direct(sample3);
//     MaskC valid = bss.is_valid;

//     // _p0 on a face edge, _p2 on an emitter
//     const Vector3fC &_p0    = detach(bss.p0);
//     Vector3fC       &_p2    = bss.p2,
//                     _dir    = normalize(_p2 - _p0);                    

//     // check visibility between _p0 and _p2
//     IntersectionC _its2;
//     TriangleInfoD tri_info;
//     // if constexpr ( ad ) {
//         // _its2 = scene.ray_intersect<false>(RayC(_p0, _dir), valid);
//     // } else {
//     _its2 = scene.ray_intersect<false>(RayC(_p0, _dir), valid);
//     // }
//     valid &= (~_its2.is_valid());// && norm(_its2.p - _p2) < ShadowEpsilon;

//     // trace another ray in the opposite direction to complete the boundary segment (_p1, _p2)
//     // IntersectionD its4;
//      IntersectionC _its1;
//     if constexpr ( ad ) {
//         _its1 = scene.ray_intersect<false>(RayC(_p0, -_dir), MaskC(valid), &tri_info);
//         // its4 = scene.ray_intersect<true, true>(RayD(_p0, normalize(_p0 - bss.p2)), MaskD(valid));
//     } else {
//         _its1 = scene.ray_intersect<false>(RayC(_p0, -_dir), MaskC(valid));
//     }
//     // IntersectionC _its1 = detach(its1);
//     valid &= _its1.is_valid();
//     // std::cout<<" valid : "<<any(valid)<<std::endl;
//     // Vector3fC &_p1 = _its1.p;

//     // Start: Xi Deng added:  sample a point for bssrdf 
//     BSDFSampleC bs1, bs2;
//     BSDFArrayC bsdf_array = _its1.shape->bsdf(valid);
//     bs1 = bsdf_array->sample(&scene, _its1, scene.m_samplers[2].next_nd<8, false>(), valid);
//     // std::cout << "direct:400" <<bs1.rgb << std::endl;
//     Vector3fC &_p1 = bs1.po.p;
    
//     // std::cout<<bs1.po.abs_prob<<std::endl;
//     // End: Xi Deng added
//     bs2.po = _its1;
//     // std::cout<<bs2.po.abs_prob<<std::endl;
//     bs2.wo = _its1.sh_frame.to_local(_dir);
//     bs2.is_valid = _its1.is_valid();
//     bs2.pdf = bss.pdf;
//     bs2.is_sub = bsdf_array->hasbssdf();
//     bs2.rgb_rv = bs1.rgb_rv;
//     // TODO: this part needs rgb mapping
//     valid &= bs1.po.is_valid();
//     // std::cout<<" valid : "<<any(valid)<<std::endl;

//     // project _p1 onto the image plane and compute the corresponding pixel id
//     SensorDirectSampleC sds = sensor.sample_direct(_p1);
//     valid &= sds.is_valid;
//     // std::cout<<" valid : "<<any(valid)<<std::endl;
//     // trace a camera ray toward _p1 in a differentiable fashion
//     Ray<ad> camera_ray;
//     Intersection<ad> its2;
//     if constexpr ( ad ) {
//         camera_ray = sensor.sample_primary_ray(Vector2fD(sds.q));
//         its2 = scene.ray_intersect<true, false>(camera_ray, valid);
//         valid &= its2.is_valid() && norm(detach(its2.p) - _p1) < ShadowEpsilon;
//     } else {
//         camera_ray = sensor.sample_primary_ray(sds.q);
//         its2 = scene.ray_intersect<false>(camera_ray, valid);
//         valid &= its2.is_valid() && norm(its2.p - _p1) < ShadowEpsilon;
//     }
//     // std::cout<<" valid : "<<any(valid)<<std::endl;
//     // calculate base_value
//     FloatC      dist    = norm(_p2 - _its1.p),
//                 part_dist    = norm(_p2 - _p0);
//                 // cos2    = abs(dot(_its1.n, -_dir));
//     Vector3fC   e       = cross(bss.edge, _dir); // perpendicular to edge and ray
//     FloatC      sinphi  = norm(e);
//     Vector3fC   proj    = normalize(cross(e, _its1.n)); // on the light plane
//     FloatC      sinphi2 = norm(cross(_dir, proj));
//     FloatC      base_v  = (dist/part_dist)*(sinphi/sinphi2);
//     valid &= (sinphi > Epsilon) && (sinphi2 > Epsilon);
//     // std::cout<<" valid : "<<any(valid)<<std::endl;
//     // evaluate BSDF at _p1
    
//     Vector3fC d0;
//     if constexpr ( ad ) {
//         d0 = -detach(camera_ray.d);
//     } else {
//         d0 = -camera_ray.d;
//     }
//     // std::cout<<" print direction : "<<d0<<std::endl;
//     // std::cout<<" print frame : "<<bs1.po.sh_frame.n<<std::endl;
//     // std::cout<<" print frame : "<<bs1.wo<<std::endl;

//     bs1.wo =  bs1.po.sh_frame.to_local(d0);
//     bs1.po.wi =  bs1.po.sh_frame.to_local(d0);
//     SpectrumC bsdf_val = bsdf_array->eval(bs1.po, bs2, valid);
//     FloatC pdfpoint = bsdf_array->pdfpoint(bs2.po, bs1, valid);
//     SpectrumC value0 = (bsdf_val*scene.m_emitters[0]->eval(bs2.po, valid)*(base_v * sds.sensor_val/bss.pdf/pdfpoint)) & valid;
    
//     // std::cout << "pdfpoint " << pdfpoint << std::endl;
//     // std::cout << "value0 " << value0 << std::endl;
    
//     if constexpr ( ad ) {
//         Vector3fC n = normalize(cross(_its1.n, proj));
//         value0 *= sign(dot(e, bss.edge2))*sign(dot(e, n));

//         // Start Xi Deng modified
//         const Vector3fD &v0 = tri_info.p0,
//                         &e1 = tri_info.e1,
//                         &e2 = tri_info.e2;

//         Vector3fD myp2 = bss.p2 * 0.0f;
//         set_slices(myp2, slices(bss.p0));
//         myp2 = myp2 + bss.p2;
//         Vector2fD uv;
        
//         RayD shadow_ray(myp2, normalize(bss.p0 - myp2));
//         std::tie(uv, std::ignore) = ray_intersect_triangle<true>(v0, e1, e2, shadow_ray);
//         Vector3fD u2 = bilinear<true>(detach(v0), detach(e1), detach(e2), uv);
//         // End Xi Deng added

//         // Start: Joon modified
//         // auto v_b = bsdf_array -> getKernelEps();
//         // auto v_b = 1/s;
//         // SpectrumD result = (SpectrumD(value0) * v_b) & valid;

//         // End : Joon modified
//         SpectrumD result = (SpectrumD(value0) * dot(Vector3fD(n), u2)) & valid;
//         return { select(valid, sds.pixel_idx, -1), result - detach(result) };
//     } else {
//         return { -1, value0 };
//     }
// }

// template <bool ad>
// std::pair<IntC, Spectrum<ad>> DirectIntegrator::eval_boundary_edge(const Scene &scene, const Sensor &sensor, const Vector3fC &sample3) const {
//     std::cout << "eval_boundary_edge" << std::endl;
    
//     // S
    
//     cout value0 = SpectrumD(1000.0f);
//     return { -1, value0 };
// }
void DirectIntegrator::__render_boundary(const Scene &scene, int sensor_id, SpectrumD &result) const {
    const bool ad = true;
    PSDR_ASSERT_MSG(scene.is_ready(), "Input scene must be configured!");
    PSDR_ASSERT_MSG(sensor_id >= 0 && sensor_id < scene.m_num_sensors, "Invalid sensor id!");

    const RenderOption &opts = scene.m_opts;
    const int num_pixels = opts.cropwidth*opts.cropheight;
    // Spectrum<ad> result = zero<Spectrum<ad>>(num_pixels);
    if ( likely(opts.sppse > 0) ) {
        int64_t num_samples = static_cast<int64_t>(num_pixels)*opts.sppse;

        Int<ad> idx = arange<Int<ad>>(num_samples);
        if ( likely(opts.sppse> 1) ) idx /= opts.sppse;
        // std::cout<<"idx: "<<slices(idx)<<std::endl;
        Vector2f<ad> samples_base = gather<Vector2f<ad>>(meshgrid(arange<Float<ad>>(opts.cropwidth),
                                                        arange<Float<ad>>(opts.cropheight)),
                                                        idx);
        // std::cout << "samples_base " << samples_base << std::endl;


        Vector2f<ad> samples = (samples_base + scene.m_samplers[2].next_2d<ad>())
                                / ScalarVector2f(opts.cropwidth, opts.cropheight);
        Ray<ad> camera_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);

        Intersection<ad> its = scene.ray_intersect<ad>(camera_ray, true);
        MaskD active = its.is_valid();
        // active &= its.is_valid();

        IntC cameraIdx = arange<IntC>(slices(camera_ray.o));
        IntC validCameraSampleIdx = cameraIdx.compress_(active);
        // Invalidate everything
        // active &= 1.0f == 0.0f;
        // std::cout << "active " << active << std::endl;

        // std::cout << "validCameraSampleIdx " << validCameraSampleIdx << std::endl;
        // std::cout << "validCameraSampleIdx[0] " << validCameraSampleIdx[0] << std::endl;
        // // IntC firstValidIdx(validCameraSampleIdx[0]);
        // active |= cameraIdx < validCameraSampleIdx[0]+1000;
        // std::cout << "count(active) " << count(active) << std::endl;
        // scatter_add(active,true,firstValidIdx);
        its.p = slice(its.p,validCameraSampleIdx[0]);
        its.sh_frame.s = slice(its.sh_frame.s,validCameraSampleIdx[0]);
        its.sh_frame.t = slice(its.sh_frame.t,validCameraSampleIdx[0]);
        its.sh_frame.n = slice(its.sh_frame.n,validCameraSampleIdx[0]);
        std::cout << "its.p " << its.p << std::endl;
        set_slices(its.p,slices(active));
        set_slices(its.sh_frame.s,slices(active));
        set_slices(its.sh_frame.t,slices(active));
        set_slices(its.sh_frame.n,slices(active));
        std::cout << "its.p " << its.p << std::endl;

        // Sample disk points
        Vector2f<ad> disk_samples = scene.m_samplers[2].next_2d<ad>();
        disk_samples = float(M_SQRT2) * erfinv(2.f*disk_samples-1.f);
        Vector3f<ad> vx = its.sh_frame.s;
        Vector3f<ad> vy = its.sh_frame.t;
        Vector3f<ad> vz = its.sh_frame.n;
        // auto r = vz;
        // TODO: tmp
        float r = 0.05f;
        float l = 0.02f;
        // Float<ad> l = 2.0f * sqrt(rmax * rmax - r * r);
        Ray<ad> ray2(its.p + r * (vx * disk_samples.x() + vy * disk_samples.y()) - vz * 0.5f * l, vz, l);
        // SensorDirectSampleC sds = sensor->sample_direct(bs1.po.p);
        auto sample = scene.m_samplers[2].next_nd<8,ad>();
        // auto bs1_samples = scene.m_samplers[2].next_nd<8, false>();
        Intersection<ad> its2 = scene.ray_all_intersect<ad, ad>(ray2, active, sample, 5);
        active &= its2.is_valid();
        // std::cout << "HIII " << std::endl;

        auto sensor = scene.m_sensors[sensor_id];
        auto sds= sensor->sample_direct(detach(its2.p));
        auto sds2 = sensor->sample_direct(detach(its.p));
        SpectrumD value(1000.000f);
        value = value & active;
        // std::cout << "count(value) " << count(active) << std::endl;
        // std::cout << "slices(value) " << slices(value) << std::endl;

        SpectrumD red(1000.000f,0.0f,0.f);
        // scatter_add(result,value,IntD(sds.pixel_idx));
        scatter_add(result,red,IntD(sds2.pixel_idx));
        // Compute dot(gradF, v)

        // Assign test weight kernel

        // (1) Compute V

        // (2) Compute divV


    //     BSDFArray<ad> bsdf_array = its.shape->bsdf(active);
    //     Mask<ad> maskl = Mask<ad>(active);
    //     Vector3f<ad> lightpoint = (scene.m_emitters[0]->sample_position(Vector3f<ad>(1.0f), Vector2f<ad>(1.0f), maskl)).p;

    //     BSDFSample<ad> bs;
    //     bs = bsdf_array->sample(&scene, its, sampler.next_nd<8, ad>(), active);
    //     Mask<ad> active1 = active && bs.is_valid;

    //     Vector3f<ad> wo = lightpoint - bs.po.p;
    //     Float<ad> dist_sqr = squared_norm(detach(wo));
    //     Float<ad> dist = safe_sqrt(dist_sqr);
    //     wo = wo / dist;

    // // Directly connect to point light
    // bs.wo = bs.po.sh_frame.to_local(wo);
    // bs.po.wi = bs.wo;
    // Ray<ad> ray1(bs.po.p, wo, dist);
    // Intersection<ad> its1 = scene.ray_intersect<ad, ad>(ray1, active1);

    // // Note the negation as we are casting rays TO the light source.
    // // Occlusion indicates Invisibility
    // active1 &= !its1.is_valid(); 

    // Spectrum<ad> bsdf_val;
    // if constexpr ( ad ) {
    //     bsdf_val = bsdf_array->eval(its, bs, active1);
    // } else {
    //     bsdf_val = bsdf_array->eval(its, bs, active1);
    // }
    
    // Float<ad> pdfpoint = bsdf_array->pdfpoint(its, bs, active1);
    // Spectrum<ad> Le;
    // // Spectrum<ad> Le_inv;

    // if constexpr ( ad ) {
    //     Le = scene.m_emitters[0]->eval(bs.po, active1);
    // }
    // else{
    //     Le = scene.m_emitters[0]->eval(bs.po, active1);
    // }

    // // masked(result, active1) +=  bsdf_val * detach(Le)/ detach(pdfpoint); //Spectrum<ad>(detach(dd));  
    // masked(result, active1) +=  bsdf_val * Le / detach(pdfpoint); //Spectrum<ad>(detach(dd));  
    // // masked(result, active1) +=  bsdf_val * detach(Le) / pdfpoint;
    // return result;




        // Spectrum<ad> value = Li(scene, scene.m_samplers[0], camera_ray, true, sensor_id);
        // masked(value, ~enoki::isfinite<Spectrum<ad>>(value)) = 0.f;
        // FIXME: tmp setting
        // scatter_add(result, value, idx);
        }

        if ( likely(opts.sppse > 1) ) {
            result /= static_cast<float>(opts.sppse);
        }
            

}

template <bool ad>
std::pair<IntC, Spectrum<ad>> DirectIntegrator::eval_boundary_edge(const Scene &scene, const Sensor &sensor, const Vector3fC &sample3) const {
    std::cout << "eval_boundary_edge" << std::endl;

    BoundarySegSampleDirect bss;
    SecondaryEdgeInfo sec_edge_info;
    std::tie(bss,sec_edge_info) = scene.sample_boundary_segment_v2(sample3);
    MaskC valid = bss.is_valid;

    // _p0 on a face edge, _p2 on an emitter
    const Vector3fC &_p0    = detach(bss.p0);
    Vector3fC       &_p2    = bss.p2,
                    _dir    = normalize(_p2 - _p0);                    
    set_slices(_p2,slices(_p0));

    TriangleInfoD tri_info;
    IntersectionC _its1;
    // Light ray intersection on edge
    if constexpr ( ad ) {
        _its1 = scene.ray_intersect<false>(RayC(_p2, -_dir), MaskC(valid), &tri_info);
    } else {
        _its1 = scene.ray_intersect<false>(RayC(_p2, -_dir), valid);
    }
    // Discard 1) empty space and 2) occluded regions
    valid &=(_its1.is_valid());
    valid &= norm(_its1.p - _p0) < 0.02f;
    // std::cout << "[2] valid count" << count(valid) <<std::endl;
    SensorDirectSampleC sds3= sensor.sample_direct(_p0);
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
    SensorDirectSampleC sds2 = sensor.sample_direct(_its1.p);
    RayC cam_ray = sensor.sample_primary_ray(sds2.q);
    _its1.wi = _its1.sh_frame.to_local(-cam_ray.d);

    // Sample camera ray
    bs1D = bsdf_arrayD->sample(&scene, IntersectionD(_its1), Vector8fD(bs1_samples),  MaskD(valid));
    bs1 = detach(bs1D);
    // Ray<ad> camera_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);
    SensorDirectSampleC sds = sensor.sample_direct(bs1.po.p);
    
    // xo is a valid VAE output and seen from camera
    valid &= bs1.po.is_valid();
    // valid &= bs1.is_sub;
    valid &= sds.is_valid;
    // std::cout << "[3] valid count" << count(valid) <<std::endl;

    Ray<ad> camera_ray;
    // Camera its ray
    Intersection<ad> its2;
    if constexpr ( ad ) {
        camera_ray = sensor.sample_primary_ray(Vector2fD(sds.q));
        its2 = scene.ray_intersect<true, false>(camera_ray, valid);
        valid &= its2.is_valid() && norm(detach(its2.p) - bs1.po.p) < ShadowEpsilon;
    } else {
        camera_ray = sensor.sample_primary_ray(sds.q);
        its2 = scene.ray_intersect<false>(camera_ray, valid);
        valid &= its2.is_valid() && norm(its2.p - bs1.po.p) < ShadowEpsilon;
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
    SpectrumC value0 = (bsdf_val * scene.m_emitters[0]->eval(bs2.po, valid)*(sds.sensor_val/pdfpoint/bss.pdf)) & valid;
    // SpectrumC value0 = (scene.m_emitters[0]->eval(bs2.po, valid)*(sds.sensor_val/bss.pdf/pdfpoint)) & valid;
    // FIXME: tmp disabled 
    FloatC cos_theta_o = FrameC::cos_theta(bs2.wo);
    value0 /= cos_theta_o;
    // FIXME: tmp
    // SpectrumC value0 = (bsdf_val*scene.m_emitters[0]->eval(bs2.po, valid)*(sds.sensor_val/bss.pdf/pdfpoint)) & valid;

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
    // std::cout << "cos_theta_o " << cos_theta_o << std::endl;
    
    // edge normal
    // deltaG = (G-)-(G+)
    
    FloatC cos_n = dot(edge_n1,_dir);
    FloatC cos_p = dot(edge_n0,_dir);
    FloatC  delta = cos_n & vis_n;
    delta -= cos_p & vis_p;
    // valid &= delta > 0.5f;
    
    std::cout << "_its1.n " << _its1.n << std::endl;
    std::cout << "edge_n0 " << edge_n0 << std::endl;
    std::cout << "edge_n1 " << edge_n1 << std::endl;
    
    // PSDR_ASSERT(all((~valid) | (valid&(_its1.n == edge_n0)) | (valid&(_its1.n == edge_n1))));
    // _its1.n is closer to edge_n0 than edge_n1
    // FIXME: checking
    auto isN0 = (dot(_its1.n,edge_n0) - dot(_its1.n,edge_n1));
    Vector3fC n = select(isN0 > 0.0f,line_n0,line_n1);
    // Vector3fC n = normalize(cross(_its1.n,bss.edge));
    delta *= sign(isN0);
    // delta *= -sign(dot(_its1.n,edge_n0)-dot(_its1.n,edge_n1));
    value0 *= delta;
    // valid &= (vis_n^vis_p); // | (dot(edge_n0,edge_n1) < 0.1f);
    valid &= (vis_n^vis_p) | (dot(edge_n0,edge_n1) < 0.1f);
    // valid &= (vis_n^vis_p);
    // FIXME: add only sharp edges
    // valid &= (vis_n & vis_p); //
    // valid &= (vis_n ^ vis_p);
    // valid &= (~isnan(edge_n0)) && (~isnan(edge_n1));

    // // Reparameterize xi
    // // xi = xo + x_VAE / scalefactor
    FloatD kernelEps = bsdf_arrayD -> getKernelEps(IntersectionD(_its1),FloatD(bs1_samples[5]));
    FloatD maxDist = sqrt(kernelEps);
    // Vector3fC xo = detach(bs1D.po.p);
    // Vector3fC xvae = (bs2.po.p - bs1.po.p) / detach(maxDist);
    // FIXME: max
    // xvae = select(xvae > 1.0f, 1.0f, xvae);
    // FIXME: inverse xvae
    // xvae = normalize(xvae);
    // xvae = 1/xvae;

    //FIXME: derivative of inverse 
    // Vector3fD xi = Vector3fD(xo) + Vector3fD(xvae) * maxDist;
    auto xrel = bs2.po.p - bs1.po.p;
    Vector3fD xi = Vector3fD(xrel) / maxDist;

    // FIXME: testing
    // SpectrumD result = dot(Vector3fD(n), xi) & valid;
    // SpectrumD result = (SpectrumD(value0) * maxDist) & valid;
    auto grad = dot(Vector3fD(n),xi);
    // float threshold = 1.0f;
    // std::cout << "count(all(value0<threshold) " << count(value0.x()<threshold) << std::endl;
    // std::cout << "count(all(grad<threshold) " << count(grad < threshold) << std::endl;
    // // value0 = select(abs(value0) < threshold, value0, sign(grad)*threshold);
    // // grad = select(abs(grad)<threshold, grad, sign(grad)*threshold);
    // value0 = select(value0< threshold, value0, threshold);
    // grad = select(grad<threshold, grad, threshold);
    // value0 = select(value0 >-threshold, value0, -threshold);
    // grad = select(grad>-threshold, grad, -threshold);
    SpectrumD result = (SpectrumD(value0) * grad) & valid;
    // SpectrumD result = (SpectrumD(value0) * grad) & valid;
    // PSDR_ASSERT(all(all(result<threshold)));

    if constexpr(ad) {
        return { select(valid, sds.pixel_idx, -1), result - detach(result)};
    }
    else{
        return { select(valid, sds.pixel_idx, -1), SpectrumC(1000.0f) };
    }
}

template <bool ad>
std::pair<IntC, Spectrum<ad>> DirectIntegrator::eval_secondary_edge(const Scene &scene, const Sensor &sensor, const Vector3fC &sample3) const {
    std::cout << "eval_secondary_edge" << std::endl;
    
    BoundarySegSampleDirect bss = scene.sample_boundary_segment_direct(sample3);
    MaskC valid = bss.is_valid;
    // std::cout << "bss.p0 " << bss.p0 << std::endl;

    // std::cout << "count(valid) " << count(valid) << std::endl;
    // _p0 on a face edge, _p2 on an emitter
    const Vector3fC &_p0    = detach(bss.p0);
    Vector3fC       &_p2    = bss.p2,
                    _dir    = normalize(_p2 - _p0);                    

    // check visibility between _p0 and _p2
    IntersectionC _its2;
    TriangleInfoD tri_info;
    _its2 = scene.ray_intersect<false>(RayC(_p0, _dir), valid);
    // Discard ones not visible to light source
    valid &= (~_its2.is_valid());// && norm(_its2.p - _p2) < ShadowEpsilon;

    // trace another ray in the opposite direction to complete the boundary segment (_p1, _p2)
     IntersectionC _its1;
    if constexpr ( ad ) {
        _its1 = scene.ray_intersect<false>(RayC(_p0, -_dir), MaskC(valid), &tri_info);
    } else {
        _its1 = scene.ray_intersect<false>(RayC(_p0, -_dir), MaskC(valid));
    }
    // std::cout << "count(!isnan(_its1.sh_frame.n)) " << count(!isnan(hsum(_its1.sh_frame.n))) << std::endl;
    // std::cout << "count(_its1.is_valid()) " << count(_its1.is_valid()) << std::endl;
    // std::cout << "count(!isnan(_its1.sh_frame.n)) " << count(!isnan(hsum(_its1.sh_frame.n))) << std::endl;

    // TODO: filter out valid intersections (xi) and repeat N times using gather/scatter
    // IntC xiIdx = arange<IntC>(slices(_its1));
    // IntC validXiIdx= cameraIdx.compress_(valid);

    // Intersection<ad>
    // Spectrum<ad> masked_val = gather<Spectrum<ad>>(bs.po.p,validRayIdx);
    // std::cout << "bsdf_val " << masked_val << std::endl;

    


    valid &= _its1.is_valid();
    // Discard backfacing intersections
    valid &= dot(_its1.n,-_dir) <= 0;

    BSDFSampleC bs1, bs2;
    BSDFSampleD bs1D;
    BSDFArrayC bsdf_array = _its1.shape->bsdf(valid);
    BSDFArrayD bsdf_arrayD = BSDFArrayD(bsdf_array);
    
    // Start: Xi Deng added:  sample a point for bssrdf 
    auto bs1_samples = scene.m_samplers[2].next_nd<8, false>();
    // FIXME: set to sample ONLY SSS
    // auto tmp_sample = full<FloatC>(1.1f);
    // set_slices(tmp_sample,slices(bs1_samples));
    // bs1_samples[0] = tmp_sample;
    // std::cout << "bs1_samples.x() " << bs1_samples.x() << std::endl;
    bs1 = bsdf_array->sample(&scene, _its1, bs1_samples, valid);
    // std::cout << "valid " << count(valid) << std::endl;
    // std::cout << "Sampled bs is_sub?" <<count(bs1.is_sub) << std::endl;
    bs1D = BSDFSampleD(bs1);
    
    // Select valid rays visible to camera
    SensorDirectSampleC sds = sensor.sample_direct(bs1.po.p);
    // Discard intersections invisible to current camera
    valid &= bs1.po.is_valid() && sds.is_valid;
    Vector3fC &_p1 = bs1.po.p;
    // std::cout << "After VAE Sampling count(valid) " << count(valid) << std::endl;

    Ray<ad> camera_ray;
    Intersection<ad> its2;
    if constexpr ( ad ) {
        camera_ray = sensor.sample_primary_ray(Vector2fD(sds.q));
        its2 = scene.ray_intersect<true, false>(camera_ray, valid);
        valid &= its2.is_valid() && norm(detach(its2.p) - _p1) < ShadowEpsilon;
    } else {
        camera_ray = sensor.sample_primary_ray(sds.q);
        its2 = scene.ray_intersect<false>(camera_ray, valid);
        valid &= its2.is_valid() && norm(its2.p - _p1) < ShadowEpsilon;
    }
    std::cout << "Discard occluded samples: count(valid) " << count(valid) << std::endl;

    // End: Xi Deng added
    // trace a camera ray toward _p1 in a differentiable fashion
    // calculate base_value
    FloatC      dist    = norm(_p2 - _its1.p),
                part_dist    = norm(_p2 - _p0);
                // cos2    = abs(dot(_its1.n, -_dir));
    Vector3fC   e       = cross(bss.edge, _dir); // perpendicular to edge and ray
    FloatC      sinphi  = norm(e);
    Vector3fC   proj    = normalize(cross(e, _its1.n)); // on the light plane
    FloatC      sinphi2 = norm(cross(_dir, proj));
    FloatC      base_v  = (dist/part_dist)*(sinphi/sinphi2);
    valid &= (sinphi > Epsilon) && (sinphi2 > Epsilon);
    // evaluate BSDF at _p1
    
    // std::cout << "count(valid) " << count(valid) << std::endl;
    
    Vector3fC d0;
    if constexpr ( ad ) {
        d0 = -detach(camera_ray.d);
    } else {
        d0 = -camera_ray.d;
    }
    bs1.wo =  bs1.po.sh_frame.to_local(d0);
    bs1.po.wi = bs1.wo;

    // Needed for eval_sub
    bs2.po = _its1;
    bs2.wo = _its1.sh_frame.to_local(_dir);
    bs2.po.wi = bs2.wo;
    bs2.is_valid = _its1.is_valid();
    bs2.pdf = bss.pdf;
    // bs2.is_sub = bsdf_array->hasbssdf();
    // FIXME
    // Originally sample xi from xo using VAE.
    // Here, we directly sample xi from edges and sample xo using VAE.
    bs2.is_sub = bs1.is_sub;
    bs2.po.abs_prob = bs1.po.abs_prob;
    bs2.rgb_rv = bs1.rgb_rv;
    bs2.po.J = FloatC(1.0f);
    // bs2.is_sub = bsdf_array->hasbssdf();

    SpectrumC bsdf_val = bsdf_array->eval(bs1.po, bs2, valid);
    FloatC pdfpoint = bsdf_array->pdfpoint(bs1.po, bs2, valid);
    // SpectrumC bsdf_val = bsdf_array->eval(bs2.po, bs1, valid);
    // FloatC pdfpoint = bsdf_array->pdfpoint(bs2.po, bs1, valid);
    // FIXME: checking div by zero 
    // SpectrumC value0 = (bsdf_val*scene.m_emitters[0]->eval(bs2.po, valid)*( sds.sensor_val/bss.pdf/pdfpoint)) & valid;
    SpectrumC value0 = (bsdf_val*scene.m_emitters[0]->eval(bs2.po, valid)*(base_v * sds.sensor_val/bss.pdf/pdfpoint)) & valid;
    // FIXME: tmp check if bsdf is issue
    // std::cout << "count(bs2.is_sub) " << count(bs2.is_sub) << std::endl;
    // value0[~bs2.is_sub] = 0.0f;
    // std::cout << "debug0" << hsum(hsum(bsdf_val)) << std::endl;
    // std::cout << "debug0" << hsum(hsum(scene.m_emitters[0]->eval(bs2.po, valid))) << std::endl;
    // std::cout << "hmax(hmax(value0)) " << hmax(hmax(value0)) << std::endl;
    // std::cout << "hmin(hmin(value0)) " << hmin(hmin(value0)) << std::endl;
    // std::cout << "debug1" << hsum(hsum(base_v * sds.sensor_val/bss.pdf/pdfpoint))<<std::endl;
    // SpectrumC value0 = (bsdf_val*scene.m_emitters[0]->eval(bs2.po, valid)*() & valid;
    
    Vector3fC n = normalize(cross(_its1.n, proj));
    value0 *= sign(dot(e, bss.edge2))*sign(dot(e, n));
    
    if constexpr ( ad ) {

        // Reparameterize xi
        // xi = xo + x_VAE / scalefactor
        FloatD kernelEps = bsdf_arrayD -> getKernelEps(IntersectionD(_its1),FloatD(bs1_samples[5]));
        // std::cout << "kernelEps " << hsum(kernelEps) << std::endl;
        FloatD maxDist = sqrt(kernelEps);
        Vector3fC xo = detach(bs1D.po.p);
        Vector3fC xvae = (bs2.po.p - bs1.po.p) / detach(maxDist);
        Vector3fD xi = Vector3fD(xo) + Vector3fD(xvae) * maxDist;
        // std::cout << "hsum(value0) " << hsum(value0) << std::endl;
        // std::cout << "hsum(value0) " << hsum(hsum(value0)) << std::endl;


        // // dot(dx_p,n) = d/dx (dot(xp,n))
        // FIXME
        SpectrumD result = (SpectrumD(value0) * dot(Vector3fD(n), xi)) & valid;
        // SpectrumD result= (SpectrumD(value0) * maxDist) & valid;
        // SpectrumD result = (dot(Vector3fD(n), xi)) & valid;


        return { select(valid, sds.pixel_idx, -1), result - detach(result) };
    } else {
        // auto test = scene.m_emitters[0]->eval(bs2.po, valid) * bsdf_val / bss.pdf / pdfpoint * sds.sensor_val;
        // auto test = bsdf_val;
        // auto test = base_v;
        // auto test = 1/bss.pdf;
        // auto test = bsdf_val/pdfpoint;
        
        // auto test_max = hmax(hmax(test));
        // return { select(valid, sds.pixel_idx, -1), test / test_max};
        // return { select(valid, sds.pixel_idx, -1), test };
        return { select(valid, sds.pixel_idx, -1), SpectrumC(1000.0f)};
    }
}

} // namespace psdr
