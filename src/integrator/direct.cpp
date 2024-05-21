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
    // Spectrum<ad> sigmaT = bsdf_array->getSigmaT(its);
    // set_requires_gradient(sigmaT);
    // std::cout << "sigmaT " << sigmaT << std::endl;

    Mask<ad> maskl = Mask<ad>(active);
    Vector3f<ad> lightpoint = (scene.m_emitters[0]->sample_position(Vector3f<ad>(1.0f), Vector2f<ad>(1.0f), maskl)).p;

    BSDFSample<ad> bs;
    bs = bsdf_array->sample(&scene, its, sampler.next_nd<8, ad>(), active);
    // if constexpr(ad){
    //     if(count(active)>0){
    //         backward(bs.po.p[0],true);
    //         backward(bs.po.p[1],true);
    //         backward(bs.po.p[2],true);
    //         std::cout << "gradient(sigmaT) " << gradient(sigmaT) << std::endl;
    //     }
    // }
    // std::cout << "bs.velocity " << bs.velocity << std::endl;
    // std::cout << "bs.maxDist " << bs.maxDist << std::endl;
    
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

    BSDFSample<ad> bsM = bs;
    bsM.po = bs.pair;
    auto active2 = active && bs.pair.is_valid();
    Vector3f<ad> woM= lightpoint - bsM.po.p;
    Float<ad> dist_sqrM = squared_norm(detach(woM));
    Float<ad> distM = safe_sqrt(dist_sqrM);
    woM = woM / distM;

    bsM.wo = bsM.po.sh_frame.to_local(woM);
    bsM.po.wi = bsM.wo;
    // bsM.wo =  bsM.po.sh_frame.to_local(d0);
    // bsM.po.wi =  bsM.wo;
    // bsM.po = bs.pair;
    // bsM.wo =  bsM.po.sh_frame.to_local(_dir);
    // bsM.po.wi =  bsM.wo;
    bsM.rgb_rv = bs.rgb_rv;
    bsM.po.abs_prob = bs.po.abs_prob;
    bsM.is_sub = bs.is_sub;
    // std::cout << "bsM.is_sub " << bsM.is_sub << std::endl;
    // NOTE: don't forget this part...!! Important for eval_sub or ... error bomb will strike
    bsM.is_valid = bs.pair.is_valid();
    bsM.pdf = bs.pdf;
    bsM.po.J = bs.po.J;
    bsM.maxDist = bs.maxDist;
    bsM.velocity = bs.velocity;
    
    Ray<ad> ray1M(bsM.po.p, woM, distM);
    // TODO: check active
    Intersection<ad> its1M = scene.ray_intersect<ad, ad>(ray1M, active2);
    active2 &= !its1M.is_valid();

    Spectrum<ad> LeM;
    Spectrum<ad> bsdf_valM;
    if constexpr(ad){
        bsdf_valM = bsdf_array->eval(its, bsM, active2);
        LeM = scene.m_emitters[0]->eval(bsM.po, active2);
        // std::cout << "hsum(bsdf_valM) " << hsum(bsdf_valM) << std::endl;
        // std::cout << "count(active1) " << count(active1) << std::endl;
        // std::cout << "count(active2) " << count(active2) << std::endl;
    }
    // masked(result, active1) +=  bsdf_val * detach(Le)/ detach(pdfpoint); //Spectrum<ad>(detach(dd));  
    // Note: detached bsdf_val
    // masked(result, active1) +=  Le * detach(bsdf_val) / detach(pdfpoint); //Spectrum<ad>(detach(dd));  
    // FIXME: MATE2
    masked(result, active1) +=  detach(Le) * detach(bsdf_val) / detach(pdfpoint); //Spectrum<ad>(detach(dd));  
    
    if constexpr(ad){
        std::cout << "[Var 208] Testin Mate Idea2 - Before anything discontinuous happens...";
        auto pair_f = (LeM*bsdf_valM); // &active2;
        auto po_f = (Le*bsdf_val); // &active1;
        pair_f = pair_f & active2;
        po_f = po_f & active1;
        auto diff = (pair_f - po_f) / pdfpoint;
        diff = detach(diff);
        masked(result,active) += diff * bs.maxDist - diff * detach(bs.maxDist);
    }
    

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
template <bool ad>
Mask<ad> isDiff(Vector3f<ad> n1, Vector3f<ad> n2, Mask<ad> vis1, Mask<ad> vis2){
    float thresG = 0.4f;
    auto diff = ~((dot(n1,n2) > thresG) & (dot(n1,n2) > 0)) | (vis1 ^ vis2);
    // auto diff = (vis1 ^ vis2);
    
    return diff;
}

// template <bool ad>
std::tuple<Vector3fD,Vector3fD,FloatD,Vector3fD,Vector3fD,Vector3fD> DirectIntegrator::_sample_boundary_3(const Scene &scene, RayC camera_ray, IntC idx, bool debug) const {
    const bool ad = true;
        IntersectionD its_ = scene.ray_intersect<true>(RayD(camera_ray), true);
        IntersectionC its = detach(its_);
        // IntersectionC its = scene.ray_intersect<false>(camera_ray, true);
        // its_.wi = its_.sh_frame.to_local(-(camera_ray.d));
        its.wi = its.sh_frame.to_local(-camera_ray.d);
        MaskC active = its.is_valid();
        // IntersectionD its2 = scene.ray_intersect<true>(RayD(camera_ray), true);

        BSDFSampleC bs;
        BSDFSampleD bsD;
        BSDFArrayC bsdf_array = its.shape->bsdf(active);
        BSDFArrayD bsdf_arrayD = BSDFArrayD(bsdf_array);
        auto vaesub_bsdf = reinterpret_cast<VaeSub*>(bsdf_arrayD[0]);

        
        auto bs1_samples = scene.m_samplers[2].next_nd<8, true>();
        // auto bs1_samples = scene.m_samplers[2].next_nd<8, false>();
        // auto bs1_samples = Vector8fC(0.0f);
        // std::cout << "bs1_samples " << bs1_samples << std::endl;
        auto F = bsdf_array->getFresnel(its);
        FloatC weight_sub = 1.0f / (1.0f-F);
        // std::cout << "weight_sub " << weight_sub << std::endl;
        
        bs1_samples[0] = 1.1f;
        // bsD = bsdf_arrayD->sample(&scene, IntersectionD(its), Vector8fD(bs1_samples),  MaskD(active));
        // auto sigmaT = bsdf_arrayD->getSigmaT(its_);
        // set_requires_gradient(sigmaT);
        bsD = bsdf_arrayD->sample(&scene, its_, bs1_samples,  MaskD(active));
        // std::cout << "bsD.maxDist " << bsD.maxDist << std::endl;
        bs = detach(bsD);
        active &= bsD.po.is_valid();
        // std::cout << "bs.po.p " << bs.po.p << std::endl;

        Vector3f<ad> lightpoint = (scene.m_emitters[0]->sample_position(Vector3f<ad>(1.0f), Vector2f<ad>(1.0f), true)).p;
        Vector3f<ad> wo = lightpoint - bsD.po.p;
        Float<ad> dist_sqr = squared_norm(detach(wo));
        Float<ad> dist = safe_sqrt(dist_sqr);
        wo = wo / dist;

        bsD.wo = bsD.po.sh_frame.to_local(wo);
        bsD.po.wi = bsD.wo;
        Ray<ad> ray1(bsD.po.p, wo, dist);
        Intersection<ad> its1 = scene.ray_intersect<ad, ad>(ray1, MaskD(active));
        active &= !its1.is_valid();  // visible to light?
        

        Vector3fC wo2 = detach(lightpoint) - its.p;
        FloatC dist2 = norm(wo2);
        wo2 = wo2 / dist2;
        RayC ray2(its.p, wo2, dist2);
        IntersectionC its2 = scene.ray_intersect<false, false>(ray2, active); //MaskD(active));

        FloatD maxDist = bsD.maxDist;
        // ======================================================
        // Material point pK
        Vector3fD p = (bsD.po.p- its.p)/maxDist;
        // Vector3fD p = (Vector3fD(bs.po.p) - its.p)/maxDist;
        Vector3fD pK = detach(p); 
        set_requires_gradient(pK);
        // FloatD cosAngle = dot(vn,normalize(detach(pK)));
        
        
        // Vector3fD xK = vn*maxDist*cosAngle;
        // xK = xK - detach(xK) + bs.po.p; //detach(bsD.po.p);
        Vector3fD xK = pK * maxDist + its.p;
        bsD.po.p = xK;

        Spectrum<ad> bsdf_val;
        bsdf_val = bsdf_arrayD->eval(IntersectionD(its), bsD, MaskD(active));
        Float<ad> pdfpoint = bsdf_arrayD->pdfpoint(IntersectionD(its), bsD, active);
        Spectrum<ad> Le;
        Le = scene.m_emitters[0]->eval(bsD.po, MaskD(active));
        
        // Continuous velocity field
        FloatD B(0.0f);
        
        // Idea 1. Compute geometric term for all samples.
        // Find closest point with different geometric term
        
        const RenderOption &opts = scene.m_opts;
        const long unsigned int N = opts.sppse;
        IntC _idx = arange<IntC>(slices(its));
        const IntC sampleIdx = arange<IntC>(slices(its)) % N;
        // Index of boundary point in bsD.po.p
        IntD idx_list = zero<IntC>(slices(its)) - 1;
        // Maintain min and max distance, and whether a boundary point exists (isUpdated)
        FloatD distPairMin(5.0f);
        FloatD distPairMax(0.0f);
        MaskD isUpdated(false);
        
        // Create a circulation mask that ensures each neighbor appears only once for each id
        MaskC is_last_sample = (sampleIdx - (N-1) < 0.5f) && (sampleIdx + 2 - N > 0.5f);
        IntC idxMask = select(is_last_sample,_idx-(N-1),_idx+1);
        
        // Geometric term of xo
        auto xo_n = its.n;
        auto xo_vis = !its2.is_valid(); 

        // Geometric term of xi
        auto xi_n = bs.po.n;
        auto xi_vis = ~detach(its1).is_valid();
        auto xi_valid = bs.po.is_valid();

        auto diff_xi = isDiff<false>(xo_n,xi_n,xo_vis,xi_vis);
        
        // FIXME: testing with simple G function
        // auto G_diff = (xi_vis^xo_vis);
        // auto G_diff = (abs(dot(xi_n,xo_n)) < thresG) | (xi_vis^xo_vis);
        //
        // Iterate through all neighbor indices.
        for (int id=1;id<N;id++){
            
            _idx = gather<IntC>(_idx,idxMask);
            
            auto nei_n = gather<Vector3fC>(bs.po.n,_idx);
            auto nei_vis = gather<MaskC>(xi_vis,_idx);
            auto nei_valid = gather<MaskC>(xi_valid,_idx);
            
            // Is the neighbor a boundary candidate?
            // auto isDiff = (xi_vis ^ nei_vis) | dot(xi_n,nei_n) < thresG;
            // auto isCand = xi_valid & nei_valid & ((abs(dot(xi_n,nei_n)) < thresG) | (xi_vis^nei_vis));
            // auto isCand = MaskC(true);
            // MaskC isCand(true);
            // auto isDiff = ((abs(dot(xi_n,nei_n)) < thresG) | (xi_vis^nei_vis));
            // auto isDiff = nei_valid & ((abs(dot(xi_n,nei_n)) < thresG) | (xi_vis^nei_vis));
            auto diff_nei = isDiff<false>(xo_n,nei_n,xo_vis,nei_vis);
            auto isCand = xi_valid & nei_valid & (diff_nei ^ diff_xi);


            // If yes, add position to samples
            auto samples = gather<Vector3fC>(bs.po.p,_idx,isCand);
            auto _norm = norm(Vector3fD(samples) - Vector3fD(bs.po.p));
            
            // Update if necessary
            auto update = isCand && (distPairMin > _norm);
            distPairMin = select(update, _norm, distPairMin); //hmin(distPairMin,_norm);
            distPairMax = select(nei_valid && (_norm>distPairMax),_norm,distPairMax);
            idx_list = select(update, IntD(_idx), idx_list); //hmin(distPairMin,_norm);
            isUpdated = select(update, MaskD(true),isUpdated);
        }

        auto xB = gather<Vector3fC>(bs.po.p,detach(idx_list),detach(isUpdated));
        B = norm(Vector3fD(xB) - xK);
        B = select(isUpdated, B, detach(distPairMax));

        FloatD sigma = 0.003f;
        int a = 3;
        // auto r = squared_norm(xK - its.p);
        auto r_ = dot(pK,pK) / maxDist / maxDist; //xK-its.p,xK-its.p);
        FloatD D = 1.0f/sigma*pow(1-exp(-r_/sigma),a);
        // auto r = dot(pK,pK);
        // FloatD D = 1/sigma*1/exp(-r*r);
        FloatD weight = 1.0f/(B);
        // FloatD D = 1/(sigma * exp(-r));
        // FloatD weight = 1.0f/(D+B);
        // FloatD weight = sigma*(exp(-dot(pK,pK)/50.0f));
        FloatC W(0.0f);
        set_slices(W,slices(idx));
        scatter_add(W,detach(weight),idx);
        auto weight_sum = gather<FloatC>(W,idx);
        weight /= FloatD(weight_sum);
        masked(weight,~bs.po.is_valid() | ~active) = 0.0f;
        // // Interior
        // Vector3fD wLe = Le & active; 
        // // Boundary
        Vector3fD wLe = Le * bsdf_val * weight;
        // std::cout << "wLe " << wLe << std::endl;
        backward(wLe[0],true);
        backward(wLe[1],true);
        backward(wLe[2],true);
        Vector3fC d_wLe = gradient(pK);
        // std::cout << "d_wLe" << d_wLe <<std::endl;

        // Vector3fD vDis = p;
        // FIXME: Debug for viz
        Vector3fD vDis = bsD.velocity;
        // Boundary
        // std::cout << "count(~G_diff) " << count(~G_diff) << std::endl;
        // std::cout << "count(hsum(vDis)==0.0f) " << count(hsum(G_diff)==0.0f) << std::endl;
        std::cout << "vDis " << vDis << std::endl;

        auto dot_ = dot(vDis,d_wLe);
        // Detached bsdf_val
        // Vector3fD value = Vector3fD(detach(bsdf_val))* dot_; 
        Vector3fD value = dot_;
        
        // value = value / Vector3fD(weight_sum);
        // Disable fresnel weights for now... this causes silhouette edges to overshoot
        // value = value * weight_sub;
        value = value / detach(pdfpoint);
        value = value & active; // V
    
        
        return std::tie(xK,value,weight,vDis,d_wLe,dot_);
}
        



void DirectIntegrator::__render_boundary(const Scene &scene, int sensor_id, SpectrumD &result) const {
    const bool ad = true;
    PSDR_ASSERT_MSG(scene.is_ready(), "Input scene must be configured!");
    PSDR_ASSERT_MSG(sensor_id >= 0 && sensor_id < scene.m_num_sensors, "Invalid sensor id!");

    const RenderOption &opts = scene.m_opts;
    const int num_pixels = opts.cropwidth*opts.cropheight;
    // Spectrum<ad> result = zero<Spectrum<ad>>(num_pixels);
    if ( likely(opts.sppse > 0) ) {
        int64_t num_samples = static_cast<int64_t>(num_pixels)*opts.sppse;

        IntC idx = arange<Int<false>>(num_samples);
        if ( likely(opts.sppse> 1) ) idx /= opts.sppse;
        // std::cout<<"idx: "<<slices(idx)<<std::endl;
        Vector2fC samples_base = gather<Vector2f<false>>(meshgrid(arange<Float<false>>(opts.cropwidth),
                                                        arange<Float<false>>(opts.cropheight)),
                                                        idx);
        // std::cout << "samples_base " << samples_base << std::endl;


        Vector2fC samples = (samples_base + scene.m_samplers[2].next_2d<false>())
                                / ScalarVector2f(opts.cropwidth, opts.cropheight);
        RayC camera_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);

        Vector3fD pK,value;
        FloatD weight;
        std::tie(pK,value,weight,std::ignore,std::ignore,std::ignore) = _sample_boundary_3(scene,camera_ray,idx);
        // IntersectionC its = scene.ray_intersect<false>(camera_ray, true);
        // MaskC active = its.is_valid();

        // BSDFSampleC bs;
        // BSDFSampleD bsD;
        // BSDFArrayC bsdf_array = its.shape->bsdf(active);
        // BSDFArrayD bsdf_arrayD = BSDFArrayD(bsdf_array);
        // auto vaesub_bsdf = reinterpret_cast<VaeSub*>(bsdf_arrayD[0]);
        // auto bs1_samples = scene.m_samplers[2].next_nd<8, false>();
        // // TODO: account for bias with weight
        // bs1_samples[0] = 1.1f;
        // // auto weight_sub = (1-F);
        // bsD = bsdf_arrayD->sample(&scene, IntersectionD(its), Vector8fD(bs1_samples),  MaskD(active));
        // bs = detach(bsD);

        // Vector3f<ad> lightpoint = (scene.m_emitters[0]->sample_position(Vector3f<ad>(1.0f), Vector2f<ad>(1.0f), true)).p;
        // Vector3f<ad> wo = lightpoint - bsD.po.p;
        // Float<ad> dist_sqr = squared_norm(detach(wo));
        // Float<ad> dist = safe_sqrt(dist_sqr);
        // wo = wo / dist;

        // bsD.wo = bsD.po.sh_frame.to_local(wo);
        // bsD.po.wi = bsD.wo;
        // Ray<ad> ray1(bsD.po.p, wo, dist);
        // Intersection<ad> its1 = scene.ray_intersect<ad, ad>(ray1, MaskD(active));
        // active &= !its1.is_valid(); 

        // Vector3fD pK = detach(bsD.po.p);
        // set_requires_gradient(pK);
        // bsD.po.p = pK;

        // Spectrum<ad> bsdf_val;
        // bsdf_val = bsdf_arrayD->eval(IntersectionD(its), bsD, MaskD(active));
        // Float<ad> pdfpoint = bsdf_arrayD->pdfpoint(IntersectionD(its), bsD, active);
        // Spectrum<ad> Le;
        // Le = scene.m_emitters[0]->eval(bsD.po, MaskD(active));
        // // TODO: pdfpoint
        
        // // Continuous velocity field
        // FloatD kernelEps = bsdf_arrayD -> getKernelEps(IntersectionD(its),FloatD(bs1_samples[5]));
        // FloatD maxDist = sqrt(kernelEps);
        // // TODO: Boundary test function
        // FloatD B = abs(dot(its1.sh_frame.n,its1.wi));
        // masked(B, ~enoki::isfinite<FloatD>(B)) = 0.f;
        // std::cout << "B " << B << std::endl;
        // // FloatD B = 1.0
        // // TODO: distance function. Tmp setting sigma=1.0f
        // // FloatD sigma = maxDist;
        // FloatD sigma = 0.0003f;
        // int a = 3;
        // auto x_p = (pK- its.p)/maxDist;
        // auto r = norm(x_p);
        // FloatD D = 1.0f/sigma*pow(1-exp(-r*r/sigma),a);
        // // FloatD D = 1.0f/sigma*pow(1-exp(-r*r/sigma),a);
        // // FloatD weight = r;
        // FloatD weight = 1.0f/(D+B);
        
        // auto wLe = weight*Le;
        // // auto F = Le * detach(bsdf_val); /// detach(pdfpoint);
        // backward(wLe[0],true);
        // backward(wLe[1],true);
        // backward(wLe[2],true);
        // auto d_wLe = gradient(pK);
        // // std::cout << "gradient(pK) " << d_wLe <<std::endl;

        // // // Analytic gradient computed by hand...
        // // auto gradF = - 2.0f * (bsD.po.p - lightpoint) / dist_sqr * F;
        // // std::cout << "gradF " << gradF << std::endl;

        // // // std::cout << "weight " << weight << std::endl;
        // FloatD W(0.0f);
        // set_slices(W,slices(idx));
        // scatter_add(W,weight,IntD(idx));
        // auto weight_sum = gather<FloatD>(W,IntD(idx));

        // auto vDis = -(pK - its.p) / (maxDist * maxDist);
        // vDis = vDis & (!its1.is_valid()); // Visible points should be zero
        // // auto vCon = vDis * weight;
        // // vCon /= detach(weight_sum);

        // // The formula
        // auto value = bsdf_val / weight_sum * dot(vDis,d_wLe);
        // value = value & active; // V
        
        // std::cout << "vCon" << vCon << std::endl;
        // SpectrumD value = dot(Vector3fD(gradF),vCon);
        // // FIXME: tmp setting
        masked(value, ~enoki::isfinite<Spectrum<ad>>(value)) = 0.f;
        // std::cout << "value " << value << std::endl;
        scatter_add(result,value-detach(value),IntD(idx));
        }

        if ( likely(opts.sppse > 1) ) {
            result /= static_cast<float>(opts.sppse);
        }
}
// template <bool ad>
// std::pair<IntC, Spectrum<ad>> DirectIntegrator::eval_boundary_edge(const Scene &scene, const Sensor &sensor, const Vector3fC &sample3) const {
//     std::cout << "eval_boundary_edge" << std::endl;

//     BoundarySegSampleDirect bss;
//     SecondaryEdgeInfo sec_edge_info;
//     std::tie(bss,sec_edge_info) = scene.sample_boundary_segment_v2(sample3);
//     MaskC valid = bss.is_valid;

//     // _p0 on a face edge, _p2 on an emitter
//     const Vector3fC &_p0    = detach(bss.p0);
//     Vector3fC       &_p2    = bss.p2,
//                     _dir    = normalize(_p2 - _p0);                    
//     set_slices(_p2,slices(_p0));

//     TriangleInfoD tri_info;
//     IntersectionC _its1;
//     // Light ray intersection on edge
//     if constexpr ( ad ) {
//         _its1 = scene.ray_intersect<false>(RayC(_p2, -_dir), MaskC(valid), &tri_info);
//     } else {
//         _its1 = scene.ray_intersect<false>(RayC(_p2, -_dir), valid);
//     }
//     // Discard 1) empty space and 2) occluded regions
//     valid &=(_its1.is_valid());
//     valid &= norm(_its1.p - _p0) < 0.02f;
//     // std::cout << "[2] valid count" << count(valid) <<std::endl;
//     SensorDirectSampleC sds3= sensor.sample_direct(_p0);
//     // RayC cam_ray = sensor.sample_primary_ray(sds2.q);

//     BSDFSampleC bs1, bs2;
//     BSDFSampleD bs1D;
//     BSDFArrayC bsdf_array = _its1.shape->bsdf(valid);
//     BSDFArrayD bsdf_arrayD = BSDFArrayD(bsdf_array);
//     auto vaesub_bsdf = reinterpret_cast<VaeSub*>(bsdf_arrayD[0]);
//     auto bs1_samples = scene.m_samplers[2].next_nd<8, false>();
//     // TODO: account for bias with weight
//     bs1_samples[0] = 1.1f;
//     // auto weight_sub = (1-F);
//     SensorDirectSampleC sds2 = sensor.sample_direct(_its1.p);
//     RayC cam_ray = sensor.sample_primary_ray(sds2.q);
//     _its1.wi = _its1.sh_frame.to_local(-cam_ray.d);

//     // Sample camera ray
//     bs1D = bsdf_arrayD->sample(&scene, IntersectionD(_its1), Vector8fD(bs1_samples),  MaskD(valid));
//     bs1 = detach(bs1D);
//     // Ray<ad> camera_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);
//     SensorDirectSampleC sds = sensor.sample_direct(bs1.po.p);
    
//     // xo is a valid VAE output and seen from camera
//     valid &= bs1.po.is_valid();
//     // valid &= bs1.is_sub;
//     valid &= sds.is_valid;
//     // std::cout << "[3] valid count" << count(valid) <<std::endl;

//     Ray<ad> camera_ray;
//     // Camera its ray
//     Intersection<ad> its2;
//     if constexpr ( ad ) {
//         camera_ray = sensor.sample_primary_ray(Vector2fD(sds.q));
//         its2 = scene.ray_intersect<true, false>(camera_ray, valid);
//         valid &= its2.is_valid() && norm(detach(its2.p) - bs1.po.p) < ShadowEpsilon;
//     } else {
//         camera_ray = sensor.sample_primary_ray(sds.q);
//         its2 = scene.ray_intersect<false>(camera_ray, valid);
//         valid &= its2.is_valid() && norm(its2.p - bs1.po.p) < ShadowEpsilon;
//     }

//     Vector3fC d0;
//     if constexpr ( ad ) {
//         d0 = -detach(camera_ray.d);
//     } else {
//         d0 = -camera_ray.d;
//     }
//     bs1.wo =  bs1.po.sh_frame.to_local(d0);
//     bs1.po.wi =  bs1.wo;
//     bs2.po = _its1;
//     bs2.wo =  bs2.po.sh_frame.to_local(_dir);
//     bs2.po.wi =  bs2.wo;
//     bs2.rgb_rv = bs1.rgb_rv;
//     bs2.po.abs_prob = bs1.po.abs_prob;
//     bs2.is_sub = bsdf_array -> hasbssdf();
//     // NOTE: don't forget this part...!! 
//     // Needed for eval_sub
//     bs2.is_valid = _its1.is_valid();
//     bs2.pdf = bss.pdf;
//     bs2.po.J = FloatC(1.0f);

//     SpectrumC bsdf_val = bsdf_array->eval(bs1.po, bs2, valid);
//     FloatC pdfpoint = bsdf_array->pdfpoint(bs1.po, bs2, valid);
//     SpectrumC value0 = (bsdf_val * scene.m_emitters[0]->eval(bs2.po, valid)*(sds.sensor_val/pdfpoint/bss.pdf)) & valid;
//     // SpectrumC value0 = (scene.m_emitters[0]->eval(bs2.po, valid)*(sds.sensor_val/bss.pdf/pdfpoint)) & valid;
//     // FIXME: tmp disabled 
//     FloatC cos_theta_o = FrameC::cos_theta(bs2.wo);
//     value0 /= cos_theta_o;
//     // FIXME: tmp
//     // SpectrumC value0 = (bsdf_val*scene.m_emitters[0]->eval(bs2.po, valid)*(sds.sensor_val/bss.pdf/pdfpoint)) & valid;

//     Vector3fC edge_n0 = normalize(detach(sec_edge_info.n0));
//     Vector3fC edge_n1 = normalize(detach(sec_edge_info.n1));
//     // // FIXME: should this be dependent on relative position of light ray?
    
//     // Vector3fC line_n0 = normalize(cross(edge_n0,bss.edge));
//     // Vector3fC line_n1 = normalize(cross(edge_n1,bss.edge));
//     // FIXED
//     FloatC n0IsLeft = sign(dot(edge_n0,cross(bss.edge,bss.edge2)));
//     Vector3fC line_n0 = normalize(cross(edge_n0,bss.edge))*n0IsLeft;
//     Vector3fC line_n1 = -normalize(cross(edge_n1,bss.edge))*n0IsLeft;
    
//     // Vector3fC line_n0 = normalize(cross(-bss.edge,edge_n0));
//     // Vector3fC line_n1 = normalize(cross(-edge_n1,bss.edge));

//     // Vector3fC n = line_n0;
    
//     // Compute geometry term for left and right limits
//     float eps = 0.05f;
//     auto lightpoint = _p2;
//     Vector3fC pos_p = bs2.po.p + eps * line_n0;
//     Vector3fC pos_n = bs2.po.p + eps * line_n1;
//     Vector3fC dir_p = normalize(lightpoint - pos_p);
//     Vector3fC dir_n = normalize(lightpoint - pos_n);
//     IntersectionC its_p = scene.ray_intersect<false>(RayC(pos_p,dir_p),valid);
//     IntersectionC its_n = scene.ray_intersect<false>(RayC(pos_n,dir_n),valid);
//     MaskC vis_n = ~its_n.is_valid();
//     MaskC vis_p = ~its_p.is_valid();
//     // std::cout << "cos_theta_o " << cos_theta_o << std::endl;
    
//     // edge normal
//     // deltaG = (G-)-(G+)
    
//     FloatC cos_n = dot(edge_n1,_dir);
//     FloatC cos_p = dot(edge_n0,_dir);
//     FloatC  delta = cos_n & vis_n;
//     delta -= cos_p & vis_p;
//     // valid &= delta > 0.5f;
    
//     std::cout << "_its1.n " << _its1.n << std::endl;
//     std::cout << "edge_n0 " << edge_n0 << std::endl;
//     std::cout << "edge_n1 " << edge_n1 << std::endl;
    
//     // PSDR_ASSERT(all((~valid) | (valid&(_its1.n == edge_n0)) | (valid&(_its1.n == edge_n1))));
//     // _its1.n is closer to edge_n0 than edge_n1
//     // FIXME: checking
//     auto isN0 = (dot(_its1.n,edge_n0) - dot(_its1.n,edge_n1));
//     Vector3fC n = select(isN0 > 0.0f,line_n0,line_n1);
//     // Vector3fC n = normalize(cross(_its1.n,bss.edge));
//     delta *= sign(isN0);
//     // delta *= -sign(dot(_its1.n,edge_n0)-dot(_its1.n,edge_n1));
//     value0 *= delta;
//     // valid &= (vis_n^vis_p); // | (dot(edge_n0,edge_n1) < 0.1f);
//     valid &= (vis_n^vis_p) | (dot(edge_n0,edge_n1) < 0.1f);
//     // valid &= (vis_n^vis_p);
//     // FIXME: add only sharp edges
//     // valid &= (vis_n & vis_p); //
//     // valid &= (vis_n ^ vis_p);
//     // valid &= (~isnan(edge_n0)) && (~isnan(edge_n1));

//     // // Reparameterize xi
//     // // xi = xo + x_VAE / scalefactor
//     FloatD kernelEps = bsdf_arrayD -> getKernelEps(IntersectionD(_its1),FloatD(bs1_samples[5]));
//     FloatD maxDist = sqrt(kernelEps);
//     // Vector3fC xo = detach(bs1D.po.p);
//     // Vector3fC xvae = (bs2.po.p - bs1.po.p) / detach(maxDist);
//     // FIXME: max
//     // xvae = select(xvae > 1.0f, 1.0f, xvae);
//     // FIXME: inverse xvae
//     // xvae = normalize(xvae);
//     // xvae = 1/xvae;

//     //FIXME: derivative of inverse 
//     // Vector3fD xi = Vector3fD(xo) + Vector3fD(xvae) * maxDist;
//     auto xrel = bs2.po.p - bs1.po.p;
//     Vector3fD xi = Vector3fD(xrel) / maxDist;

//     // FIXME: testing
//     // SpectrumD result = dot(Vector3fD(n), xi) & valid;
//     // SpectrumD result = (SpectrumD(value0) * maxDist) & valid;
//     auto grad = dot(Vector3fD(n),xi);
//     // float threshold = 1.0f;
//     // std::cout << "count(all(value0<threshold) " << count(value0.x()<threshold) << std::endl;
//     // std::cout << "count(all(grad<threshold) " << count(grad < threshold) << std::endl;
//     // // value0 = select(abs(value0) < threshold, value0, sign(grad)*threshold);
//     // // grad = select(abs(grad)<threshold, grad, sign(grad)*threshold);
//     // value0 = select(value0< threshold, value0, threshold);
//     // grad = select(grad<threshold, grad, threshold);
//     // value0 = select(value0 >-threshold, value0, -threshold);
//     // grad = select(grad>-threshold, grad, -threshold);
//     SpectrumD result = (SpectrumD(value0) * grad) & valid;
//     // SpectrumD result = (SpectrumD(value0) * grad) & valid;
//     // PSDR_ASSERT(all(all(result<threshold)));

//     if constexpr(ad) {
//         return { select(valid, sds.pixel_idx, -1), result - detach(result)};
//     }
//     else{
//         return { select(valid, sds.pixel_idx, -1), SpectrumC(1000.0f) };
//     }
// }


// template <bool ad>
std::tuple<Vector3f<true>,IntC,Vector3f<true>,IntC,
Vector3f<true>,Float<true>,Float<true>,SpectrumD,
Vector3fC,Vector3fC,Vector3fC,Vector3fC> DirectIntegrator::_eval_boundary_edge(const Scene &scene, const Sensor &sensor, const Vector3fC &sample3,
BoundarySegSampleDirect bss, SecondaryEdgeInfo sec_edge_info) const {
    const bool ad = true;
// std::tuple<Vector3fD,IntC,Vector3fD,IntC,Vectof3fD,FloatD,FloatD,FloatD> DirectIntegrator::_eval_boundary_edge(const Scene &scene, const Sensor &sensor, const Vector3fC &sample3) const {
    MaskC valid = bss.is_valid;

    // _p0 on a face edge, _p2 on an emitter
    Vector3fC _p0    = detach(bss.p0);
    Vector3fC       &_p2    = bss.p2,
                    _dir    = normalize(_p2 - _p0);                    
    set_slices(_p2,slices(_p0));

    TriangleInfoD tri_info;
    // IntersectionC _its1,_its2,_its3;
    // Testing silhouette edges 1) Should not have intersection when extended.
    IntersectionC _its1_ext,_its1_light;
    _its1_ext = scene.ray_intersect<false>(RayC(_p0, -_dir), valid);
    auto isShadowEdge = _its1_ext.is_valid() & dot(_dir, _its1_ext.n) > 0.0f;
    
    // 2) Should not be occluded
    // valid &= (~_its1_ext.is_valid());
    valid &= ((~_its1_ext.is_valid()) | (dot(_its1_ext.n,-_dir) < 0.0f));
    // valid &= ((~_its1_ext.is_valid()) | (dot(_its1_ext.n,-_dir) < 0.0f)) | isShadowEdge;
    // valid &= ((~_its1_ext.is_valid()) | (dot(_its1_ext.n,-_dir) < 0.0f)) | isShadowEdge | (~(dot(sec_edge_info.n0,sec_edge_info.n1) > 0.8f));
    // std::cout << "dot(sec_edge_info.n0,sec_edge_info.n1) " << dot(sec_edge_info.n0,sec_edge_info.n1) << std::endl;
    _its1_light = scene.ray_intersect<false>(RayC(_p0, _dir), valid);
    valid &= ~_its1_light.is_valid();
    // _p0 = select(isShadowEdge,_its1_ext.p,_p0);
    // std::cout << "_p0 " << _p0 << std::endl;
    // valid &= ~(dot(sec_edge_info.n0,sec_edge_info.n1) > 0.5f);

    SensorDirectSampleC sds_light = sensor.sample_direct(_p0);
    
    
    IntersectionC _its1= scene.ray_intersect<false>(RayC(_p2,-_dir),valid);
    // FIXME: FIX THIS
    std::cout << "[Before its] count(valid) " << count(valid) << std::endl;
    valid &= _its1.is_valid();
    std::cout << "[AFter its] count(valid) " << count(valid) << std::endl;

    // Connect light ray with camera for setting wi
    BSDFSampleC bs1, bs2;
    BSDFSampleD bs1D;
    BSDFArrayC bsdf_array = _its1.shape->bsdf(valid);
    BSDFArrayD bsdf_arrayD = BSDFArrayD(bsdf_array);
    // auto bs1_samples = scene.m_samplers[2].next_nd<8, false>();
    // This is crucial! Or ya get zero grads
    Sampler sampler;
    sampler.seed(arange<UInt64C>(slices(bss.p0))); 
    Vector8fC bs1_samples = sampler.next_nd<8, false>();
    set_slices(bs1_samples,slices(bss.p0));
    FloatC tmp(1.1f);
    set_slices(tmp,slices(bss.p0));
    bs1_samples[0] = tmp; //FloatC(1.1f);


    // Set incident angle like it's a camera sample
    SensorDirectSampleC sds_camera = sensor.sample_direct(_p0);
    RayC cam_ray = sensor.sample_primary_ray(sds_camera.q);
    _its1.wi = _its1.sh_frame.to_local(-cam_ray.d);

    // Sample camera ray with VAE
    bs1D = bsdf_arrayD->sample(&scene, IntersectionD(_its1), Vector8fD(bs1_samples),  MaskD(valid));
    bs1 = detach(bs1D);
    valid &= bs1D.po.is_valid();
    std::cout << "Number of silhouette samaples: count(valid) " << count(valid) << std::endl;
    
    // xo is a valid VAE output and seen from camera
    SensorDirectSampleC sds = sensor.sample_direct(detach(bs1D.po.p));
    valid &= sds.is_valid;
    std::cout << "Number of silhouette samaples: count(valid) " << count(valid) << std::endl;


    // Connect it to the camera
    Ray<ad> camera_ray;
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
    
    // Set appropriate member variables
    bs1.wo =  bs1.po.sh_frame.to_local(d0);
    bs1.po.wi =  bs1.wo;
    bs2.po = _its1;
    bs2.wo =  bs2.po.sh_frame.to_local(_dir);
    bs2.po.wi =  bs2.wo;
    bs2.rgb_rv = bs1.rgb_rv;
    bs2.po.abs_prob = bs1.po.abs_prob;
    bs2.is_sub = bs1_samples[0] > 1.0f;  //bsdf_array -> hasbssdf();
    // NOTE: don't forget this part...!! Important for eval_sub or ... error bomb will strike
    bs2.is_valid = _its1.is_valid();
    bs2.pdf = bss.pdf;
    bs2.po.J = FloatC(1.0f);
    bs2.maxDist = bs1.maxDist;
    bs2.velocity = bs1.velocity;
    
    // SpectrumC bsdf_val = bsdf_array->eval(bs1.po, bs2, valid);
    FloatC pdfpoint = bsdf_array->pdfpoint(bs1.po, bs2, valid);
    std::cout << "pdfpoint " << pdfpoint << std::endl;
    // SpectrumC value0 = (scene.m_emitters[0]->eval(bs2.po, valid)*(sds.sensor_val/bss.pdf)) & valid;
    // SpectrumC value0 = ( scene.m_emitters[0]->eval(bs2.po, valid)) & valid;

    Vector3fC edge_n0 = normalize(detach(sec_edge_info.n0));
    Vector3fC edge_n1 = normalize(detach(sec_edge_info.n1));
    
    FloatC n0IsLeft = sign(dot(edge_n0,cross(bss.edge,bss.edge2)));
    Vector3fC line_n0 = normalize(cross(edge_n0,bss.edge))*n0IsLeft;
    Vector3fC line_n1 = -normalize(cross(edge_n1,bss.edge))*n0IsLeft;

    // // Compute geometry term for left and right limits
    // Compute offset points
    float eps = 0.05f;
    auto lightpoint = _p2;
    Vector3fC pos_0 = bs2.po.p + eps * line_n0;
    Vector3fC pos_1 = bs2.po.p + eps * line_n1;
    Vector3fC dir_0 = normalize(pos_0 - lightpoint);
    Vector3fC dir_1 = normalize(pos_1 - lightpoint);
    IntersectionC its_0 = scene.ray_intersect<false>(RayC(lightpoint,dir_0),valid);
    IntersectionC its_1 = scene.ray_intersect<false>(RayC(lightpoint,dir_1),valid);
    
    // Chock visibility
    IntersectionC _its_0 = scene.ray_intersect<false>(RayC(pos_0,-dir_0),valid);
    IntersectionC _its_1 = scene.ray_intersect<false>(RayC(pos_1,-dir_1),valid);

    auto bs2_0 = bs2;
    auto bs2_1 = bs2;
    bs2_0.po = its_0;
    bs2_0.wo =  bs2_0.po.sh_frame.to_local(-dir_0);
    bs2_0.po.wi =  bs2_0.wo;
    bs2_0.is_valid = its_0.is_valid();
    bs2_0.po.abs_prob = bs2.po.abs_prob;
    bs2_1.po = its_1;
    bs2_1.wo =  bs2_1.po.sh_frame.to_local(-dir_1);
    bs2_1.po.wi =  bs2_1.wo;
    bs2_1.is_valid = its_1.is_valid();
    bs2_1.po.abs_prob = bs2.po.abs_prob;
    auto valid_0 = valid & ~_its_0.is_valid();
    auto valid_1 = valid & ~_its_1.is_valid();
   std::cout << "count(valid_0) " <<  count(valid_0) << std::endl; 
   std::cout << "count(valid_1) " <<  count(valid_1) << std::endl; 
    // TODO: pddfpoint
    SpectrumC bsdf_0 = bsdf_array->eval(bs1.po, bs2_0, valid);
    // FloatC pdf_0 = bsdf_array->pdfpoint(bs1.po, bs2_0, valid);
    // SpectrumC value_0 = (bsdf_0*scene.m_emitters[0]->eval(bs2_0.po, valid)*(sds.sensor_val/bss.pdf)) & valid_0;
    // SpectrumC value_0 = (bsdf_0*scene.m_emitters[0]->eval(bs2_0.po, valid)) & valid_0; //*(sds.sensor_val/bss.pdf));
    SpectrumC value_0 = bsdf_0 & valid_0; //*(sds.sensor_val/bss.pdf));
    
    SpectrumC bsdf_1 = bsdf_array->eval(bs1.po, bs2_1, valid);
    // FloatC pdf_1 = bsdf_array->pdfpoint(bs1.po, bs2_1, valid);
    // SpectrumC value_1 = (bsdf_1*bsdf_val*scene.m_emitters[0]->eval(bs2_1.po, valid)*(sds.sensor_val/bss.pdf)) & valid_1;
    // SpectrumC value_1 = (bsdf_1*scene.m_emitters[0]->eval(bs2_1.po, valid)) & valid_1; //*(sds.sensor_val/bss.pdf));
    SpectrumC value_1 = bsdf_1 & valid_1; //*(sds.sensor_val/bss.pdf));
    
    // Whether we should flip the pos / neg
    // TODO: check detach
    auto sampleDirection = bs1.po.p - bs2.po.p;
    auto isN0 = dot(line_n1,sampleDirection) < dot(line_n0,sampleDirection); //=bs2.po.p-bs1.po.p)<0.0f;
    auto delta = (value_0 - value_1) * scene.m_emitters[0] -> eval(bs2.po,valid);
    std::cout << "hmean(bss.pdf) " << hmean(bss.pdf) << std::endl;
    std::cout << "hmean(pdfpoint) " << hmean(pdfpoint) << std::endl;
    std::cout << "hmean(sds.sensor_val) " << hmean(sds.sensor_val) << std::endl;

    // delta = delta * (sds.sensor_val/pdfpoint) & valid;
    delta = delta * (sds.sensor_val/bss.pdf/pdfpoint) & valid;
    // delta = delta * (base_v * sds.sensor_val/bss.pdf/pdfpoint) & valid;
    
    delta = select(isN0,delta,-delta);
    Vector3fC n = select(isN0,line_n0,line_n1);

    // vDis ===========================
    // FIXME: Debug viz
    // auto vDis = -bs2.velocity;

    // FloatD kernelEps = bsdf_arrayD -> getKernelEps(IntersectionD(_its1),FloatD(bs1_samples[5]));
    // FloatD maxDist = sqrt(kernelEps);
    // auto xrel = bs2.po.p - bs1.po.p;
    // Vector3fD xi = Vector3fD(xrel) / maxDist;
    // auto vDis = -xi;
    
    auto vDis = -bs1D.po.p;
    // =============================

    auto grad = dot(Vector3fD(n),vDis); //* 100000.0f;
    SpectrumD result = (SpectrumD(delta) * grad) & valid;

    auto pixel_idx = select(valid,sds_light.pixel_idx,-1);
    auto vae_pixel_idx= select(valid,sds.pixel_idx,-1);
    auto deltaMean = hmean(delta);
    auto resultMean = hmean(result);
    // resultMean = resultMean * 0.01f;
    // FIXME
    // result = result * 0.0005f;
    // result = result * 0.000005f;
    SpectrumD value = select(valid,result, 0.0f);
    // FloatD value = select(valid,resultMean, 0.0f);
    auto debug_pos_0 = select(~_its_0.is_valid(),its_0.p,0.0f);
    auto debug_pos_1 = select(~_its_1.is_valid(),its_1.p,0.0f);
    std::cout << "[exp30] Only Boundary... can it go up?" << std::endl;
    
    if constexpr(ad){
        Vector3fD test(0.0f);
        FloatD testF(0.0f);
        Vector3fD _p0D = Vector3fD(_p0) & valid;
        return std::tie(_p0D,pixel_idx,bs1D.po.p,vae_pixel_idx,
                n,deltaMean,grad,value,
                debug_pos_0, debug_pos_1,line_n0,line_n1);
                // its_0.p,its_1.p,line_n0,line_n1);
    }
    else{
        Vector3fC test(0.0f);
        FloatC testF(0.0f);
        return std::tie(_p0,pixel_idx,bs1.po.p,sds.pixel_idx,
                test,testF,testF,value, //detach(value),
                pos_0,pos_1,line_n0,line_n1);
                // pos_0,pos_1,dir_0,dir_1);
                // its_p.p,its_n.p,dir_p,dir_n);
    }
}

template <bool ad>
std::pair<IntC, Spectrum<ad>> DirectIntegrator::eval_boundary_edge(const Scene &scene, const Sensor &sensor, const Vector3fC &sample3) const {
    std::cout << "eval_boundary_edge" << std::endl;
    BoundarySegSampleDirect bss;
    SecondaryEdgeInfo sec_edge_info;
    std::tie(bss,sec_edge_info) = scene.sample_boundary_segment_v2(sample3);
    IntC sampleIdx;
    Spectrum<ad> value;
    if constexpr(ad){
        std::tie(std::ignore,std::ignore,std::ignore,sampleIdx,
        std::ignore,std::ignore,std::ignore,value,
        std::ignore,std::ignore,std::ignore,std::ignore
        ) = _eval_boundary_edge(scene,sensor,sample3,bss,sec_edge_info);
    }

    return {sampleIdx,value};

    // MaskC valid = bss.is_valid;

    // // _p0 on a face edge, _p2 on an emitter
    // const Vector3fC &_p0    = detach(bss.p0);
    // Vector3fC       &_p2    = bss.p2,
    //                 _dir    = normalize(_p2 - _p0);                    
    // set_slices(_p2,slices(_p0));
    
    // IntC sampleIdx;
    // Spectrum<ad> value;
    // if constexpr(ad){
    //     std::tie(std::ignore,std::ignore,std::ignore,sampleIdx,
    //     std::ignore,std::ignore,std::ignore,value) = _eval_boundary_edge(scene,sensor,sample3,bss,sec_edge_info);
    // }

    // return {sampleIdx,value};
    // BoundarySegSampleDirect bss;
    // SecondaryEdgeInfo sec_edge_info;
    // std::tie(bss,sec_edge_info) = scene.sample_boundary_segment_v2(sample3);
    // MaskC valid = bss.is_valid;

    // // _p0 on a face edge, _p2 on an emitter
    // const Vector3fC &_p0    = detach(bss.p0);
    // Vector3fC       &_p2    = bss.p2,
    //                 _dir    = normalize(_p2 - _p0);                    
    // set_slices(_p2,slices(_p0));

    // TriangleInfoD tri_info;
    // IntersectionC _its1,_its2;
    // // Light ray intersection on edge
    // if constexpr ( ad ) {
    //     // Meets at single tangent point?
    //     _its1 = scene.ray_intersect<false>(RayC(_p0, -_dir), MaskC(valid), &tri_info);
    // } else {
    //     _its1 = scene.ray_intersect<false>(RayC(_p0, -_dir), valid);
    // }
    
    // // Testing silhouette edges 1) Should not have intersection when extended.
    // std::cout << "[Var 148] Testing new silhouette detection " << std::endl;
    // valid &= (~_its1.is_valid()) | (dot(_its1.n,-_dir) < 0.0f);
    // // valid &= (~_its1.is_valid()) | (_its1.t < 0.005f);
    // if constexpr ( ad ) {
    //     // Test for occlusion
    //     _its2 = scene.ray_intersect<false>(RayC(_p0, _dir), MaskC(valid), &tri_info);
    // } else {
    //     _its2 = scene.ray_intersect<false>(RayC(_p0, _dir), valid);
    // }
    // valid &= ~_its2.is_valid();
    // // 2) Should not be occluded
    // SensorDirectSampleC sds_silhouette = sensor.sample_direct(_p0);
    


    // valid &=(_its1.is_valid());
    // valid &= norm(_its1.p - _p0) < 0.02f;
    // SensorDirectSampleC sds3= sensor.sample_direct(_p0);
    // // RayC cam_ray = sensor.sample_primary_ray(sds2.q);

    // BSDFSampleC bs1, bs2;
    // BSDFSampleD bs1D;
    // BSDFArrayC bsdf_array = _its1.shape->bsdf(valid);
    // BSDFArrayD bsdf_arrayD = BSDFArrayD(bsdf_array);
    // auto vaesub_bsdf = reinterpret_cast<VaeSub*>(bsdf_arrayD[0]);
    // auto bs1_samples = scene.m_samplers[2].next_nd<8, false>();
    // // TODO: account for bias with weight
    // bs1_samples[0] = 1.1f;
    // // auto weight_sub = (1-F);
    // SensorDirectSampleC sds2 = sensor.sample_direct(_its1.p);
    // RayC cam_ray = sensor.sample_primary_ray(sds2.q);
    // _its1.wi = _its1.sh_frame.to_local(-cam_ray.d);

    // // Sample camera ray
    // bs1D = bsdf_arrayD->sample(&scene, IntersectionD(_its1), Vector8fD(bs1_samples),  MaskD(valid));
    // bs1 = detach(bs1D);
    // // Ray<ad> camera_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);
    // SensorDirectSampleC sds = sensor.sample_direct(bs1.po.p);
    
    // // xo is a valid VAE output and seen from camera
    // valid &= bs1.po.is_valid();
    // // valid &= bs1.is_sub;
    // valid &= sds.is_valid;
    // // std::cout << "[3] valid count" << count(valid) <<std::endl;

    // Ray<ad> camera_ray;
    // // Camera its ray
    // Intersection<ad> its2;
    // if constexpr ( ad ) {
    //     camera_ray = sensor.sample_primary_ray(Vector2fD(sds.q));
    //     its2 = scene.ray_intersect<true, false>(camera_ray, valid);
    //     valid &= its2.is_valid() && norm(detach(its2.p) - bs1.po.p) < ShadowEpsilon;
    // } else {
    //     camera_ray = sensor.sample_primary_ray(sds.q);
    //     its2 = scene.ray_intersect<false>(camera_ray, valid);
    //     valid &= its2.is_valid() && norm(its2.p - bs1.po.p) < ShadowEpsilon;
    // }

    // Vector3fC d0;
    // if constexpr ( ad ) {
    //     d0 = -detach(camera_ray.d);
    // } else {
    //     d0 = -camera_ray.d;
    // }
    // bs1.wo =  bs1.po.sh_frame.to_local(d0);
    // bs1.po.wi =  bs1.wo;
    // bs2.po = _its1;
    // bs2.wo =  bs2.po.sh_frame.to_local(_dir);
    // bs2.po.wi =  bs2.wo;
    // bs2.rgb_rv = bs1.rgb_rv;
    // bs2.po.abs_prob = bs1.po.abs_prob;
    // bs2.is_sub = bsdf_array -> hasbssdf();
    // // NOTE: don't forget this part...!! 
    // // Needed for eval_sub
    // bs2.is_valid = _its1.is_valid();
    // bs2.pdf = bss.pdf;
    // bs2.po.J = FloatC(1.0f);

    // SpectrumC bsdf_val = bsdf_array->eval(bs1.po, bs2, valid);
    // FloatC pdfpoint = bsdf_array->pdfpoint(bs1.po, bs2, valid);
    // SpectrumC value0 = (bsdf_val * scene.m_emitters[0]->eval(bs2.po, valid)*(sds.sensor_val/pdfpoint/bss.pdf)) & valid;
    // // SpectrumC value0 = (scene.m_emitters[0]->eval(bs2.po, valid)*(sds.sensor_val/bss.pdf/pdfpoint)) & valid;
    // // FIXME: tmp disabled 
    // FloatC cos_theta_o = FrameC::cos_theta(bs2.wo);
    // value0 /= cos_theta_o;
    // // FIXME: tmp
    // // SpectrumC value0 = (bsdf_val*scene.m_emitters[0]->eval(bs2.po, valid)*(sds.sensor_val/bss.pdf/pdfpoint)) & valid;

    // Vector3fC edge_n0 = normalize(detach(sec_edge_info.n0));
    // Vector3fC edge_n1 = normalize(detach(sec_edge_info.n1));
    // // // FIXME: should this be dependent on relative position of light ray?
    
    // // Vector3fC line_n0 = normalize(cross(edge_n0,bss.edge));
    // // Vector3fC line_n1 = normalize(cross(edge_n1,bss.edge));
    // // FIXED
    // FloatC n0IsLeft = sign(dot(edge_n0,cross(bss.edge,bss.edge2)));
    // Vector3fC line_n0 = normalize(cross(edge_n0,bss.edge))*n0IsLeft;
    // Vector3fC line_n1 = -normalize(cross(edge_n1,bss.edge))*n0IsLeft;
    
    // // Vector3fC line_n0 = normalize(cross(-bss.edge,edge_n0));
    // // Vector3fC line_n1 = normalize(cross(-edge_n1,bss.edge));

    // // Vector3fC n = line_n0;
    
    // // Compute geometry term for left and right limits
    // float eps = 0.05f;
    // auto lightpoint = _p2;
    // Vector3fC pos_p = bs2.po.p + eps * line_n0;
    // Vector3fC pos_n = bs2.po.p + eps * line_n1;
    // Vector3fC dir_p = normalize(lightpoint - pos_p);
    // Vector3fC dir_n = normalize(lightpoint - pos_n);
    // IntersectionC its_p = scene.ray_intersect<false>(RayC(pos_p,dir_p),valid);
    // IntersectionC its_n = scene.ray_intersect<false>(RayC(pos_n,dir_n),valid);
    // MaskC vis_n = ~its_n.is_valid();
    // MaskC vis_p = ~its_p.is_valid();
    // // std::cout << "cos_theta_o " << cos_theta_o << std::endl;
    
    // // edge normal
    // // deltaG = (G-)-(G+)
    
    // FloatC cos_n = dot(edge_n1,_dir);
    // FloatC cos_p = dot(edge_n0,_dir);
    // FloatC  delta = cos_n & vis_n;
    // delta -= cos_p & vis_p;
    // // valid &= delta > 0.5f;
    
    // std::cout << "_its1.n " << _its1.n << std::endl;
    // std::cout << "edge_n0 " << edge_n0 << std::endl;
    // std::cout << "edge_n1 " << edge_n1 << std::endl;
    
    // // PSDR_ASSERT(all((~valid) | (valid&(_its1.n == edge_n0)) | (valid&(_its1.n == edge_n1))));
    // // _its1.n is closer to edge_n0 than edge_n1
    // // FIXME: checking
    // auto isN0 = (dot(_its1.n,edge_n0) - dot(_its1.n,edge_n1));
    // Vector3fC n = select(isN0 > 0.0f,line_n0,line_n1);
    // // Vector3fC n = normalize(cross(_its1.n,bss.edge));
    // delta *= sign(isN0);
    // // delta *= -sign(dot(_its1.n,edge_n0)-dot(_its1.n,edge_n1));
    // value0 *= delta;
    // // valid &= (vis_n^vis_p); // | (dot(edge_n0,edge_n1) < 0.1f);
    // valid &= (vis_n^vis_p) | (dot(edge_n0,edge_n1) < 0.1f);
    // // valid &= (vis_n^vis_p);
    // // FIXME: add only sharp edges
    // // valid &= (vis_n & vis_p); //
    // // valid &= (vis_n ^ vis_p);
    // // valid &= (~isnan(edge_n0)) && (~isnan(edge_n1));

    // // // Reparameterize xi
    // // // xi = xo + x_VAE / scalefactor
    // FloatD kernelEps = bsdf_arrayD -> getKernelEps(IntersectionD(_its1),FloatD(bs1_samples[5]));
    // FloatD maxDist = sqrt(kernelEps);
    // // Vector3fC xo = detach(bs1D.po.p);
    // // Vector3fC xvae = (bs2.po.p - bs1.po.p) / detach(maxDist);
    // // FIXME: max
    // // xvae = select(xvae > 1.0f, 1.0f, xvae);
    // // FIXME: inverse xvae
    // // xvae = normalize(xvae);
    // // xvae = 1/xvae;

    // //FIXME: derivative of inverse 
    // // Vector3fD xi = Vector3fD(xo) + Vector3fD(xvae) * maxDist;
    // auto xrel = bs2.po.p - bs1.po.p;
    // Vector3fD xi = Vector3fD(xrel) / maxDist;

    // // FIXME: testing
    // // SpectrumD result = dot(Vector3fD(n), xi) & valid;
    // // SpectrumD result = (SpectrumD(value0) * maxDist) & valid;
    // auto grad = dot(Vector3fD(n),xi);
    // // float threshold = 1.0f;
    // // std::cout << "count(all(value0<threshold) " << count(value0.x()<threshold) << std::endl;
    // // std::cout << "count(all(grad<threshold) " << count(grad < threshold) << std::endl;
    // // // value0 = select(abs(value0) < threshold, value0, sign(grad)*threshold);
    // // // grad = select(abs(grad)<threshold, grad, sign(grad)*threshold);
    // // value0 = select(value0< threshold, value0, threshold);
    // // grad = select(grad<threshold, grad, threshold);
    // // value0 = select(value0 >-threshold, value0, -threshold);
    // // grad = select(grad>-threshold, grad, -threshold);
    // SpectrumD result = (SpectrumD(value0) * grad) & valid;
    // // SpectrumD result = (SpectrumD(value0) * grad) & valid;
    // // PSDR_ASSERT(all(all(result<threshold)));

    // if constexpr(ad) {
    //     return { select(valid, sds.pixel_idx, -1), result - detach(result)};
    // }
    // else{
    //     return { select(valid, sds.pixel_idx, -1), SpectrumC(1000.0f) };
    // }
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
