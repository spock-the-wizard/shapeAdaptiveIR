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
template <bool ad>
void onb(const Spectrum<ad> &n, Spectrum<ad> &b1, Spectrum<ad> &b2) {
    auto sign = select(n[2] > 0, full<Float<ad>>(1.0f), full<Float<ad>>(-1.0f));
    auto a = -1.0f / (sign + n[2]);
    auto b = n[0] * n[1] * a;

    b1 = Spectrum<ad>(1.0f + sign * n[0] * n[0] * a, sign * b, -sign * n[0]);
    b2 = Spectrum<ad>(b, sign + n[1]*n[1]*a, -n[1]);
}
    template<int order,bool ad>
    static Array<Float<ad>,20> rotatePolynomial(Array<Float<ad>,20> &c,
                                    const Array<Float<ad>,3> &s,
                                    const Array<Float<ad>,3> &t,
                                    const Array<Float<ad>,3> &n) {

        auto c2 = zero<Array<Float<ad>,20>>();

        c2[0] = c[0];
        c2[1] = c[1]*s[0] + c[2]*s[1] + c[3]*s[2];
        c2[2] = c[1]*t[0] + c[2]*t[1] + c[3]*t[2];
        c2[3] = c[1]*n[0] + c[2]*n[1] + c[3]*n[2];
        
        c2[4] = c[4]*pow(s[0], 2) + c[5]*s[0]*s[1] + c[6]*s[0]*s[2] + c[7]*pow(s[1], 2) + c[8]*s[1]*s[2] + c[9]*pow(s[2], 2);
        c2[5] = 2*c[4]*s[0]*t[0] + c[5]*(s[0]*t[1] + s[1]*t[0]) + c[6]*(s[0]*t[2] + s[2]*t[0]) + 2*c[7]*s[1]*t[1] + c[8]*(s[1]*t[2] + s[2]*t[1]) + 2*c[9]*s[2]*t[2];
        c2[6] = 2*c[4]*n[0]*s[0] + c[5]*(n[0]*s[1] + n[1]*s[0]) + c[6]*(n[0]*s[2] + n[2]*s[0]) + 2*c[7]*n[1]*s[1] + c[8]*(n[1]*s[2] + n[2]*s[1]) + 2*c[9]*n[2]*s[2];
        c2[7] = c[4]*pow(t[0], 2) + c[5]*t[0]*t[1] + c[6]*t[0]*t[2] + c[7]*pow(t[1], 2) + c[8]*t[1]*t[2] + c[9]*pow(t[2], 2);
        c2[8] = 2*c[4]*n[0]*t[0] + c[5]*(n[0]*t[1] + n[1]*t[0]) + c[6]*(n[0]*t[2] + n[2]*t[0]) + 2*c[7]*n[1]*t[1] + c[8]*(n[1]*t[2] + n[2]*t[1]) + 2*c[9]*n[2]*t[2];
        c2[9] = c[4]*pow(n[0], 2) + c[5]*n[0]*n[1] + c[6]*n[0]*n[2] + c[7]*pow(n[1], 2) + c[8]*n[1]*n[2] + c[9]*pow(n[2], 2);
        if (order > 2) {
            c2[10] = c[10]*pow(s[0], 3) + c[11]*pow(s[0], 2)*s[1] + c[12]*pow(s[0], 2)*s[2] + c[13]*s[0]*pow(s[1], 2) + c[14]*s[0]*s[1]*s[2] + c[15]*s[0]*pow(s[2], 2) + c[16]*pow(s[1], 3) + c[17]*pow(s[1], 2)*s[2] + c[18]*s[1]*pow(s[2], 2) + c[19]*pow(s[2], 3);
            c2[11] = 3*c[10]*pow(s[0], 2)*t[0] + c[11]*(pow(s[0], 2)*t[1] + 2*s[0]*s[1]*t[0]) + c[12]*(pow(s[0], 2)*t[2] + 2*s[0]*s[2]*t[0]) + c[13]*(2*s[0]*s[1]*t[1] + pow(s[1], 2)*t[0]) + c[14]*(s[0]*s[1]*t[2] + s[0]*s[2]*t[1] + s[1]*s[2]*t[0]) + c[15]*(2*s[0]*s[2]*t[2] + pow(s[2], 2)*t[0]) + 3*c[16]*pow(s[1], 2)*t[1] + c[17]*(pow(s[1], 2)*t[2] + 2*s[1]*s[2]*t[1]) + c[18]*(2*s[1]*s[2]*t[2] + pow(s[2], 2)*t[1]) + 3*c[19]*pow(s[2], 2)*t[2];
            c2[12] = 3*c[10]*n[0]*pow(s[0], 2) + c[11]*(2*n[0]*s[0]*s[1] + n[1]*pow(s[0], 2)) + c[12]*(2*n[0]*s[0]*s[2] + n[2]*pow(s[0], 2)) + c[13]*(n[0]*pow(s[1], 2) + 2*n[1]*s[0]*s[1]) + c[14]*(n[0]*s[1]*s[2] + n[1]*s[0]*s[2] + n[2]*s[0]*s[1]) + c[15]*(n[0]*pow(s[2], 2) + 2*n[2]*s[0]*s[2]) + 3*c[16]*n[1]*pow(s[1], 2) + c[17]*(2*n[1]*s[1]*s[2] + n[2]*pow(s[1], 2)) + c[18]*(n[1]*pow(s[2], 2) + 2*n[2]*s[1]*s[2]) + 3*c[19]*n[2]*pow(s[2], 2);
            c2[13] = 3*c[10]*s[0]*pow(t[0], 2) + c[11]*(2*s[0]*t[0]*t[1] + s[1]*pow(t[0], 2)) + c[12]*(2*s[0]*t[0]*t[2] + s[2]*pow(t[0], 2)) + c[13]*(s[0]*pow(t[1], 2) + 2*s[1]*t[0]*t[1]) + c[14]*(s[0]*t[1]*t[2] + s[1]*t[0]*t[2] + s[2]*t[0]*t[1]) + c[15]*(s[0]*pow(t[2], 2) + 2*s[2]*t[0]*t[2]) + 3*c[16]*s[1]*pow(t[1], 2) + c[17]*(2*s[1]*t[1]*t[2] + s[2]*pow(t[1], 2)) + c[18]*(s[1]*pow(t[2], 2) + 2*s[2]*t[1]*t[2]) + 3*c[19]*s[2]*pow(t[2], 2);
            c2[14] = 6*c[10]*n[0]*s[0]*t[0] + c[11]*(2*n[0]*s[0]*t[1] + 2*n[0]*s[1]*t[0] + 2*n[1]*s[0]*t[0]) + c[12]*(2*n[0]*s[0]*t[2] + 2*n[0]*s[2]*t[0] + 2*n[2]*s[0]*t[0]) + c[13]*(2*n[0]*s[1]*t[1] + 2*n[1]*s[0]*t[1] + 2*n[1]*s[1]*t[0]) + c[14]*(n[0]*s[1]*t[2] + n[0]*s[2]*t[1] + n[1]*s[0]*t[2] + n[1]*s[2]*t[0] + n[2]*s[0]*t[1] + n[2]*s[1]*t[0]) + c[15]*(2*n[0]*s[2]*t[2] + 2*n[2]*s[0]*t[2] + 2*n[2]*s[2]*t[0]) + 6*c[16]*n[1]*s[1]*t[1] + c[17]*(2*n[1]*s[1]*t[2] + 2*n[1]*s[2]*t[1] + 2*n[2]*s[1]*t[1]) + c[18]*(2*n[1]*s[2]*t[2] + 2*n[2]*s[1]*t[2] + 2*n[2]*s[2]*t[1]) + 6*c[19]*n[2]*s[2]*t[2];
            c2[15] = 3*c[10]*pow(n[0], 2)*s[0] + c[11]*(pow(n[0], 2)*s[1] + 2*n[0]*n[1]*s[0]) + c[12]*(pow(n[0], 2)*s[2] + 2*n[0]*n[2]*s[0]) + c[13]*(2*n[0]*n[1]*s[1] + pow(n[1], 2)*s[0]) + c[14]*(n[0]*n[1]*s[2] + n[0]*n[2]*s[1] + n[1]*n[2]*s[0]) + c[15]*(2*n[0]*n[2]*s[2] + pow(n[2], 2)*s[0]) + 3*c[16]*pow(n[1], 2)*s[1] + c[17]*(pow(n[1], 2)*s[2] + 2*n[1]*n[2]*s[1]) + c[18]*(2*n[1]*n[2]*s[2] + pow(n[2], 2)*s[1]) + 3*c[19]*pow(n[2], 2)*s[2];
            c2[16] = c[10]*pow(t[0], 3) + c[11]*pow(t[0], 2)*t[1] + c[12]*pow(t[0], 2)*t[2] + c[13]*t[0]*pow(t[1], 2) + c[14]*t[0]*t[1]*t[2] + c[15]*t[0]*pow(t[2], 2) + c[16]*pow(t[1], 3) + c[17]*pow(t[1], 2)*t[2] + c[18]*t[1]*pow(t[2], 2) + c[19]*pow(t[2], 3);
            c2[17] = 3*c[10]*n[0]*pow(t[0], 2) + c[11]*(2*n[0]*t[0]*t[1] + n[1]*pow(t[0], 2)) + c[12]*(2*n[0]*t[0]*t[2] + n[2]*pow(t[0], 2)) + c[13]*(n[0]*pow(t[1], 2) + 2*n[1]*t[0]*t[1]) + c[14]*(n[0]*t[1]*t[2] + n[1]*t[0]*t[2] + n[2]*t[0]*t[1]) + c[15]*(n[0]*pow(t[2], 2) + 2*n[2]*t[0]*t[2]) + 3*c[16]*n[1]*pow(t[1], 2) + c[17]*(2*n[1]*t[1]*t[2] + n[2]*pow(t[1], 2)) + c[18]*(n[1]*pow(t[2], 2) + 2*n[2]*t[1]*t[2]) + 3*c[19]*n[2]*pow(t[2], 2);
            c2[18] = 3*c[10]*pow(n[0], 2)*t[0] + c[11]*(pow(n[0], 2)*t[1] + 2*n[0]*n[1]*t[0]) + c[12]*(pow(n[0], 2)*t[2] + 2*n[0]*n[2]*t[0]) + c[13]*(2*n[0]*n[1]*t[1] + pow(n[1], 2)*t[0]) + c[14]*(n[0]*n[1]*t[2] + n[0]*n[2]*t[1] + n[1]*n[2]*t[0]) + c[15]*(2*n[0]*n[2]*t[2] + pow(n[2], 2)*t[0]) + 3*c[16]*pow(n[1], 2)*t[1] + c[17]*(pow(n[1], 2)*t[2] + 2*n[1]*n[2]*t[1]) + c[18]*(2*n[1]*n[2]*t[2] + pow(n[2], 2)*t[1]) + 3*c[19]*pow(n[2], 2)*t[2];
            c2[19] = c[10]*pow(n[0], 3) + c[11]*pow(n[0], 2)*n[1] + c[12]*pow(n[0], 2)*n[2] + c[13]*n[0]*pow(n[1], 2) + c[14]*n[0]*n[1]*n[2] + c[15]*n[0]*pow(n[2], 2) + c[16]*pow(n[1], 3) + c[17]*pow(n[1], 2)*n[2] + c[18]*n[1]*pow(n[2], 2) + c[19]*pow(n[2], 3);
        }
        for (size_t i = 0; i < c2.size(); ++i) {
            c[i] = c2[i];
        }
        return c2;
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
Vector3f<ad> L2I(const Vector3f<ad> vLearned){
    // Perform coordinate transform between learnedsss -> invt
    return Vector3f<ad>(vLearned.x(),vLearned.z(),-vLearned.y());
    // return Vector3f<ad>(vLearned.x(),vLearned.z(),vLearned.y());
}
std::tuple<Vector3fC, Vector3fC, Vector3fC, Vector20fC, FloatC> Integrator::sample_sub(const Scene &scene, Vector3f<false> pts, Vector3f<false> dir) {
// std::tuple<Vector3fC, Vector3fC, Vector3fC, Array<Array<Float<false>,20>,3>> Integrator::sample_sub(const Scene &scene, Vector3f<false> pts, Vector3f<false> dir) {
    
    // std::cout << "testing" << std::endl;
    // std::cout << "pts[0]" << pts[0] << std::endl;
    // std::cout << dir << std::endl;

// std::cout << "pts.slices " << shape(pts)[0]<< shape(pts)[1] << std::endl;
    constexpr bool ad = false;
    auto active = full<Mask<ad>>(true); //~isnan(pts[0]);

    // TODO: do not set tmax... fu... use RayC instead (tmax set to inf)
    // auto tmax = full<Float<ad>>(3000.0f);
    // Ray<ad> ray(pts,dir,tmax);
    Intersection<ad> its = scene.ray_intersect<ad>(RayC(pts,dir), true);
    // its.poly_coeff = zero<decltype(its.poly_coeff)>();
    for (int i=0;i<3;i++){
    //     // Case 1. Plane f = z
        // its.poly_coeff = zero<decltype(its.poly_coeff)>();
        // its.poly_coeff[i][3] = 1.0;

        // Case 2. f = -z + x^2 + y^2
        // its.poly_coeff[i][3] = 1.0f;
        // its.poly_coeff[i][4] = -1.0f;
        // its.poly_coeff[i][7] = -1.0f;
    }
    // std::cout << "its.poly_coeff " << its.poly_coeff << std::endl;
    // its.poly_coeff[3] = 1.0;
    
    // std::cout << "Surface its " << its.p << std::endl;
    // std::cout << its.p << std::endl;
    // std::cout << its.is_valid() << std::endl;
    // auto active = its.is_valid();

    
    Float<ad> pdf;
    Sampler sampler;
    sampler.seed(arange<UInt64C>(slices(pts)));

    // Type<HeterSub*,ad> vaesub_array = its.shape->bsdf(true);
    Type<VaeSub*,ad> vaesub_array = its.shape->bsdf(true);
    // if(true) {
    //     vaesub_array = static_cast<Type<HeterSub*, ad>>(vaesub_array);
    // }
    // BSDFArray<ad> bsdf_array = its.shape->bsdf(true);
    // std::cout << "bsdf_array " << bsdf_array << std::endl;
    // std::cout << "vaesub_array " << vaesub_array << std::endl;
    
    // Mask<ad> active = full<Mask<ad>>(true);
    // auto sampler_ = scene.m_samplers[0]
    // Vector3fC res1, res2;
    // Intersection<ad> first;
    // Float<ad> second;
    // std::tie(first,second) = vaesub_array[0] -> __sample_sp<ad>(&scene, its, (scene.m_samplers[0].next_nd<8, false>()),pdf, active);
    // std::cout << "active" << active << std::endl;
    // std::cout << "its.p" << its.p << std::endl;
    
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
    
    // Rotate polynomials in TS -> WS
    // poly_coeffs = rotatePolynomial<3,ad>(poly_coeffs,forward,up,right); //right,up);
    // WS polynomials
    // poly_coeffs = rotatePolynomial<3,ad>(poly_coeffs,-forward,-right,up); //right,up);
    // poly_coeffs = rotatePolynomial<3,ad>(poly_coeffs,right,-forward,up); //right,up);
    // poly_coeffs = rotatePolynomial<3,ad>(poly_coeffs,forward,-right,up); //right,up);
    // std::cout << "poly_coeffs" << poly_coeffs << std::endl;
    auto active_sub = its_sub.is_valid();
    // std::cout << "active_sub " << active_sub << std::endl;
    // std::cout << "hsum(active_sub) " << hsum(active_sub) << std::endl;
    // std::cout << "second" << second << std::endl;
    // std::cout << "outPos" << outPos << std::endl;
    // std::cout << "projDir" << projDir << std::endl;

    Float<ad> color = weight; 

    return std::make_tuple(its_sub.p,outPos,projDir,poly_coeffs,color);
}

SpectrumD Integrator::renderD(const Scene &scene, int sensor_id) const {
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

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

    auto end_time = high_resolution_clock::now();
    if ( scene.m_opts.log_level ) {
        std::stringstream oss;
        oss << "Rendered in " << duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";
        LOG(oss.str().c_str());
    }
    // std::cout<<"renderD: test3"<<std::endl;
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

        Vector2f<ad> samples = (samples_base + scene.m_samplers[0].next_2d<ad>())
                                /ScalarVector2f(opts.cropwidth, opts.cropheight);
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
