#pragma once

#include <enoki/array.h>
#include <enoki/matrix.h>
#include <enoki/cuda.h>
#include <enoki/autodiff.h>

using namespace enoki;

namespace psdr
{

/********************************************
 * GPU array types
 ********************************************/

template <typename T, bool ad>
using Type      = typename std::conditional<ad,
                                            DiffArray<CUDAArray<T>>,
                                            CUDAArray<T>>::type;

// Scalar arrays (GPU)

template <bool ad>
using Float     = Type<float, ad>;

template <bool ad>
using Int       = Type<int32_t, ad>;

using FloatC    = Float<false>;
using FloatD    = Float<true>;

using IntC      = Int<false>;
using IntD      = Int<true>;

// Vector arrays (GPU)

template <int n, bool ad>
using Vectorf   = Array<Float<ad>, n>;

template <int n, bool ad>
using Vectori   = Array<Int<ad>, n>;

template <int n, bool ad>
using Matrixf   = Matrix<Float<ad>, n>;

template <bool ad>
using Vector2f  = Vectorf<2, ad>;

template <bool ad>
using Vector2i  = Vectori<2, ad>;

template <bool ad>
using Vector3f  = Vectorf<3, ad>;

template <bool ad>
using Vector20f  = Vectorf<20, ad>;

template <bool ad>
using ShapeVector20f  = Array<Vectorf<20, ad>,3>;

template <bool ad>
using Vector3i  = Vectori<3, ad>;

template <bool ad>
using Vector4f  = Vectorf<4, ad>;

template <bool ad>
using Vector4i  = Vectori<4, ad>;

template <bool ad>
using Vector5f  = Vectorf<5, ad>;

template <bool ad>
using Vector5i  = Vectori<5, ad>;

template <bool ad>
using Vector7i  = Vectori<7, ad>;

template <bool ad>
using Vector7f  = Vectorf<7, ad>;

template <bool ad>
using Vector8i  = Vectori<8, ad>;

template <bool ad>
using Vector8f  = Vectorf<8, ad>;
// // NOTE: added by Joon
// template <bool ad>
// using Vector12f  = Vectorf<12, ad>;

using Vector2fC = Vector2f<false>;
using Vector2fD = Vector2f<true>;

using Vector2iC = Vector2i<false>;
using Vector2iD = Vector2i<true>;

using Vector3fC = Vector3f<false>;
using Vector3fD = Vector3f<true>;

using Vector20fC = Vector20f<false>;
using Vector20fD = Vector20f<true>;

using ShapeVector20fC = ShapeVector20f<false>;
using ShapeVector20fD = ShapeVector20f<true>;

using Vector3iC = Vector3i<false>;
using Vector3iD = Vector3i<true>;

using Vector4fC = Vector4f<false>;
using Vector4fD = Vector4f<true>;

using Vector4iC = Vector4i<false>;
using Vector4iD = Vector4i<true>;

using Vector5iC = Vector5i<false>;
using Vector5iD = Vector5i<true>;

using Vector5fC = Vector5f<false>;
using Vector5fD = Vector5f<true>;

using Vector7fC = Vector7f<false>;
using Vector7fD = Vector7f<true>;

using Vector8fC = Vector8f<false>;
using Vector8fD = Vector8f<true>;

// using Vector12fC = Vector12f<false>;
// using Vector12fD = Vector12f<true>;
// Matrix arrays (GPU)

template <bool ad>
using Matrix3f  = Matrixf<3, ad>;

template <bool ad>
using Matrix4f  = Matrixf<4, ad>;

using Matrix3fC = Matrix3f<false>;
using Matrix3fD = Matrix3f<true>;

using Matrix4fC = Matrix4f<false>;
using Matrix4fD = Matrix4f<true>;

// Mask arrays (GPU)

template <bool ad>
using Mask      = mask_t<Float<ad>>;

using MaskC     = Mask<false>;
using MaskD     = Mask<true>;

// Spectrum types (GPU)

template <bool ad>
using Spectrum  = Vectorf<PSDR_NUM_CHANNELS, ad>;

using SpectrumC = Spectrum<false>;
using SpectrumD = Spectrum<true>;

/********************************************
 * CPU types
 ********************************************/

// Scalar types (CPU)

using ScalarVector2f = Array<float, 2>;
using ScalarVector3f = Array<float, 3>;
using ScalarVector4f = Array<float, 4>;

using ScalarVector2i = Array<int, 2>;
using ScalarVector3i = Array<int, 3>;
using ScalarVector4i = Array<int, 4>;

using ScalarMatrix2f = Matrix<float, 2>;
using ScalarMatrix3f = Matrix<float, 3>;
using ScalarMatrix4f = Matrix<float, 4>;

/********************************************
 * Triangle info types
 ********************************************/

template <typename Float_>
struct TriangleInfo_ {
    static constexpr bool ad = std::is_same_v<Float_, FloatD>;

    Vector3f<ad> p0, e1, e2, n0, n1, n2, face_normal;
    Float<ad>    face_area;
    Array<Vectorf<20,ad>,3> poly0, poly1, poly2;

    ENOKI_STRUCT(TriangleInfo_, p0, e1, e2,
                                n0, n1, n2,
                                poly0,poly1,poly2,
                                face_normal,
                                face_area)
};

template <bool ad>
using TriangleInfo  = TriangleInfo_<Float<ad>>;

using TriangleInfoC = TriangleInfo<false>;
using TriangleInfoD = TriangleInfo<true>;

template <bool ad>
using TriangleUV = Array<Vector2f<ad>, 3>;

using TriangleUVC   = TriangleUV<false>;
using TriangleUVD   = TriangleUV<true>;

/********************************************
 * Others
 ********************************************/

// For samplers

using UIntC     = Type<uint32_t, false>;
using UInt64C   = Type<uint64_t, false>;

// Render options

struct RenderOption {
    RenderOption() : width(128), height(128), spp(1), sppe(1), log_level(1) { cropwidth = width; cropheight = height; cropoffset_x = 0, cropoffset_y = 0,rgb=0;}
    RenderOption(int w, int h, int s) : width(w), height(h), spp(s), sppe(s), sppse(s), log_level(1) {cropwidth = width; cropheight = height; cropoffset_x = 0, cropoffset_y = 0,rgb=0;}
    RenderOption(int w, int h, int s1, int s2) : width(w), height(h), spp(s1), sppe(s2), sppse(s2), log_level(1) {cropwidth = width; cropheight = height; cropoffset_x = 0, cropoffset_y = 0,rgb=0;}
    RenderOption(int w, int h, int s1, int s2, int s3) : width(w), height(h), spp(s1), sppe(s2), sppse(s3), log_level(1) {cropwidth = width; cropheight = height; cropoffset_x = 0, cropoffset_y = 0,rgb=0;}
    RenderOption(int w, int h, int s1, int s2, int s3, int s4) : width(w), height(h), spp(s1), sppe(s2), sppse(s3), sppsce(s4), log_level(1) {cropwidth = width; cropheight = height; cropoffset_x = 0, cropoffset_y = 0,rgb=0;}

    int width, height;  // Image resolution
    int spp;            // Spp for the main image/interior integral
    int sppe;           // Spp for primary edge integral
    int sppse;          // Spp for secondary edge integral
    int sppsce;          // Spp for secondary edge integral
    int log_level;
    int cropwidth, cropheight;
    int cropoffset_x, cropoffset_y;
    int rgb = 0; // [1,2,3] for [R,G,B], 0 for random
    int debug = 0;
    int isFD = 0;
    float epsM = 2.0f;
    float maxDistScale = 2.0f;
    // Options for vaesub
    // 0: shape-adaptive, light-space, 1: shape-adaptive: tangent-space
    // 2: planar, light-space, 3: planar, tangent-space
    int vaeMode; 
    int isFull;
    int mode; // 0 for SSS+opaque, 1 for opaque, 2 for SSS
};

} // namespace psdr

ENOKI_STRUCT_SUPPORT(psdr::TriangleInfo_, p0, e1, e2, n0, n1, n2,poly0,poly1,poly2,
                                          face_normal, face_area)
