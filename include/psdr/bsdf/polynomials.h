#pragma once
#if !defined(__POLYNOMIALS_H)
#define __POLYNOMIALS_H_

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// #include <enoki/matrix.h>
#include <enoki/array.h>
#include <enoki/dynamic.h>

#include <psdr/psdr.h>
#include <psdr/core/bitmap.h>
#include <psdr/types.h>


const static int PERMX2[10] = {1,  4,  5,  6, -1, -1, -1, -1, -1, -1};
const static int PERMY2[10] = {2,  5,  7,  8, -1, -1, -1, -1, -1, -1};
const static int PERMZ2[10] =  {3,  6,  8,  9, -1, -1, -1, -1, -1, -1};
const static int PERMX3[20] = {1,  4,  5,  6, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1,-1, -1};
const static int PERMY3[20] = {2,  5,  7,  8, 11, 13, 14, 16, 17, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
const static int PERMZ3[20] = {3,  6,  8,  9, 12, 14, 15, 17, 18, 19, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

namespace psdr{ 
    
    template <bool ad>
    using Point = Vector3f<ad>;


    template <bool ad>
    using Vector = Point<ad>;
    
    template <bool ad>
    using VectorShape = Array<Float<ad>,20>;
    
    const int *derivPermutation(size_t degree, size_t axis) {
        switch (axis) {
            case 0:
                return degree == 2 ? PERMX2 : PERMX3;
            case 1:
                return degree == 2 ? PERMY2 : PERMY3;
            case 2:
                return degree == 2 ? PERMZ2 : PERMZ3;
            default:
                return nullptr;
        }
    }


// template<size_t degree>
// float evalPolyImpl(const Point &pos,
//  const Point &evalP, Float scaleFactor, bool useLocalDir, const Vector &refDir,
//     const std::vector<float> &coeffs) {
//     assert(degree <= 3);

//     Vector relPos;
//     if (useLocalDir) {
//         Vector s, t;
//         Volpath3D::onbDuff(refDir, s, t);
//         Frame local(s, t, refDir);
//         relPos = local.toLocal(evalP - pos) * scaleFactor;
//     } else {
//         relPos = (evalP - pos) * scaleFactor;
//     }

//     size_t termIdx = 4;
//     float value = coeffs[0] + coeffs[1] * relPos.x + coeffs[2] * relPos.y + coeffs[3] * relPos.z;
//     float xPowers[4] = {1.0, relPos.x, relPos.x * relPos.x, relPos.x * relPos.x * relPos.x};
//     float yPowers[4] = {1.0, relPos.y, relPos.y * relPos.y, relPos.y * relPos.y * relPos.y};
//     float zPowers[4] = {1.0, relPos.z, relPos.z * relPos.z, relPos.z * relPos.z * relPos.z};

//     for (size_t d = 2; d <= degree; ++d) {
//         for (size_t i = 0; i <= d; ++i) {
//             size_t dx = d - i;
//             for (size_t j = 0; j <= i; ++j) {
//                 size_t dy = d - dx - j;
//                 size_t dz = d - dx - dy;
//                 value += coeffs[termIdx] * xPowers[dx] * yPowers[dy] * zPowers[dz];
//                 ++termIdx;
//             }
//         }
//     }
//     return value;
// }



// float evalPoly(const Point &pos,
//  const Point &evalP, size_t degree,
//     Float scaleFactor, bool useLocalDir, const Vector &refDir,
//     const std::vector<float> &coeffs) {
//     assert(degree <= 3 && degree >= 2);
//     if (degree == 3)
//         return evalPolyImpl<3>(pos, evalP, scaleFactor, useLocalDir, refDir, coeffs);
//     else
//         return evalPolyImpl<2>(pos, evalP, scaleFactor, useLocalDir, refDir, coeffs);

// }

template <bool ad>
void onb(const Spectrum<ad> &n, Spectrum<ad> &b1, Spectrum<ad> &b2) {
    auto sign = enoki::sign(n.z()); //select(n[2] > 0, full<Float<ad>>(1.0f), full<Float<ad>>(-1.0f));
    auto a = -1.0f / (sign + n[2]);
    auto b = n[0] * n[1] * a;

    b1 = Spectrum<ad>(1.0f + sign * n[0] * n[0] * a, sign * b, -sign * n[0]);
    b2 = Spectrum<ad>(b, sign + n[1]*n[1]*a, -n[1]);
}

template<bool ad>
std::pair<Float<ad>, Vector<ad>> evalPolyGrad(const Point<ad> &pos,
 const Point<ad> &evalP, size_t degree,
    const int *permX, const int *permY, const int *permZ,
    Float<ad> scaleFactor, bool useLocalDir, const Vector<ad> &refDir,
    const VectorShape<ad> &coeffs) {
    
    Vector<ad> relPos;
    if (useLocalDir) {
        Frame<ad> local(refDir); //s, t, refDir);
        relPos = local.to_local(evalP - pos) * scaleFactor;
    } else {
        relPos = (evalP - pos) * scaleFactor;
    }

    size_t termIdx = 0;
    Vector<ad> deriv(0.0f, 0.0f, 0.0f);
    Float<ad> value = 0.0f;

    // auto xPowers = concat(1.0f,relPos.x()); 
    Array<Float<ad>,4> xPowers(1.0f, relPos.x(), relPos.x() * relPos.x(), relPos.x() * relPos.x() * relPos.x());
    Array<Float<ad>,4> yPowers(1.0f, relPos.y(), relPos.y() * relPos.y(), relPos.y() * relPos.y() * relPos.y());
    Array<Float<ad>,4> zPowers(1.0f, relPos.z(), relPos.z() * relPos.z(), relPos.z() * relPos.z() * relPos.z());
    for (size_t d = 0; d <= degree; ++d) {
        for (size_t i = 0; i <= d; ++i) {
            size_t dx = d - i;
            for (size_t j = 0; j <= i; ++j) {
                size_t dy = d - dx - j;
                size_t dz = d - dx - dy;
                Float<ad> t = xPowers[dx] * yPowers[dy] * zPowers[dz];
                value += coeffs[termIdx] * t;

                int pX = permX[termIdx];
                int pY = permY[termIdx];
                int pZ = permZ[termIdx];
                if (pX > 0)
                    deriv[0] += (dx + 1) * t * coeffs[pX];
                if (pY > 0)
                    deriv[1] += (dy + 1) * t * coeffs[pY];
                if (pZ > 0)
                    deriv[2] += (dz + 1) * t * coeffs[pZ];
                ++termIdx;
            }
        }
    }
    return std::make_pair(value, deriv);
}

template<bool ad> 
std::pair<Float<ad>,Vector<ad>> evalGradient(const Point<ad> &pos,
 const VectorShape<ad> &coeffs, const Point<ad> &p, size_t degree, Float<ad> scaleFactor, 
 bool useLocalDir, const Vector<ad> &refDir) {

    const int *permX = derivPermutation(degree, 0);
    const int *permY = derivPermutation(degree, 1);
    const int *permZ = derivPermutation(degree, 2);
    Float<ad> polyValue;
    Vector<ad> gradient;
    std::tie(polyValue, gradient) = evalPolyGrad<ad>(pos, p, degree, permX, permY, permZ, scaleFactor, useLocalDir, refDir, coeffs);
    // std::cout << "gradient " << gradient<< std::endl;

    if (useLocalDir) {
        // Vector<ad> s, t;
        // onb<ad>(refDir, s, t);
        // Frame<ad> local(s, t, refDir);
        Frame<ad> local(refDir); //s, t, refDir);
        gradient = local.to_world(gradient);
        // std::cout << "gradient after " << gradient<< std::endl;
    }

    return std::make_pair(polyValue,gradient);
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

}


// class PolyUtils {
//     public:
//         template <bool ad>
//         struct Polynomial { 
//             Array<Float<ad>,20> coeffs;
//             Vector3f<ad> refPos;
//             Vector3f<ad> refDir;
            
//         };

//         template<int order,bool ad>
//     static void rotatePolynomial(Array<Float<ad>,20> &c,
//                                     std::vector<float> &c2,
//                                     const Vector &s,
//                                     const Vector &t,
//                                     const Vector &n) {

//         c2[0] = c[0];
//         c2[1] = c[1]*s[0] + c[2]*s[1] + c[3]*s[2];
//         c2[2] = c[1]*t[0] + c[2]*t[1] + c[3]*t[2];
//         c2[3] = c[1]*n[0] + c[2]*n[1] + c[3]*n[2];
//         c2[4] = c[4]*powi(s[0], 2) + c[5]*s[0]*s[1] + c[6]*s[0]*s[2] + c[7]*powi(s[1], 2) + c[8]*s[1]*s[2] + c[9]*powi(s[2], 2);
//         c2[5] = 2*c[4]*s[0]*t[0] + c[5]*(s[0]*t[1] + s[1]*t[0]) + c[6]*(s[0]*t[2] + s[2]*t[0]) + 2*c[7]*s[1]*t[1] + c[8]*(s[1]*t[2] + s[2]*t[1]) + 2*c[9]*s[2]*t[2];
//         c2[6] = 2*c[4]*n[0]*s[0] + c[5]*(n[0]*s[1] + n[1]*s[0]) + c[6]*(n[0]*s[2] + n[2]*s[0]) + 2*c[7]*n[1]*s[1] + c[8]*(n[1]*s[2] + n[2]*s[1]) + 2*c[9]*n[2]*s[2];
//         c2[7] = c[4]*powi(t[0], 2) + c[5]*t[0]*t[1] + c[6]*t[0]*t[2] + c[7]*powi(t[1], 2) + c[8]*t[1]*t[2] + c[9]*powi(t[2], 2);
//         c2[8] = 2*c[4]*n[0]*t[0] + c[5]*(n[0]*t[1] + n[1]*t[0]) + c[6]*(n[0]*t[2] + n[2]*t[0]) + 2*c[7]*n[1]*t[1] + c[8]*(n[1]*t[2] + n[2]*t[1]) + 2*c[9]*n[2]*t[2];
//         c2[9] = c[4]*powi(n[0], 2) + c[5]*n[0]*n[1] + c[6]*n[0]*n[2] + c[7]*powi(n[1], 2) + c[8]*n[1]*n[2] + c[9]*powi(n[2], 2);
//         if (order > 2) {
//             c2[10] = c[10]*powi(s[0], 3) + c[11]*powi(s[0], 2)*s[1] + c[12]*powi(s[0], 2)*s[2] + c[13]*s[0]*powi(s[1], 2) + c[14]*s[0]*s[1]*s[2] + c[15]*s[0]*powi(s[2], 2) + c[16]*powi(s[1], 3) + c[17]*powi(s[1], 2)*s[2] + c[18]*s[1]*powi(s[2], 2) + c[19]*powi(s[2], 3);
//             c2[11] = 3*c[10]*powi(s[0], 2)*t[0] + c[11]*(powi(s[0], 2)*t[1] + 2*s[0]*s[1]*t[0]) + c[12]*(powi(s[0], 2)*t[2] + 2*s[0]*s[2]*t[0]) + c[13]*(2*s[0]*s[1]*t[1] + powi(s[1], 2)*t[0]) + c[14]*(s[0]*s[1]*t[2] + s[0]*s[2]*t[1] + s[1]*s[2]*t[0]) + c[15]*(2*s[0]*s[2]*t[2] + powi(s[2], 2)*t[0]) + 3*c[16]*powi(s[1], 2)*t[1] + c[17]*(powi(s[1], 2)*t[2] + 2*s[1]*s[2]*t[1]) + c[18]*(2*s[1]*s[2]*t[2] + powi(s[2], 2)*t[1]) + 3*c[19]*powi(s[2], 2)*t[2];
//             c2[12] = 3*c[10]*n[0]*powi(s[0], 2) + c[11]*(2*n[0]*s[0]*s[1] + n[1]*powi(s[0], 2)) + c[12]*(2*n[0]*s[0]*s[2] + n[2]*powi(s[0], 2)) + c[13]*(n[0]*powi(s[1], 2) + 2*n[1]*s[0]*s[1]) + c[14]*(n[0]*s[1]*s[2] + n[1]*s[0]*s[2] + n[2]*s[0]*s[1]) + c[15]*(n[0]*powi(s[2], 2) + 2*n[2]*s[0]*s[2]) + 3*c[16]*n[1]*powi(s[1], 2) + c[17]*(2*n[1]*s[1]*s[2] + n[2]*powi(s[1], 2)) + c[18]*(n[1]*powi(s[2], 2) + 2*n[2]*s[1]*s[2]) + 3*c[19]*n[2]*powi(s[2], 2);
//             c2[13] = 3*c[10]*s[0]*powi(t[0], 2) + c[11]*(2*s[0]*t[0]*t[1] + s[1]*powi(t[0], 2)) + c[12]*(2*s[0]*t[0]*t[2] + s[2]*powi(t[0], 2)) + c[13]*(s[0]*powi(t[1], 2) + 2*s[1]*t[0]*t[1]) + c[14]*(s[0]*t[1]*t[2] + s[1]*t[0]*t[2] + s[2]*t[0]*t[1]) + c[15]*(s[0]*powi(t[2], 2) + 2*s[2]*t[0]*t[2]) + 3*c[16]*s[1]*powi(t[1], 2) + c[17]*(2*s[1]*t[1]*t[2] + s[2]*powi(t[1], 2)) + c[18]*(s[1]*powi(t[2], 2) + 2*s[2]*t[1]*t[2]) + 3*c[19]*s[2]*powi(t[2], 2);
//             c2[14] = 6*c[10]*n[0]*s[0]*t[0] + c[11]*(2*n[0]*s[0]*t[1] + 2*n[0]*s[1]*t[0] + 2*n[1]*s[0]*t[0]) + c[12]*(2*n[0]*s[0]*t[2] + 2*n[0]*s[2]*t[0] + 2*n[2]*s[0]*t[0]) + c[13]*(2*n[0]*s[1]*t[1] + 2*n[1]*s[0]*t[1] + 2*n[1]*s[1]*t[0]) + c[14]*(n[0]*s[1]*t[2] + n[0]*s[2]*t[1] + n[1]*s[0]*t[2] + n[1]*s[2]*t[0] + n[2]*s[0]*t[1] + n[2]*s[1]*t[0]) + c[15]*(2*n[0]*s[2]*t[2] + 2*n[2]*s[0]*t[2] + 2*n[2]*s[2]*t[0]) + 6*c[16]*n[1]*s[1]*t[1] + c[17]*(2*n[1]*s[1]*t[2] + 2*n[1]*s[2]*t[1] + 2*n[2]*s[1]*t[1]) + c[18]*(2*n[1]*s[2]*t[2] + 2*n[2]*s[1]*t[2] + 2*n[2]*s[2]*t[1]) + 6*c[19]*n[2]*s[2]*t[2];
//             c2[15] = 3*c[10]*powi(n[0], 2)*s[0] + c[11]*(powi(n[0], 2)*s[1] + 2*n[0]*n[1]*s[0]) + c[12]*(powi(n[0], 2)*s[2] + 2*n[0]*n[2]*s[0]) + c[13]*(2*n[0]*n[1]*s[1] + powi(n[1], 2)*s[0]) + c[14]*(n[0]*n[1]*s[2] + n[0]*n[2]*s[1] + n[1]*n[2]*s[0]) + c[15]*(2*n[0]*n[2]*s[2] + powi(n[2], 2)*s[0]) + 3*c[16]*powi(n[1], 2)*s[1] + c[17]*(powi(n[1], 2)*s[2] + 2*n[1]*n[2]*s[1]) + c[18]*(2*n[1]*n[2]*s[2] + powi(n[2], 2)*s[1]) + 3*c[19]*powi(n[2], 2)*s[2];
//             c2[16] = c[10]*powi(t[0], 3) + c[11]*powi(t[0], 2)*t[1] + c[12]*powi(t[0], 2)*t[2] + c[13]*t[0]*powi(t[1], 2) + c[14]*t[0]*t[1]*t[2] + c[15]*t[0]*powi(t[2], 2) + c[16]*powi(t[1], 3) + c[17]*powi(t[1], 2)*t[2] + c[18]*t[1]*powi(t[2], 2) + c[19]*powi(t[2], 3);
//             c2[17] = 3*c[10]*n[0]*powi(t[0], 2) + c[11]*(2*n[0]*t[0]*t[1] + n[1]*powi(t[0], 2)) + c[12]*(2*n[0]*t[0]*t[2] + n[2]*powi(t[0], 2)) + c[13]*(n[0]*powi(t[1], 2) + 2*n[1]*t[0]*t[1]) + c[14]*(n[0]*t[1]*t[2] + n[1]*t[0]*t[2] + n[2]*t[0]*t[1]) + c[15]*(n[0]*powi(t[2], 2) + 2*n[2]*t[0]*t[2]) + 3*c[16]*n[1]*powi(t[1], 2) + c[17]*(2*n[1]*t[1]*t[2] + n[2]*powi(t[1], 2)) + c[18]*(n[1]*powi(t[2], 2) + 2*n[2]*t[1]*t[2]) + 3*c[19]*n[2]*powi(t[2], 2);
//             c2[18] = 3*c[10]*powi(n[0], 2)*t[0] + c[11]*(powi(n[0], 2)*t[1] + 2*n[0]*n[1]*t[0]) + c[12]*(powi(n[0], 2)*t[2] + 2*n[0]*n[2]*t[0]) + c[13]*(2*n[0]*n[1]*t[1] + powi(n[1], 2)*t[0]) + c[14]*(n[0]*n[1]*t[2] + n[0]*n[2]*t[1] + n[1]*n[2]*t[0]) + c[15]*(2*n[0]*n[2]*t[2] + powi(n[2], 2)*t[0]) + 3*c[16]*powi(n[1], 2)*t[1] + c[17]*(powi(n[1], 2)*t[2] + 2*n[1]*n[2]*t[1]) + c[18]*(2*n[1]*n[2]*t[2] + powi(n[2], 2)*t[1]) + 3*c[19]*powi(n[2], 2)*t[2];
//             c2[19] = c[10]*powi(n[0], 3) + c[11]*powi(n[0], 2)*n[1] + c[12]*powi(n[0], 2)*n[2] + c[13]*n[0]*powi(n[1], 2) + c[14]*n[0]*n[1]*n[2] + c[15]*n[0]*powi(n[2], 2) + c[16]*powi(n[1], 3) + c[17]*powi(n[1], 2)*n[2] + c[18]*n[1]*powi(n[2], 2) + c[19]*powi(n[2], 3);
//         }
//         for (size_t i = 0; i < c2.size(); ++i) {
//             c[i] = c2[i];
//         }
//     }
// }


#endif