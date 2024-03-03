#if !defined(__SCATTER_EIGEN_H__)
#define __SCATTER_EIGEN_H__

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


// #include <mitsuba/render/sss_particle_tracer.h>

#include <Eigen/Core>
#include <Eigen/Dense>
// #include <enoki/matrix.h>
#include <enoki/array.h>
#include <enoki/dynamic.h>

#include <psdr/psdr.h>
#include <psdr/core/bitmap.h>
#include <psdr/types.h>

// using Vector4f = Array<float, 4>;
// using Vector3f = Array<float, 3>;

// MTS_NAMESPACE_BEGIN

namespace psdr {

    class Volpath3D {
        using Spectrum = Vector3f<true>;
        public: 
            static float effectiveAlbedo(const float &albedo) {
                return -std::log(1.0f - albedo * (1.0f - std::exp(-8.0f))) / 8.0f;
            }

            static Spectrum getSigmaTp(const Spectrum &albedo, float g, const Spectrum &sigmaT) {
                Spectrum sigmaS = albedo * sigmaT;
                Spectrum sigmaA = sigmaT - sigmaS;
                return (1 - g) * sigmaS + sigmaA;
            }

            static Spectrum effectiveAlbedo(const Spectrum &albedo) {
                auto res = -log(1.0f - albedo * (1.0f - exp(-8.0f))) / 8.0f;
                return res; 
                
            }
        
    };

class NetworkHelpers {
public:
    // template <bool ad>
    // static void onb(Spectrum<ad> &n, Spectrum<ad> &b1, Spectrum<ad> &b2) {
    //     auto sign = select(n[2] > 0, full<Float<ad>>(1.0f), full<Float<ad>>(-1.0f));
    //     auto a = -1.0f / (sign + n[2]);
    //     auto b = n[0] * n[1] * a;

    //     b1 = Spectrum<ad>(1.0f + sign * n[0] * n[0] * a, sign * b, -sign * n[0]);
    //     b2 = Spectrum<ad>(b, sign + n[1]*n[1]*a, -n[1]);
    // }

    static inline constexpr int nChooseK(int n, int k) {
        return (k == 0 || n == k) ? 1 : nChooseK(n - 1, k - 1) + nChooseK(n - 1, k);
    }

    static inline constexpr int nPolyCoeffs(int polyOrder) { return nChooseK(3 + polyOrder, polyOrder); }

    static inline constexpr int nInFeatures(int polyOrder) { return nPolyCoeffs(polyOrder) + 3; }

    template<bool ad>
    static inline Float<ad> sigmoid(Float<ad> x) { return 1.0f / (1.0f + exp(-x)); }

    static void loadVector(const std::string &filename,float* array) {
        std::ifstream f(filename, std::ios::binary);
        if (!f.is_open()) {
            std::cout << "FILE NOT FOUND " << filename << std::endl;
            return;
        }
        std::cout << "Loading " << filename << std::endl;

        int32_t nDims;
        f.read(reinterpret_cast<char *>(&nDims), sizeof(nDims));
        int32_t size;
        f.read(reinterpret_cast<char *>(&size), sizeof(size));

        // Enoki dynamic array initialization
        for (int i = 0; i < size; ++i) {
            float value;
            f.read(reinterpret_cast<char *>(&value), sizeof(value));
            array[i] = value;
        }
    }
    // // template <size_t rows, size_t cols>
    static void loadMat(const std::string &filename,float* array,int size=64)
    {
        std::ifstream f(filename, std::ios::binary);
        if (!f.is_open()) {
            std::cout << "FILE NOT FOUND " << filename << std::endl;
            return;
        }
        std::cout << "Loading " << filename << std::endl;

        int32_t nDims;
        f.read(reinterpret_cast<char *>(&nDims), sizeof(nDims));

        int32_t rows, cols;
        f.read(reinterpret_cast<char *>(&rows), sizeof(rows));
        f.read(reinterpret_cast<char *>(&cols), sizeof(cols));

        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j) {
                f.read(reinterpret_cast<char *>(array+j*size + i), sizeof(float));
            }
        }
    }
    







    };
}
#endif /* __SCATTER_EIGEN_H__ */