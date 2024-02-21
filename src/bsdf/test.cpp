
// #include <enoki/cuda.h>
// #include <enoki/autodiff.h>
// #include <enoki/special.h>
// #include <psdr/core/bitmap.h>
#include <enoki/matrix.h>
#include <enoki/array.h>
#include <fstream>

#include <enoki/random.h>
#include <enoki/special.h>

// #include <enoki/dynamic.h>

// using Matrix = enoki::Matrix;
// using Array = enoki::Array;
void loadVector(const std::string &filename,float* array) {
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

void loadMat(const std::string &filename,float* array)
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
    // f.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    // f.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    f.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    f.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    std::cout << rows << " " << cols << std::endl;

    // for (size_t i = 0; i < cols; ++i)
    // {
    //     for (size_t j = 0; j < rows; ++j) {
    //         float value;
    //         f.read(reinterpret_cast<char *>(array+64*j+i), sizeof(float));
    //     }
    // }
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j) {
            float value;
            f.read(reinterpret_cast<char *>(array+j*64+i), sizeof(float));
        }
    }
}

// template <typename Array>
// void printArray(const Array& arr) {
//     for (size_t i = 0; i < arr.size(); ++i) {
//         if constexpr (enoki::is_array_v<typename Array::Value>) {
//             // If the element is also an array, recurse
//             std::cout << "[" << i <<"]" << " ";
//             printArray(arr[i]);
//         } else {
//             // Base case: print the element
//             std::cout << arr[i] << " ";
//         }
//     }
//     std::cout << std::endl; // Newline for readability
// }

using namespace enoki;
int main() {

    /////////////////////////////////////////////////////////////////
    std::cout << "[Test 1] Matrix vector multiplication" << std::endl;

    Matrix<float, 3> mat(1,2,3,4,5,6,7,8,9);
    Array<float, 3> vec(1,2,3);
    auto result = mat*vec;
    std::cout << result << std::endl;
    
    /////////////////////////////////////////////////////////////////
    const std::string variablePath = "./variables";
    std::cout << "[Test 2] 64 dim matrix" << std::endl;
    const int DIM = 64;
    const int ROWS = 23;
    const int COLS = 64;
    auto mat2 = arange<Matrix<float,DIM>>(); //(0);
    auto vec2 = arange<Array<float, DIM>>();
    auto dummy = zero<Array<float,DIM>>();
    for(int i=0;i<64;i++){
        for(int j=0;j<23;j++){
            mat2[j][i] = 1.0;
        }
    }
    for(int i=0;i<23;i++){
        dummy[i] = 1.0;
    }

    // loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_biases.bin",(float*) &vec2);
    // loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_weights.bin",(float*) &mat2);
    
    // std::cout << dummy << mat2[0] <<std::endl;
    // std::cout<< "hsum " <<hsum(mat2[0]) <<std::endl;
    std::cout << mat2 << vec2 << std::endl; //result2 <<std::endl; //[0] <<" " <<result2[1]<<std::endl;
    auto result2 = mat2*dummy; // + vec2;
    std::cout << result2 <<std::endl; //[0] <<" " <<result2[1]<<std::endl;
    // for (int i=0;i<ROWS;i++){
    //     for(int j=0;j<COLS;j++){
    //         mat2[i][j] = 1.0;
    //     }
    // }
    // for (int i=0;i<ROWS;i++){
    //     vec2[i] = 1.0;
    // }


    auto mat3 = zero<Matrix<float,DIM>>(); //(0);
    auto vec3 = zero<Array<float, DIM>>();
    auto mat31 = zero<Matrix<float,DIM>>(); //(0);
    auto vec31 = zero<Array<float, DIM>>();
    auto mat32 = zero<Matrix<float,DIM>>(); //(0);
    auto vec32 = zero<Array<float, DIM>>();
    // auto dummy = zero<Array<float,DIM>>();
    loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_biases.bin",(float*) &vec3);
    loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_weights.bin",(float*) &mat3);
    loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_biases.bin",(float*) &vec31);
    loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_weights.bin",(float*) &mat31);
    loadVector(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_biases.bin",(float*) &vec32);
    loadMat(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_weights.bin",(float*) &mat32);

    using Matrix64f = Matrix<float,64>;
    using Vector64f = Array<float,64>;
    using Vector32f = Array<float,32>;
    using Vector1f = Array<float,1>;
    
    Matrix64f absorption_mlp_fcn_0_weights = zero<Matrix64f>(); // Eigen::Matrix<float,32,64>
    Vector64f absorption_mlp_fcn_0_biases = zero<Vector64f>();
    Matrix64f absorption_dense_kernel = zero<Matrix64f>();
    Vector1f absorption_dense_bias;
    loadMat(variablePath + "/absorption_mlp_fcn_0_weights.bin",(float*) &absorption_mlp_fcn_0_weights);
    loadMat(variablePath + "/absorption_dense_kernel.bin",(float*) &absorption_dense_kernel);
    loadVector(variablePath + "/absorption_dense_bias.bin", (float*) &absorption_dense_bias);
    loadVector(variablePath + "/absorption_mlp_fcn_0_biases.bin", (float*) &absorption_mlp_fcn_0_biases);
    
    auto result3 = max(mat3 * dummy + vec3,0.0f);
    result3 = max(mat31 * result3 + vec31,0.0f);
    result3 = max(mat32 * result3 + vec32,0.0f);
    std::cout << result3 << std::endl;
    
    auto feature = result3;
    // auto abs = max(absorption_mlp_fcn_0_weights * feature + absorption_mlp_fcn_0_biases,0.0f);
    auto abs = max(absorption_mlp_fcn_0_weights * feature + absorption_mlp_fcn_0_biases,0.0f);
    abs = absorption_dense_kernel * abs + absorption_dense_bias[0];
    // abs = absorption_dense_kernel * abs + absorption_dense_bias;
    std::cout << "Absorption is " << abs << std::endl;
    
    // Array<float,64+4> inputFeature;
    // Array<float,4> latent;
    // input << latent,feature;
    // Sample random vector
    using UInt32      = Packet<uint32_t>;
    using RNG         = PCG32<UInt32>;
    using UInt64      = RNG::UInt64;
    RNG rng(PCG32_DEFAULT_STATE, arange<UInt64>() + (0 * UInt32::Size));
    auto x = rng.next_float32();
    Array<float,4> latent(rng.next_float32()[0],rng.next_float32()[0],rng.next_float32()[0],rng.next_float32()[0]);
    
    // Importance sample from normal distribution using output of PCG32
    auto y = float(M_SQRT2) * erfinv(2.f*latent - 1.f);
    std::cout << y << std::endl;
    
    return 0;
    // enoki::Array<float, 4> v(1,1,1,1); // = {2.3, 1.0, 2.2, 3.0};


    // mat = identity<Matrix<float,4>>(1.0);
    // printArray(result);

    // std::cout<< mat<< std::endl
    // auto result = mat * v;

    // printArray(result);

    // std::cout<< mat<< std::endl;


}