#include <vector>
#include <armadillo>
#include "../data/tensor.hpp"

namespace kuiper_infer {
//创建一维张量
Tensor<float>::Tensor(uint32_t size){
    data_ = arma::fcube(1, size, 1);
    this->raw_shapes_ = std::vector<uint32_t>{size};
}

//创建二维张量
Tensor<float>::Tensor(uint32_t rows, uint32_t cols){
    data_ = arma::fcube(rows, cols, 1);
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
}

//创建三维张量
Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols){
    data_ = arma::fcube(rows, cols, channels);
    if (channels == 1 && rows == 1){
        this->raw_shapes_ = std::vector<uint32_t>{cols};
    }else if(channels == 1){
        this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
    }else{
        this->raw_shapes_ = std::vector<uint32_t>{rows, cols, channels};
    }
}
template<typename T>
void Tensor<T>::Fill(const std::vector<T>& values, bool row_major = true){

    if(row_major){
        const uint32_t rows = this->rows();
        const uint32_t cols = this->cols();
        const uint32_t channels = this->channels();
        const uint32_t planes = rows * cols;

        for(uint32_t i = 0; i < channels; i ++){
            auto& channel_data = this->data_.slice(i);
            const arma::fmat& channel_data_t = arma::fmat(values.data() + i * planes, this->cols(), this->rows());
            channels_data = channel_data_t.t();
        }
    }
}
}
