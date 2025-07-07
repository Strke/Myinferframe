#ifndef INFER_FRAME
#define INFER_FRAME
#include <glog/logging.h>
#include <armadillo>
#include <memory>
#include <numeric>
#include <vector>

namespace kuiper_infer {
template <typename T>
class Tensor {
public:
    uint32_t rows() const;
    uint32_t cols() const;
    uint32_t channels() const;
    uint32_t size() const;
    void set_data(const arma:fcube& data);

    //创建三维张量
    explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

    //创建二维张量
    explicit Tensor(uint32_t rows, uint32_t cols);

    //创建一维张量
    explicit Tensor(uint32_t size);

    void Fill(const std::vector<T>& values, bool row_major = true);

private:
    std::vector<uint32_t> raw_shapes_;
    arma::fcube data_;
};
}

#endif