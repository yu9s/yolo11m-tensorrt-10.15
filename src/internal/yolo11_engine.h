#ifndef YOLO11_ENGINE_H
#define YOLO11_ENGINE_H
#include "algorithm_sdk_types.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <mutex>
namespace Algorithm_Sdk
{
    class Engine
    {
    public:
        Engine() = default;
        ~Engine();
        ErrCode init(const std::string &engine_path, int device_id = 0);
        ErrCode infer(std::vector<void *> &buffers, int batch, cudaStream_t stream = nullptr);
        int input_count() const;
        int output_count() const;
        std::vector<int64_t> input_shape(int index) const;
        std::vector<int64_t> output_shape(int index) const;
        nvinfer1::DataType output_dtype(int index) const;
        int output_dtype_size(int index) const;
        int max_batch_size() const;
    private:
        nvinfer1::IRuntime *runtime_ = nullptr;
        nvinfer1::ICudaEngine *engine_ = nullptr;
        nvinfer1::IExecutionContext *context_ = nullptr;
        bool dynamic_batch_ = false;
        int input_num_ = 0;
        int output_num_ = 0;
        std::mutex infer_mutex_;
    };
    int dtype_size(nvinfer1::DataType t);
}
#endif
