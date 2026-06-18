#include "yolo11_engine.h"
#include <NvInferRuntime.h>
#include <fstream>
#include <iostream>
#include <vector>
namespace Algorithm_Sdk
{
    class TrtLogger : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char *msg) noexcept override
        {
            if (severity <= Severity::kWARNING)
            {
                std::cout << "[TRT] " << msg << std::endl;
            }
        }
    };
    static TrtLogger g_logger;
    int dtype_size(nvinfer1::DataType t)
    {
        switch (t)
        {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT8:
            return 1;
        default:
            return 4;
        }
    }
    Engine::~Engine()
    {
        if (context_)
        {
            delete context_;
            context_ = nullptr;
        }
        if (engine_)
        {
            delete engine_;
            engine_ = nullptr;
        }
        if (runtime_)
        {
            delete runtime_;
            runtime_ = nullptr;
        }
    }
    ErrCode Engine::init(const std::string &engine_path, int device_id)
    {
        int gpu_cnt = 0;
        cudaGetDeviceCount(&gpu_cnt);
        if (device_id < 0 || device_id >= gpu_cnt)
        {
            return GPU_DEVICE_ID_ERR;
        }
        cudaSetDevice(device_id);
        std::ifstream file(engine_path, std::ios::in | std::ios::binary);
        if (!file)
        {
            return MODEL_PATH_ERR;
        }
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);
        if (!file)
        {
            return MODEL_ERR;
        }
        file.close();
        runtime_ = nvinfer1::createInferRuntime(g_logger);
        if (runtime_ == nullptr)
        {
            return CUDA_RUNTIME_ERR;
        }
        engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
        if (engine_ == nullptr)
        {
            return MODEL_ERR;
        }
        int total = engine_->getNbIOTensors();
        input_num_ = 0;
        output_num_ = 0;
        for (int i = 0; i < total; ++i)
        {
            const char *name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            {
                input_num_++;
            }
            else
            {
                output_num_++;
            }
        }
        if (input_num_ > 0)
        {
            auto dims0 = engine_->getTensorShape(engine_->getIOTensorName(0));
            dynamic_batch_ = (dims0.d[0] == -1);
        }
        context_ = engine_->createExecutionContext();
        if (context_ == nullptr)
        {
            return CUDA_RUNTIME_ERR;
        }
        return SUCCESS;
    }
    ErrCode Engine::infer(std::vector<void *> &buffers, int batch, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(infer_mutex_);
        if (engine_ == nullptr || context_ == nullptr)
        {
            return MODEL_INFERENCE_ERR;
        }
        if (dynamic_batch_ && input_num_ > 0)
        {
            const char *in_name = engine_->getIOTensorName(0);
            auto in_dims = context_->getTensorShape(in_name);
            if (in_dims.d[0] != batch)
            {
                in_dims.d[0] = batch;
                if (!context_->setInputShape(in_name, in_dims))
                {
                    return MODEL_INFERENCE_ERR;
                }
            }
        }
        int total = engine_->getNbIOTensors();
        if (static_cast<int>(buffers.size()) != total)
        {
            return PARAM_ERR;
        }
        for (int i = 0; i < total; ++i)
        {
            context_->setTensorAddress(engine_->getIOTensorName(i), buffers[i]);
        }
        bool ok = context_->enqueueV3(stream);
        if (!ok)
        {
            return MODEL_INFERENCE_ERR;
        }
        return SUCCESS;
    }
    int Engine::input_count() const { return input_num_; }
    int Engine::output_count() const { return output_num_; }
    std::vector<int64_t> Engine::input_shape(int index) const
    {
        std::vector<int64_t> shape;
        int input_seen = 0;
        int total = engine_->getNbIOTensors();
        for (int i = 0; i < total; ++i)
        {
            const char *name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            {
                if (input_seen == index)
                {
                    auto d = engine_->getTensorShape(name);
                    for (int j = 0; j < d.nbDims; ++j)
                    {
                        shape.push_back(d.d[j]);
                    }
                    return shape;
                }
                input_seen++;
            }
        }
        return shape;
    }
    std::vector<int64_t> Engine::output_shape(int index) const
    {
        std::vector<int64_t> shape;
        int output_seen = 0;
        int total = engine_->getNbIOTensors();
        for (int i = 0; i < total; ++i)
        {
            const char *name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT)
            {
                if (output_seen == index)
                {
                    auto d = engine_->getTensorShape(name);
                    for (int j = 0; j < d.nbDims; ++j)
                    {
                        shape.push_back(d.d[j]);
                    }
                    return shape;
                }
                output_seen++;
            }
        }
        return shape;
    }
    nvinfer1::DataType Engine::output_dtype(int index) const
    {
        int output_seen = 0;
        int total = engine_->getNbIOTensors();
        for (int i = 0; i < total; ++i)
        {
            const char *name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT)
            {
                if (output_seen == index)
                {
                    return engine_->getTensorDataType(name);
                }
                output_seen++;
            }
        }
        return nvinfer1::DataType::kFLOAT;
    }
    int Engine::output_dtype_size(int index) const
    {
        return dtype_size(output_dtype(index));
    }
    int Engine::max_batch_size() const
    {
        if (!dynamic_batch_)
        {
            auto s = input_shape(0);
            return s.empty() ? 1 : static_cast<int>(s[0]);
        }
        auto max_dim = engine_->getProfileShape(engine_->getIOTensorName(0), 0,
                                                nvinfer1::OptProfileSelector::kMAX);
        return max_dim.d[0];
    }
}
