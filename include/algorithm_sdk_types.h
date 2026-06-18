#ifndef ALGORITHM_SDK_TYPES_H
#define ALGORITHM_SDK_TYPES_H
#include <vector>
namespace Algorithm_Sdk
{
    enum ErrCode
    {
        SUCCESS = 0,
        INIT_ERR = 1,
        INIT_MODEL_ERR = 2,
        MODEL_PATH_ERR = 3,
        MODEL_ERR = 4,
        POSTPROCESS_ERR = 5,
        PARAM_ERR = 6,
        CUDA_RUNTIME_ERR = 7,
        CUDA_MEMORY_ERR = 8,
        MODEL_INFERENCE_ERR = 9,
        BATCHSIZE_ERR = 10,
        GPU_DEVICE_ID_ERR = 11,
        INPUT_FORMAT_ERR = 12,
        IMAGE_PREPROCESS_ERR = 13
    };
    struct BoundingBox
    {
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;
    };
    struct DetectionResult
    {
        BoundingBox box;
        float score = 0.0f;
        int class_id = 0;
    };
    struct ImageBuffer;
    using ImageHandle = ImageBuffer *;
}
#endif