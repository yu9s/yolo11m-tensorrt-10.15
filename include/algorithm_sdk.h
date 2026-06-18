#ifndef ALGORITHM_SDK_H
#define ALGORITHM_SDK_H
#include <algorithm_sdk_types.h>
#include <memory>
#include <opencv2/core/core.hpp>
#include <string>
#include <vector>
namespace Algorithm_Sdk
{
    class DetectionSDK
    {
    public:
        DetectionSDK();
        ~DetectionSDK();
        DetectionSDK(const DetectionSDK &) = delete;
        ErrCode init(const std::string &model_path, int max_batch_size = 1, int device_id = 0);
        ErrCode image_handle_create(ImageHandle *handle);
        ErrCode image_handle_free(ImageHandle *handle);
        ErrCode image_uniformization(const cv::Mat &inout, ImageHandle handle);
        ErrCode image_uniformization(const std::vector<cv::Mat> &inputs, ImageHandle handle);
        ErrCode forward(ImageHandle handle, std::vector<std::vector<DetectionResult>> *results, float confidence_thresh, float nms_thresh, bool verbose = false);
        bool is_initialized() const;
    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}
#endif