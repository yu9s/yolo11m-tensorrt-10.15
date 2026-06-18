#ifndef TEST_YOLO11_SDK_H
#define TEST_YOLO11_SDK_H
#include "algorithm_sdk.h"
#include <string>
Algorithm_Sdk::ErrCode
test_yolo11_sdk(const std::string &model_path,
                const std::string &test_image_dir,
                int max_batch,
                float confidence_thresh,
                const std::string &save_dir,
                int loop_count);
#endif
