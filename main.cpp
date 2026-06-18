#include "test/algorithm_sdk_test.h"
#include <string>

int main(void)
{
    int test_total_loop = 1;
    std::string Yolo11_model = "../data/models/weed_detection.trt";
    std::string Yolo11_image_path = "../data/images";
    std::string Yolo_save_path = "./weed_detection/";
    float Yolo11_confidence = 0.50f;
    int Yolo11_batch = 1;
    test_yolo11_sdk(Yolo11_model,
                    Yolo11_image_path,
                    Yolo11_batch,
                    Yolo11_confidence,
                    Yolo_save_path,
                    test_total_loop);
    return 0;
}

// #include <opencv2/opencv.hpp>
// #include <iostream>

// int main()
// {
//     std::cout << cv::getBuildInformation() << std::endl;
//     return 0;
// }