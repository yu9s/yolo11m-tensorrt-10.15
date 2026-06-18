#include "algorithm_sdk_test.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <dirent.h>
#include <cstdio>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <sys/types.h>
#include <string>
#include <vector>
namespace
{
    bool ends_with_ignore_case(const std::string &text, const std::string &suffix)
    {
        if (text.size() < suffix.size())
            return false;
        return std::equal(suffix.rbegin(), suffix.rend(), text.rbegin(),
                          [](char a, char b)
                          {
                              return std::tolower(static_cast<unsigned char>(a)) ==
                                     std::tolower(static_cast<unsigned char>(b));
                          });
    }
    int list_images(const std::string &dir, std::vector<std::string> &files)
    {
        DIR *dp = opendir(dir.c_str());
        if (!dp)
        {
            std::cerr << "open dir failed: " << dir << std::endl;
            return -1;
        }
        while (dirent *ent = readdir(dp))
        {
            std::string name = ent->d_name;
            if (name == "." || name == "..")
                continue;
            if (ends_with_ignore_case(name, ".jpg") ||
                ends_with_ignore_case(name, ".jpeg") ||
                ends_with_ignore_case(name, ".png") ||
                ends_with_ignore_case(name, ".bmp"))
            {
                files.push_back(dir + "/" + name);
            }
        }
        closedir(dp);
        std::sort(files.begin(), files.end());
        return 0;
    }
    void sdk_clear_txt_files(const std::string &dir)
    {
        DIR *dp = opendir(dir.c_str());
        if (!dp)
            return;
        while (dirent *ent = readdir(dp))
        {
            std::string name = ent->d_name;
            if (name == "." || name == "..")
                continue;
            if (ends_with_ignore_case(name, ".txt"))
            {
                std::string file_path = dir + "/" + name;
                std::remove(file_path.c_str());
            }
        }
        closedir(dp);
    }
    std::string join_path(const std::string &dir, const std::string &name)
    {
        if (dir.empty())
            return name;
        if (dir.back() == '/')
            return dir + name;
        return dir + "/" + name;
    }
    std::string base_name(const std::string &path)
    {
        size_t pos = path.find_last_of("/\\");
        if (pos == std::string::npos)
            return path;
        return path.substr(pos + 1);
    }
    std::string stem_name(const std::string &path)
    {
        std::string name = base_name(path);
        size_t pos = name.find_last_of('.');
        if (pos == std::string::npos)
            return name;
        return name.substr(0, pos);
    }
    int save_detection_result(const cv::Mat &image,
                              const std::string &image_path,
                              const std::vector<Algorithm_Sdk::DetectionResult> &results,
                              const std::string &image_save_dir,
                              const std::string &label_save_dir)
    {
        cv::Mat vis = image.clone();
        const std::string image_save_path = join_path(image_save_dir, base_name(image_path));
        const std::string label_save_path = join_path(label_save_dir, stem_name(image_path) + ".txt");
        std::ofstream label_file(label_save_path);
        if (!label_file.is_open())
        {
            std::cerr << "open label file failed: " << label_save_path << std::endl;
            return -1;
        }
        label_file << std::fixed << std::setprecision(6);
        const float image_w = static_cast<float>(image.cols);
        const float image_h = static_cast<float>(image.rows);
        for (const auto &obj : results)
        {
            int x = std::max(0, obj.box.x);
            int y = std::max(0, obj.box.y);
            int w = std::min(obj.box.width, image.cols - x);
            int h = std::min(obj.box.height, image.rows - y);
            if (w <= 0 || h <= 0)
                continue;
            cv::rectangle(vis, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
            std::string text = std::to_string(obj.class_id) + " " + cv::format("%.2f", obj.score);
            cv::putText(vis, text, cv::Point(x, std::max(0, y - 5)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            const float cx = (static_cast<float>(x) + static_cast<float>(w) * 0.5f) / image_w;
            const float cy = (static_cast<float>(y) + static_cast<float>(h) * 0.5f) / image_h;
            const float nw = static_cast<float>(w) / image_w;
            const float nh = static_cast<float>(h) / image_h;
            label_file << obj.class_id << " " << cx << " " << cy << " " << nw << " " << nh << "\n";
        }
        label_file.close();
        if (!cv::imwrite(image_save_path, vis))
        {
            std::cerr << "save image failed: " << image_save_path << std::endl;
            return -1;
        }
        return 0;
    }
    void free_handle_if_needed(Algorithm_Sdk::DetectionSDK &model, Algorithm_Sdk::ImageHandle *handle)
    {
        if (handle && *handle)
            model.image_handle_free(handle);
    }
}
Algorithm_Sdk::ErrCode
test_yolo11_sdk(const std::string &model_path,
                const std::string &test_image_dir,
                int max_batch,
                float confidence_thresh,
                const std::string &save_dir,
                int loop_count)
{
    if (max_batch <= 0 || loop_count <= 0)
        return Algorithm_Sdk::PARAM_ERR;
    if (save_dir.empty())
        return Algorithm_Sdk::PARAM_ERR;
    std::string save_root = save_dir;
    while (!save_root.empty() && save_root.back() == '/')
        save_root.pop_back();
    if (save_root.empty())
        return Algorithm_Sdk::PARAM_ERR;
    const mode_t mode = S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO;
    int is_create = mkdir(save_root.c_str(), mode);
    (void)is_create;
    const std::string image_save_dir = save_root + "/images";
    const std::string label_save_dir = save_root + "/labels";
    mkdir(image_save_dir.c_str(), mode);
    mkdir(label_save_dir.c_str(), mode);
    sdk_clear_txt_files(label_save_dir);
    Algorithm_Sdk::ErrCode ret = Algorithm_Sdk::SUCCESS;
    Algorithm_Sdk::DetectionSDK m_model;
    Algorithm_Sdk::ImageHandle handles[2] = {nullptr, nullptr};
    ret = m_model.init(model_path, max_batch, 0);
    if (ret != Algorithm_Sdk::SUCCESS)
    {
        std::cerr << "model init failed, ret=" << ret << std::endl;
        return ret;
    }
    for (int i = 0; i < 2; ++i)
    {
        ret = m_model.image_handle_create(&handles[i]);
        if (ret != Algorithm_Sdk::SUCCESS)
        {
            std::cerr << "image_handle_create failed, ret=" << ret << std::endl;
            for (int j = 0; j < i; ++j)
                free_handle_if_needed(m_model, &handles[j]);
            return ret;
        }
    }
    std::vector<std::string> image_files;
    if (list_images(test_image_dir, image_files) != 0 || image_files.empty())
    {
        std::cerr << "no images found in " << test_image_dir << std::endl;
        free_handle_if_needed(m_model, &handles[0]);
        free_handle_if_needed(m_model, &handles[1]);
        return Algorithm_Sdk::INPUT_FORMAT_ERR;
    }
    struct BatchSlot
    {
        Algorithm_Sdk::ImageHandle handle = nullptr;
        std::vector<cv::Mat> batch;
        std::vector<std::string> paths;
        double preprocess_ms = 0.0;
        int loop = 0;
        bool valid = false;
    };

    BatchSlot slots[2];
    slots[0].handle = handles[0];
    slots[1].handle = handles[1];

    auto cleanup_handles = [&]() {
        free_handle_if_needed(m_model, &handles[0]);
        free_handle_if_needed(m_model, &handles[1]);
    };

    auto prepare_slot = [&](int loop, int start, BatchSlot &slot) -> Algorithm_Sdk::ErrCode {
        slot.batch.clear();
        slot.paths.clear();
        slot.preprocess_ms = 0.0;
        slot.loop = loop;
        slot.valid = false;
        slot.batch.reserve(max_batch);
        slot.paths.reserve(max_batch);
        for (int i = 0; i < max_batch && start + i < static_cast<int>(image_files.size()); ++i)
        {
            cv::Mat image = cv::imread(image_files[start + i]);
            if (image.empty())
            {
                std::cerr << "read image failed: " << image_files[start + i] << std::endl;
                continue;
            }
            slot.batch.push_back(image);
            slot.paths.push_back(image_files[start + i]);
        }
        if (slot.batch.empty())
            return Algorithm_Sdk::SUCCESS;

        auto time_start = std::chrono::steady_clock::now();
        Algorithm_Sdk::ErrCode code = m_model.image_uniformization(slot.batch, slot.handle);
        auto time_end = std::chrono::steady_clock::now();
        slot.preprocess_ms = std::chrono::duration<double, std::milli>(time_end - time_start).count();
        if (code != Algorithm_Sdk::SUCCESS)
        {
            std::cerr << "image_uniformization failed, ret=" << code << std::endl;
            return code;
        }
        slot.valid = true;
        return Algorithm_Sdk::SUCCESS;
    };

    auto forward_and_save = [&](BatchSlot &slot) -> Algorithm_Sdk::ErrCode {
        if (!slot.valid)
            return Algorithm_Sdk::SUCCESS;
        std::vector<std::vector<Algorithm_Sdk::DetectionResult>> results;
        auto time_start = std::chrono::steady_clock::now();
        Algorithm_Sdk::ErrCode code = m_model.forward(slot.handle, &results, confidence_thresh, 0.50f, false);
        auto time_end = std::chrono::steady_clock::now();
        double forward_ms = std::chrono::duration<double, std::milli>(time_end - time_start).count();
        if (code != Algorithm_Sdk::SUCCESS)
        {
            std::cerr << "forward failed, ret=" << code << std::endl;
            return code;
        }
        std::cout << "[YOLO11] preprocess=" << slot.preprocess_ms
                  << " ms forward=" << forward_ms << " ms" << std::endl;
        for (size_t image_idx = 0; image_idx < results.size(); ++image_idx)
        {
            if (save_detection_result(slot.batch[image_idx], slot.paths[image_idx], results[image_idx],
                                      image_save_dir, label_save_dir) != 0)
            {
                return Algorithm_Sdk::PARAM_ERR;
            }
            std::cout << "[YOLO11] loop=" << slot.loop
                      << " image=" << slot.paths[image_idx]
                      << " objects=" << results[image_idx].size() << std::endl;
        }
        return Algorithm_Sdk::SUCCESS;
    };

    bool has_current = false;
    int current = 0;
    int next = 1;
    for (int loop = 0; loop < loop_count; ++loop)
    {
        for (int start = 0; start < static_cast<int>(image_files.size()); start += max_batch)
        {
            if (!has_current)
            {
                ret = prepare_slot(loop, start, slots[current]);
                if (ret != Algorithm_Sdk::SUCCESS)
                {
                    cleanup_handles();
                    return ret;
                }
                has_current = slots[current].valid;
                continue;
            }

            std::future<Algorithm_Sdk::ErrCode> forward_future = std::async(std::launch::async, [&]() {
                return forward_and_save(slots[current]);
            });

            ret = prepare_slot(loop, start, slots[next]);
            Algorithm_Sdk::ErrCode forward_ret = forward_future.get();
            if (forward_ret != Algorithm_Sdk::SUCCESS)
            {
                cleanup_handles();
                return forward_ret;
            }
            if (ret != Algorithm_Sdk::SUCCESS)
            {
                cleanup_handles();
                return ret;
            }
            if (slots[next].valid)
            {
                current = next;
                next = 1 - current;
                has_current = true;
            }
            else
            {
                has_current = false;
            }
        }
    }

    if (has_current)
    {
        ret = forward_and_save(slots[current]);
        if (ret != Algorithm_Sdk::SUCCESS)
        {
            cleanup_handles();
            return ret;
        }
    }

    ret = m_model.image_handle_free(&handles[0]);
    Algorithm_Sdk::ErrCode ret1 = m_model.image_handle_free(&handles[1]);
    if (ret != Algorithm_Sdk::SUCCESS || ret1 != Algorithm_Sdk::SUCCESS)
    {
        std::cerr << "image_handle_free failed, ret=" << (ret != Algorithm_Sdk::SUCCESS ? ret : ret1) << std::endl;
        return ret != Algorithm_Sdk::SUCCESS ? ret : ret1;
    }
    return Algorithm_Sdk::SUCCESS;
}
