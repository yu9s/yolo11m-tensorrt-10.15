#include "algorithm_sdk.h"
#include "yolo11_engine.h"
#include <cuda_runtime.h>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
namespace Algorithm_Sdk
{
    struct LetterboxInfo
    {
        float scale = 1.0f;
        int pad_left = 0;
        int pad_top = 0;
    };

    struct ImageBuffer
    {
        static constexpr int kMagic = 0x59313143;
        int magic = kMagic;
        int batch = 0;
        int width = 0;
        int height = 0;
        int channel = 0;
        std::vector<int> src_widths;
        std::vector<int> src_heights;
        std::vector<LetterboxInfo> letterbox_info;
        void *image_data_d = nullptr;
        size_t image_data_d_bytes = 0;
        void *input_host = nullptr;
        size_t input_host_bytes = 0;
        cudaStream_t upload_stream = nullptr;
        cudaEvent_t input_ready = nullptr;
        bool input_ready_recorded = false;
        std::vector<cv::Mat> letterbox_images;
        std::vector<void *> output_device;
        std::vector<void *> output_host;
    };
    namespace
    {
        struct BoxInfo
        {
            float x1;
            float y1;
            float x2;
            float y2;
            float score;
            int label;
        };
        inline float fp16_to_float(uint16_t h)
        {
            const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
            uint32_t exp = (h >> 10) & 0x1fu;
            uint32_t mant = h & 0x03ffu;
            uint32_t bits = 0;
            if (exp == 0)
            {
                if (mant == 0)
                {
                    bits = sign;
                }
                else
                {
                    exp = 1;
                    while ((mant & 0x0400u) == 0)
                    {
                        mant <<= 1;
                        --exp;
                    }
                    mant &= 0x03ffu;
                    bits = sign | ((exp + 112u) << 23) | (mant << 13);
                }
            }
            else if (exp == 31)
            {
                bits = sign | 0x7f800000u | (mant << 13);
            }
            else
            { 
                bits = sign | ((exp + 112u) << 23) | (mant << 13);
            }
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }
        void letter_box(const cv::Mat &image, cv::Size new_shape, cv::Mat &output, LetterboxInfo *info = nullptr, cv::Scalar color = cv::Scalar(114, 114, 114))
        {
            const float r = std::min(new_shape.height / static_cast<float>(image.rows),
                                     new_shape.width / static_cast<float>(image.cols));
            const int new_unpad_w = static_cast<int>(std::round(image.cols * r));
            const int new_unpad_h = static_cast<int>(std::round(image.rows * r));
            const float dw = static_cast<float>(new_shape.width - new_unpad_w) * 0.5f;
            const float dh = static_cast<float>(new_shape.height - new_unpad_h) * 0.5f;
            const int top = static_cast<int>(std::round(dh - 0.1f));
            const int left = static_cast<int>(std::round(dw - 0.1f));
            if (info)
            {
                info->scale = r;
                info->pad_left = left;
                info->pad_top = top;
            }

            output.create(new_shape, image.type());
            output.setTo(color);
            cv::Mat roi = output(cv::Rect(left, top, new_unpad_w, new_unpad_h));
            if (image.cols != new_unpad_w || image.rows != new_unpad_h)
            {
                cv::resize(image, roi, cv::Size(new_unpad_w, new_unpad_h), 0.0, 0.0, cv::INTER_LINEAR);
            }
            else
            {
                image.copyTo(roi);
            }
        }
        inline cv::Mat letterBox(const cv::Mat &image, cv::Size newShape, cv::Scalar color = cv::Scalar(114, 114, 114))
        {
            cv::Mat output;
            letter_box(image, newShape, output, nullptr, color);
            return output;
        }

        template <typename VecU32>
        inline cv::v_float32 u32_to_scaled_f32(const VecU32 &value, const cv::v_float32 &scale)
        {
            return cv::v_mul(cv::v_cvt_f32(cv::v_reinterpret_as_s32(value)), scale);
        }

        void bgr_to_rgb_float_nchw_simd(const cv::Mat &src, float *dst)
        {
            const int width = src.cols;
            const int height = src.rows;
            const int plane = width * height;
            float *r_plane = dst;
            float *g_plane = dst + plane;
            float *b_plane = dst + plane * 2;
            constexpr float inv255 = 1.0f / 255.0f;

            for (int y = 0; y < height; ++y)
            {
                const uchar *row = src.ptr<uchar>(y);
                const int offset = y * width;
                int x = 0;
#if CV_SIMD
                const cv::v_float32 scale = cv::vx_setall_f32(inv255);
                constexpr int lanes8 = cv::VTraits<cv::v_uint8>::nlanes;
                for (; x <= width - lanes8; x += lanes8)
                {
                    cv::v_uint8 b8, g8, r8;
                    cv::v_load_deinterleave(row + x * 3, b8, g8, r8);
                    cv::v_uint16 b16_l, b16_h, g16_l, g16_h, r16_l, r16_h;
                    cv::v_expand(b8, b16_l, b16_h);
                    cv::v_expand(g8, g16_l, g16_h);
                    cv::v_expand(r8, r16_l, r16_h);
                    cv::v_uint32 b32_0, b32_1, b32_2, b32_3;
                    cv::v_uint32 g32_0, g32_1, g32_2, g32_3;
                    cv::v_uint32 r32_0, r32_1, r32_2, r32_3;
                    cv::v_expand(b16_l, b32_0, b32_1);
                    cv::v_expand(b16_h, b32_2, b32_3);
                    cv::v_expand(g16_l, g32_0, g32_1);
                    cv::v_expand(g16_h, g32_2, g32_3);
                    cv::v_expand(r16_l, r32_0, r32_1);
                    cv::v_expand(r16_h, r32_2, r32_3);
                    float *r_out = r_plane + offset + x;
                    float *g_out = g_plane + offset + x;
                    float *b_out = b_plane + offset + x;
                    cv::v_store(r_out, u32_to_scaled_f32(r32_0, scale));
                    cv::v_store(r_out + cv::VTraits<cv::v_float32>::nlanes, u32_to_scaled_f32(r32_1, scale));
                    cv::v_store(r_out + cv::VTraits<cv::v_float32>::nlanes * 2, u32_to_scaled_f32(r32_2, scale));
                    cv::v_store(r_out + cv::VTraits<cv::v_float32>::nlanes * 3, u32_to_scaled_f32(r32_3, scale));
                    cv::v_store(g_out, u32_to_scaled_f32(g32_0, scale));
                    cv::v_store(g_out + cv::VTraits<cv::v_float32>::nlanes, u32_to_scaled_f32(g32_1, scale));
                    cv::v_store(g_out + cv::VTraits<cv::v_float32>::nlanes * 2, u32_to_scaled_f32(g32_2, scale));
                    cv::v_store(g_out + cv::VTraits<cv::v_float32>::nlanes * 3, u32_to_scaled_f32(g32_3, scale));
                    cv::v_store(b_out, u32_to_scaled_f32(b32_0, scale));
                    cv::v_store(b_out + cv::VTraits<cv::v_float32>::nlanes, u32_to_scaled_f32(b32_1, scale));
                    cv::v_store(b_out + cv::VTraits<cv::v_float32>::nlanes * 2, u32_to_scaled_f32(b32_2, scale));
                    cv::v_store(b_out + cv::VTraits<cv::v_float32>::nlanes * 3, u32_to_scaled_f32(b32_3, scale));
                }
#endif
                for (; x < width; ++x)
                {
                    const uchar *p = row + x * 3;
                    r_plane[offset + x] = static_cast<float>(p[2]) * inv255;
                    g_plane[offset + x] = static_cast<float>(p[1]) * inv255;
                    b_plane[offset + x] = static_cast<float>(p[0]) * inv255;
                }
            }
        }

        void bgr_to_rgb_float_nhwc_simd(const cv::Mat &src, float *dst)
        {
            const int total = src.rows * src.cols;
            const uchar *input = src.ptr<uchar>(0);
            constexpr float inv255 = 1.0f / 255.0f;
            int i = 0;
#if CV_SIMD
            const cv::v_float32 scale = cv::vx_setall_f32(inv255);
            constexpr int lanes8 = cv::VTraits<cv::v_uint8>::nlanes;
            for (; i <= total - lanes8; i += lanes8)
            {
                cv::v_uint8 b8, g8, r8;
                cv::v_load_deinterleave(input + i * 3, b8, g8, r8);
                cv::v_uint16 b16_l, b16_h, g16_l, g16_h, r16_l, r16_h;
                cv::v_expand(b8, b16_l, b16_h);
                cv::v_expand(g8, g16_l, g16_h);
                cv::v_expand(r8, r16_l, r16_h);
                cv::v_uint32 b32_0, b32_1, b32_2, b32_3;
                cv::v_uint32 g32_0, g32_1, g32_2, g32_3;
                cv::v_uint32 r32_0, r32_1, r32_2, r32_3;
                cv::v_expand(b16_l, b32_0, b32_1);
                cv::v_expand(b16_h, b32_2, b32_3);
                cv::v_expand(g16_l, g32_0, g32_1);
                cv::v_expand(g16_h, g32_2, g32_3);
                cv::v_expand(r16_l, r32_0, r32_1);
                cv::v_expand(r16_h, r32_2, r32_3);
                float *out = dst + i * 3;
                cv::v_store_interleave(out, u32_to_scaled_f32(r32_0, scale), u32_to_scaled_f32(g32_0, scale), u32_to_scaled_f32(b32_0, scale));
                cv::v_store_interleave(out + cv::VTraits<cv::v_float32>::nlanes * 3, u32_to_scaled_f32(r32_1, scale), u32_to_scaled_f32(g32_1, scale), u32_to_scaled_f32(b32_1, scale));
                cv::v_store_interleave(out + cv::VTraits<cv::v_float32>::nlanes * 6, u32_to_scaled_f32(r32_2, scale), u32_to_scaled_f32(g32_2, scale), u32_to_scaled_f32(b32_2, scale));
                cv::v_store_interleave(out + cv::VTraits<cv::v_float32>::nlanes * 9, u32_to_scaled_f32(r32_3, scale), u32_to_scaled_f32(g32_3, scale), u32_to_scaled_f32(b32_3, scale));
            }
#endif
            for (; i < total; ++i)
            {
                const uchar *p = input + i * 3;
                float *out = dst + i * 3;
                out[0] = static_cast<float>(p[2]) * inv255;
                out[1] = static_cast<float>(p[1]) * inv255;
                out[2] = static_cast<float>(p[0]) * inv255;
            }
        }

        void image_preprocess_simd(const cv::Mat &src, float *dst, bool nchw)
        {
            CV_Assert(src.type() == CV_8UC3);
            CV_Assert(src.isContinuous());
            if (nchw)
            {
                bgr_to_rgb_float_nchw_simd(src, dst);
            }
            else
            {
                bgr_to_rgb_float_nhwc_simd(src, dst);
            }
        }
        inline void apply_nms(std::vector<BoxInfo> &boxes, float thresh)
        {
            if (boxes.empty())
            {
                return;
            }
            std::sort(boxes.begin(), boxes.end(),
                      [](const BoxInfo &a, const BoxInfo &b)
                      { return a.score > b.score; });

            std::vector<float> areas(boxes.size());
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                areas[i] = std::max(0.0f, boxes[i].x2 - boxes[i].x1) *
                           std::max(0.0f, boxes[i].y2 - boxes[i].y1);
            }

            std::vector<unsigned char> suppressed(boxes.size(), 0);
            std::vector<BoxInfo> kept;
            kept.reserve(boxes.size());
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                if (suppressed[i])
                {
                    continue;
                }
                kept.push_back(boxes[i]);
                for (size_t j = i + 1; j < boxes.size(); ++j)
                {
                    if (suppressed[j] || boxes[i].label != boxes[j].label)
                    {
                        continue;
                    }
                    float xx1 = std::max(boxes[i].x1, boxes[j].x1);
                    float yy1 = std::max(boxes[i].y1, boxes[j].y1);
                    float xx2 = std::min(boxes[i].x2, boxes[j].x2);
                    float yy2 = std::min(boxes[i].y2, boxes[j].y2);
                    float w = std::max(0.0f, xx2 - xx1);
                    float h = std::max(0.0f, yy2 - yy1);
                    float inter = w * h;
                    float denom = areas[i] + areas[j] - inter;
                    float ovr = denom > 0.0f ? inter / denom : 0.0f;
                    if (ovr > thresh)
                    {
                        suppressed[j] = 1;
                    }
                }
            }
            boxes.swap(kept);
        }
    }
    class DetectionSDK::Impl
    {
    public:
        Engine engine;
        bool initialized = false;
        int max_batch = 1;
        int input_w = 640;
        int input_h = 640;
        int input_c = 3;
        bool nchw = true;
        int output_dims = 0;
        std::vector<int64_t> output_shape;
        int output_numel = 0;
        int output_bytes_per_elem = 4;
        cudaStream_t stream = nullptr;
        Impl() {}
        ~Impl()
        {
            if (stream)
            {
                cudaStreamDestroy(stream);
                stream = nullptr;
            }
        }
        void release_image(ImageBuffer *buf)
        {
            if (!buf)
                return;
            if (buf->image_data_d)
            {
                cudaFree(buf->image_data_d);
                buf->image_data_d = nullptr;
                buf->image_data_d_bytes = 0;
            }
            if (buf->input_host)
            {
                cudaFreeHost(buf->input_host);
                buf->input_host = nullptr;
                buf->input_host_bytes = 0;
            }
            if (buf->input_ready)
            {
                cudaEventDestroy(buf->input_ready);
                buf->input_ready = nullptr;
                buf->input_ready_recorded = false;
            }
            if (buf->upload_stream)
            {
                cudaStreamDestroy(buf->upload_stream);
                buf->upload_stream = nullptr;
            }
            buf->letterbox_images.clear();
            for (size_t i = 0; i < buf->output_device.size(); ++i)
            {
                if (buf->output_device[i])
                {
                    cudaFree(buf->output_device[i]);
                    buf->output_device[i] = nullptr;
                }
                if (i < buf->output_host.size() && buf->output_host[i])
                {
                    free(buf->output_host[i]);
                    buf->output_host[i] = nullptr;
                }
            }
            buf->output_device.clear();
            buf->output_host.clear();
        }
    };
    DetectionSDK::DetectionSDK() : impl_(new Impl) {}
    DetectionSDK::~DetectionSDK() {}
    ErrCode DetectionSDK::init(const std::string &model_path, int max_batch_size, int device_id)
    {
        if (max_batch_size <= 0)
        {
            return PARAM_ERR;
        }
        ErrCode ret = impl_->engine.init(model_path, device_id);
        if (ret != SUCCESS)
        {
            return ret;
        }
        if (impl_->stream == nullptr)
        {
            if (cudaStreamCreate(&impl_->stream) != cudaSuccess)
            {
                return CUDA_RUNTIME_ERR;
            }
        }
        auto in_shape = impl_->engine.input_shape(0);
        if (in_shape.size() != 4)
        {
            return MODEL_ERR;
        }
        if (in_shape[1] == 1 || in_shape[1] == 3)
        {
            impl_->nchw = true;
            impl_->input_c = static_cast<int>(in_shape[1]);
            impl_->input_h = static_cast<int>(in_shape[2]);
            impl_->input_w = static_cast<int>(in_shape[3]);
        }
        else if (in_shape[3] == 1 || in_shape[3] == 3)
        {
            impl_->nchw = false;
            impl_->input_h = static_cast<int>(in_shape[1]);
            impl_->input_w = static_cast<int>(in_shape[2]);
            impl_->input_c = static_cast<int>(in_shape[3]);
        }
        else
        {
            return MODEL_ERR;
        }
        if (impl_->input_c != 3 || impl_->input_h <= 0 || impl_->input_w <= 0)
        {
            return MODEL_ERR;
        }
        if (impl_->engine.output_count() != 1)
        {
            return MODEL_ERR;
        }
        impl_->output_shape = impl_->engine.output_shape(0);
        impl_->output_dims = static_cast<int>(impl_->output_shape.size());
        if (impl_->output_dims < 2)
        {
            return MODEL_ERR;
        }
        impl_->output_numel = 1;
        for (int i = 1; i < impl_->output_dims; ++i)
        {
            impl_->output_numel *= static_cast<int>(impl_->output_shape[i]);
        }
        impl_->output_bytes_per_elem = impl_->engine.output_dtype_size(0);
        int model_max = impl_->engine.max_batch_size();
        if (max_batch_size > model_max)
        {
            std::cerr << "max_batch_size " << max_batch_size
                      << " > model max " << model_max << std::endl;
            return INIT_ERR;
        }
        impl_->max_batch = max_batch_size;
        impl_->initialized = true;
        {
            ImageHandle warm = nullptr;
            if (image_handle_create(&warm) == SUCCESS)
            {
                std::vector<cv::Mat> dummy;
                cv::Mat img = cv::Mat::ones(cv::Size(impl_->input_w, impl_->input_h), CV_8UC3);
                for (int i = 0; i < impl_->max_batch; ++i)
                {
                    dummy.push_back(img);
                }
                std::vector<std::vector<DetectionResult>> tmp;
                if (image_uniformization(dummy, warm) == SUCCESS)
                {
                    forward(warm, &tmp, 0.99f, 0.5f, false);
                }
                image_handle_free(&warm);
            }
        }
        return SUCCESS;
    }
    ErrCode DetectionSDK::image_handle_create(ImageHandle *handle)
    {
        if (!handle)
            return PARAM_ERR;
        *handle = new ImageBuffer();
        return SUCCESS;
    }
    ErrCode DetectionSDK::image_handle_free(ImageHandle *handle)
    {
        if (!handle || !*handle)
            return PARAM_ERR;
        if ((*handle)->magic != ImageBuffer::kMagic)
            return PARAM_ERR;
        impl_->release_image(*handle);
        delete *handle;
        *handle = nullptr;
        return SUCCESS;
    }
    ErrCode DetectionSDK::image_uniformization(const cv::Mat &input, ImageHandle handle)
    {
        return image_uniformization(std::vector<cv::Mat>{input}, handle);
    }
    ErrCode DetectionSDK::image_uniformization(const std::vector<cv::Mat> &inputs, ImageHandle handle)
    {
        if (!impl_->initialized)
        {
            return INIT_MODEL_ERR;
        }
        if (handle == nullptr || handle->magic != ImageBuffer::kMagic)
        {
            return PARAM_ERR;
        }
        if (inputs.empty())
        {
            return INPUT_FORMAT_ERR;
        }
        if (static_cast<int>(inputs.size()) > impl_->max_batch)
        {
            return BATCHSIZE_ERR;
        }
        const int batch = static_cast<int>(inputs.size());
        const size_t image_numel = static_cast<size_t>(impl_->input_w) *
                                   impl_->input_h * impl_->input_c;
        const size_t image_bytes = static_cast<size_t>(batch) * image_numel * sizeof(float);
        handle->src_widths.clear();
        handle->src_heights.clear();
        handle->letterbox_info.clear();
        handle->width = impl_->input_w;
        handle->height = impl_->input_h;
        handle->channel = impl_->input_c;

        if (handle->image_data_d == nullptr || handle->image_data_d_bytes < image_bytes)
        {
            if (handle->image_data_d)
            {
                cudaFree(handle->image_data_d);
                handle->image_data_d = nullptr;
                handle->image_data_d_bytes = 0;
            }
            if (cudaMalloc(&handle->image_data_d, image_bytes) != cudaSuccess)
            {
                return CUDA_MEMORY_ERR;
            }
            handle->image_data_d_bytes = image_bytes;
        }

        if (handle->input_host == nullptr || handle->input_host_bytes < image_bytes)
        {
            if (handle->input_host)
            {
                cudaFreeHost(handle->input_host);
                handle->input_host = nullptr;
                handle->input_host_bytes = 0;
            }
            if (cudaMallocHost(&handle->input_host, image_bytes) != cudaSuccess)
            {
                return CUDA_MEMORY_ERR;
            }
            handle->input_host_bytes = image_bytes;
        }
        if (handle->upload_stream == nullptr)
        {
            if (cudaStreamCreateWithFlags(&handle->upload_stream, cudaStreamNonBlocking) != cudaSuccess)
            {
                return CUDA_RUNTIME_ERR;
            }
        }
        if (handle->input_ready == nullptr)
        {
            if (cudaEventCreateWithFlags(&handle->input_ready, cudaEventDisableTiming) != cudaSuccess)
            {
                return CUDA_RUNTIME_ERR;
            }
        }

        try
        {
            const bool profile_preprocess = std::getenv("YOLO_PREPROCESS_PROFILE") != nullptr;
            double letterbox_ms = 0.0;
            double blob_ms = 0.0;
            float *host_input = static_cast<float *>(handle->input_host);
            handle->letterbox_images.resize(batch);
            for (int b = 0; b < batch; ++b)
            {
                const cv::Mat &src = inputs[b];
                if (src.empty() || src.channels() != 3)
                {
                    return INPUT_FORMAT_ERR;
                }
                handle->src_widths.push_back(src.cols);
                handle->src_heights.push_back(src.rows);

                LetterboxInfo info;
                cv::Mat &letterbox_image = handle->letterbox_images[b];
                const auto letterbox_start = std::chrono::steady_clock::now();
                letter_box(src, cv::Size(impl_->input_w, impl_->input_h), letterbox_image, &info);
                if (profile_preprocess)
                {
                    const auto letterbox_end = std::chrono::steady_clock::now();
                    letterbox_ms += std::chrono::duration<double, std::milli>(letterbox_end - letterbox_start).count();
                }

                float *dst = host_input + static_cast<size_t>(b) * image_numel;
                const auto blob_start = std::chrono::steady_clock::now();
                image_preprocess_simd(letterbox_image, dst, impl_->nchw);
                if (profile_preprocess)
                {
                    const auto blob_end = std::chrono::steady_clock::now();
                    blob_ms += std::chrono::duration<double, std::milli>(blob_end - blob_start).count();
                }
                handle->letterbox_info.push_back(info);
            }

            const auto h2d_start = std::chrono::steady_clock::now();
            if (cudaMemcpyAsync(handle->image_data_d, host_input, image_bytes,
                                cudaMemcpyHostToDevice, handle->upload_stream) != cudaSuccess)
            {
                return CUDA_MEMORY_ERR;
            }
            if (cudaEventRecord(handle->input_ready, handle->upload_stream) != cudaSuccess)
            {
                return CUDA_RUNTIME_ERR;
            }
            handle->input_ready_recorded = true;
            if (profile_preprocess)
            {
                if (cudaStreamSynchronize(handle->upload_stream) != cudaSuccess)
                {      
                    return CUDA_RUNTIME_ERR;
                }
                const auto h2d_end = std::chrono::steady_clock::now();
                const double h2d_ms = std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
                std::fprintf(stderr, "[YOLO11][preprocess_profile] batch=%d letterbox=%.3f ms blob=%.3f ms h2d_sync=%.3f ms\n",
                             batch, letterbox_ms, blob_ms, h2d_ms);
            }
        }
        catch (...)
        {
            return IMAGE_PREPROCESS_ERR;
        }
        handle->batch = batch;
        if (handle->output_device.empty())
        {
            size_t bytes = static_cast<size_t>(batch) *
                           impl_->output_numel *
                           impl_->output_bytes_per_elem;
            void *dev = nullptr;
            void *host = malloc(bytes);
            if (host == nullptr || cudaMalloc(&dev, bytes) != cudaSuccess)
            {
                if (host)
                    free(host);
                return CUDA_MEMORY_ERR;
            }
            handle->output_host.push_back(host);
            handle->output_device.push_back(dev);
        }
        return SUCCESS;
    }
    ErrCode DetectionSDK::forward(ImageHandle handle,
                                  std::vector<std::vector<DetectionResult>> *results,
                                  float confidence_thresh,
                                  float nms_thresh,
                                  bool verbose)
    {
        if (!impl_->initialized)
        {
            return INIT_MODEL_ERR;
        }
        if (handle == nullptr || handle->magic != ImageBuffer::kMagic || results == nullptr)
        {
            return PARAM_ERR;
        }
        if (handle->batch <= 0 || handle->batch > impl_->max_batch)
        {
            return BATCHSIZE_ERR;
        }
        results->clear();
        try 
        {
            if (handle->input_ready_recorded)
            {
                if (cudaStreamWaitEvent(impl_->stream, handle->input_ready, 0) != cudaSuccess)
                {
                    return CUDA_RUNTIME_ERR;
                }
            }
            void *output_d = handle->output_device[0];
            void *output_raw = handle->output_host[0];
            std::vector<void *> buffers = {handle->image_data_d, output_d};
            auto t0 = std::chrono::steady_clock::now();
            ErrCode ret = impl_->engine.infer(buffers, handle->batch, impl_->stream);
            if (ret != SUCCESS)
            {
                return ret;
            }
            size_t bytes = static_cast<size_t>(handle->batch) *
                           impl_->output_numel *
                           impl_->output_bytes_per_elem;
            if (cudaMemcpyAsync(output_raw, output_d, bytes,
                                cudaMemcpyDeviceToHost, impl_->stream) != cudaSuccess)
            {
                return CUDA_MEMORY_ERR;
            }
            if (cudaStreamSynchronize(impl_->stream) != cudaSuccess)
            {
                return CUDA_RUNTIME_ERR;
            }
            if (verbose)
            {
                auto t1 = std::chrono::steady_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                std::cout << "[YOLO11] infer " << ms << " ms" << std::endl;
            }
            const float *output_fp32 = nullptr;
            const uint16_t *output_fp16 = nullptr;
            if (impl_->output_bytes_per_elem == 4)
            {
                output_fp32 = static_cast<const float *>(output_raw);
            }
            else if (impl_->output_bytes_per_elem == 2)
            {
                output_fp16 = static_cast<const uint16_t *>(output_raw);
            }
            else
            {
                std::cerr << "unsupported output dtype size: " << impl_->output_bytes_per_elem << std::endl;
                return MODEL_ERR;
            }
            auto output_at = [output_fp32, output_fp16](size_t index) -> float
            {
                return output_fp32 ? output_fp32[index] : fp16_to_float(output_fp16[index]);
            };
            int channel_num = static_cast<int>(impl_->output_shape[impl_->output_dims - 2]);
            int box_num = static_cast<int>(impl_->output_shape[impl_->output_dims - 1]);
            int class_num = channel_num - 4;
            if (class_num <= 0 || box_num <= 0)
            {
                return MODEL_ERR;
            }
            for (int b = 0; b < handle->batch; ++b)
            {
                std::vector<DetectionResult> mobjects;
                const size_t batch_base = static_cast<size_t>(b) * impl_->output_numel;

                const LetterboxInfo &info = handle->letterbox_info[b];
                float scale = info.scale;
                float pad_x = static_cast<float>(info.pad_left);
                float pad_y = static_cast<float>(info.pad_top);

                std::vector<BoxInfo> cand;
                for (int i = 0; i < box_num; ++i)
                {
                    float max_score = output_at(batch_base + static_cast<size_t>(4) * box_num + i);
                    int class_id = 0;
                    for (int c = 1; c < class_num; ++c)
                    {
                        float s = output_at(batch_base + static_cast<size_t>(4 + c) * box_num + i);
                        if (s > max_score)
                        {
                            max_score = s;
                            class_id = c;
                        }
                    }
                    if (max_score <= confidence_thresh)
                        continue;
                    float cx = output_at(batch_base + static_cast<size_t>(0) * box_num + i);
                    float cy = output_at(batch_base + static_cast<size_t>(1) * box_num + i);
                    float w = output_at(batch_base + static_cast<size_t>(2) * box_num + i);
                    float h = output_at(batch_base + static_cast<size_t>(3) * box_num + i);
                    BoxInfo info;
                    info.x1 = cx - w * 0.5f;
                    info.y1 = cy - h * 0.5f;
                    info.x2 = cx + w * 0.5f;
                    info.y2 = cy + h * 0.5f;
                    info.score = max_score;
                    info.label = class_id;
                    cand.push_back(info);
                }
                apply_nms(cand, nms_thresh);
                for (auto &c : cand)
                {
                    float xmin = (c.x1 - pad_x) / scale;
                    float ymin = (c.y1 - pad_y) / scale;
                    float xmax = (c.x2 - pad_x) / scale;
                    float ymax = (c.y2 - pad_y) / scale;
                    xmin = std::round(xmin);
                    ymin = std::round(ymin);
                    xmax = std::round(xmax);
                    ymax = std::round(ymax);
                    xmin = std::max(0.0f, std::min(static_cast<float>(handle->src_widths[b]), xmin));
                    xmax = std::max(0.0f, std::min(static_cast<float>(handle->src_widths[b]), xmax));
                    ymin = std::max(0.0f, std::min(static_cast<float>(handle->src_heights[b]), ymin));
                    ymax = std::max(0.0f, std::min(static_cast<float>(handle->src_heights[b]), ymax));
                    if (xmax - xmin <= 0.0f || ymax - ymin <= 0.0f)
                        continue;
                    DetectionResult obj;
                    obj.score = c.score;
                    obj.class_id = c.label;
                    obj.box.x = static_cast<int>(xmin);
                    obj.box.y = static_cast<int>(ymin);
                    obj.box.width = static_cast<int>(xmax - xmin);
                    obj.box.height = static_cast<int>(ymax - ymin);
                    mobjects.push_back(obj);
                }
                results->push_back(mobjects);
            }
            return SUCCESS;
        }
        catch (std::exception &e)
        {
            std::cerr << "[YOLO11] forward exception: " << e.what() << std::endl;
            return POSTPROCESS_ERR;
        }
    }
    bool DetectionSDK::is_initialized() const
    {
        return impl_->initialized;
    }
}
