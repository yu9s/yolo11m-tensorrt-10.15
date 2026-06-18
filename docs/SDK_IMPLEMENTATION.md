# YOLO11 TensorRT SDK 实现说明

本文档总结当前 SDK 的模块划分、公有/私有接口、CUDA/TensorRT 调用路径、图像预处理和后处理流程。代码入口主要位于 `include/algorithm_sdk.h`、`include/algorithm_sdk_types.h`、`src/yolo11_sdk.cpp`、`src/yolo11_engine.cpp`。

## 1. 总体架构

SDK 对外暴露 `Algorithm_Sdk::DetectionSDK`，内部通过 PImpl 隐藏实现细节。整体分为三层：

- 公有 SDK 层：`DetectionSDK`，负责模型初始化、图像句柄管理、预处理、推理、后处理。
- TensorRT Engine 层：`Engine`，负责反序列化 TensorRT engine、创建 execution context、设置 I/O tensor address、调用 `enqueueV3`。
- 数据与 CUDA 资源层：`ImageBuffer`，保存一次或一批图像预处理后的输入、输出 buffer、CUDA stream/event、letterbox 映射信息。

典型调用流程：

```text
DetectionSDK::init()
  -> Engine::init()
  -> 解析输入/输出 shape
  -> 创建主推理 cudaStream
  -> 执行一次 warmup

image_handle_create()
  -> 创建 ImageBuffer 句柄

image_uniformization(images, handle)
  -> letterbox
  -> BGR->RGB + normalize + layout 转换
  -> cudaMemcpyAsync H2D
  -> cudaEventRecord(input_ready)

forward(handle, results, conf, nms)
  -> cudaStreamWaitEvent 等待 H2D 完成
  -> Engine::infer/enqueueV3
  -> cudaMemcpyAsync D2H
  -> cudaStreamSynchronize
  -> decode + NMS + 坐标映射回原图

image_handle_free()
  -> 释放 ImageBuffer 内部 CUDA/CPU 资源
```

## 2. 公有接口

公有接口定义在 `include/algorithm_sdk.h`。

### 2.1 DetectionSDK

```cpp
class DetectionSDK
{
public:
    DetectionSDK();
    ~DetectionSDK();
    DetectionSDK(const DetectionSDK &) = delete;

    ErrCode init(const std::string &model_path,
                 int max_batch_size = 1,
                 int device_id = 0);

    ErrCode image_handle_create(ImageHandle *handle);
    ErrCode image_handle_free(ImageHandle *handle);

    ErrCode image_uniformization(const cv::Mat &input,
                                 ImageHandle handle);
    ErrCode image_uniformization(const std::vector<cv::Mat> &inputs,
                                 ImageHandle handle);

    ErrCode forward(ImageHandle handle,
                    std::vector<std::vector<DetectionResult>> *results,
                    float confidence_thresh,
                    float nms_thresh,
                    bool verbose = false);

    bool is_initialized() const;
};
```

### 2.2 init

职责：

- 读取并初始化 TensorRT engine。
- 创建 SDK 主推理 stream。
- 解析模型输入 shape，支持 NCHW 和 NHWC。
- 校验输入通道必须为 3。
- 校验模型只有一个输出。
- 记录输出 shape、输出元素数量、输出数据类型大小。
- 校验用户传入的 `max_batch_size` 不超过 engine profile 最大 batch。
- 执行一次 dummy warmup。

当前输入 shape 判断逻辑：

- 如果 `shape[1] == 1 || shape[1] == 3`，按 NCHW 处理。
- 如果 `shape[3] == 1 || shape[3] == 3`，按 NHWC 处理。
- 当前实际要求 `input_c == 3`。

### 2.3 image_handle_create / image_handle_free

`ImageHandle` 是不透明句柄，类型定义为：

```cpp
struct ImageBuffer;
using ImageHandle = ImageBuffer *;
```

用户只能持有句柄，不能访问内部字段。

- `image_handle_create`：new 一个 `ImageBuffer`。
- `image_handle_free`：校验 magic number，释放内部 CUDA/CPU 资源，然后 delete 句柄并置空。

### 2.4 image_uniformization

职责是完成图像预处理，并把输入上传到 GPU。支持单图和 batch：

```cpp
ErrCode image_uniformization(const cv::Mat &input, ImageHandle handle);
ErrCode image_uniformization(const std::vector<cv::Mat> &inputs, ImageHandle handle);
```

输入要求：

- SDK 必须已初始化。
- `handle` 必须由 `image_handle_create` 创建。
- 输入不能为空。
- batch size 不能超过 `max_batch`。
- 每张图必须是 3 通道 OpenCV `cv::Mat`。

输出效果：

- `handle->image_data_d` 中保存 GPU 输入 tensor。
- `handle->input_ready` 事件记录 H2D 上传完成状态。
- `handle->letterbox_info` 保存缩放比例和 padding，供后处理坐标还原。
- `handle->src_widths/src_heights` 保存原图大小。

### 2.5 forward

职责是执行推理并后处理。

```cpp
ErrCode forward(ImageHandle handle,
                std::vector<std::vector<DetectionResult>> *results,
                float confidence_thresh,
                float nms_thresh,
                bool verbose = false);
```

输出 `results` 的结构为 batch 维度：

```cpp
std::vector<std::vector<DetectionResult>> results;
results[batch_index][object_index]
```

每个 `DetectionResult` 包含：

```cpp
struct BoundingBox
{
    int x;
    int y;
    int width;
    int height;
};

struct DetectionResult
{
    BoundingBox box;
    float score;
    int class_id;
};
```

## 3. 私有接口和内部数据结构

私有实现主要在 `src/yolo11_sdk.cpp` 的匿名 namespace 和 `DetectionSDK::Impl` 中。

### 3.1 DetectionSDK::Impl

```cpp
class DetectionSDK::Impl
{
public:
    Engine engine;
    bool initialized;
    int max_batch;
    int input_w;
    int input_h;
    int input_c;
    bool nchw;
    int output_dims;
    std::vector<int64_t> output_shape;
    int output_numel;
    int output_bytes_per_elem;
    cudaStream_t stream;

    void release_image(ImageBuffer *buf);
};
```

职责：

- 持有 TensorRT `Engine`。
- 保存模型输入输出元信息。
- 持有主推理 CUDA stream。
- 统一释放 `ImageBuffer` 内部资源。

### 3.2 ImageBuffer

`ImageBuffer` 是每个图像句柄背后的真实数据结构。

关键字段：

- `batch/width/height/channel`：当前 batch 和模型输入尺寸。
- `src_widths/src_heights`：原始图像尺寸。
- `letterbox_info`：每张图的缩放比例和 padding。
- `image_data_d`：GPU 输入 buffer。
- `input_host`：pinned host 输入 buffer，由 `cudaMallocHost` 分配。
- `upload_stream`：非阻塞 H2D 上传 stream。
- `input_ready`：H2D 完成事件。
- `letterbox_images`：CPU 侧 letterbox 后的 BGR 图像缓存。
- `output_device`：GPU 输出 buffer。
- `output_host`：CPU 输出 buffer。

这个设计允许一个 SDK 实例创建多个 `ImageHandle`，从而做双缓冲或多缓冲流水线。测试代码中已经使用两个 handle 实现预处理和推理 overlap。

### 3.3 LetterboxInfo

```cpp
struct LetterboxInfo
{
    float scale;
    int pad_left;
    int pad_top;
};
```

用于把模型输入坐标映射回原图坐标。

## 4. TensorRT Engine 封装

内部类 `Engine` 定义在 `src/internal/yolo11_engine.h`，实现位于 `src/yolo11_engine.cpp`。

### 4.1 Engine::init

主要步骤：

1. `cudaGetDeviceCount` 检查 GPU 数量。
2. `cudaSetDevice(device_id)` 设置当前 GPU。
3. 读取 `.trt` engine 文件到内存。
4. `nvinfer1::createInferRuntime` 创建 runtime。
5. `runtime_->deserializeCudaEngine` 反序列化 engine。
6. 遍历 I/O tensor，统计输入输出数量。
7. 判断 batch 维是否动态：`dims0.d[0] == -1`。
8. `engine_->createExecutionContext()` 创建 execution context。

### 4.2 Engine::infer

主要步骤：

1. 使用 `std::mutex` 加锁，保护同一个 execution context。
2. 动态 batch 时调用 `context_->setInputShape`。
3. 校验 buffers 数量等于 TensorRT I/O tensor 数量。
4. 遍历 I/O tensor，调用 `context_->setTensorAddress(name, buffers[i])`。
5. 调用 `context_->enqueueV3(stream)` 异步入队推理。

注意：`enqueueV3` 是异步的，真正完成由外层 `cudaStreamSynchronize` 或事件控制。

### 4.3 Engine 查询接口

- `input_count()` / `output_count()`：输入输出 tensor 数量。
- `input_shape(index)` / `output_shape(index)`：指定输入输出 shape。
- `output_dtype(index)`：输出数据类型。
- `output_dtype_size(index)`：输出元素字节数。
- `max_batch_size()`：静态 batch 返回输入 shape batch；动态 batch 返回 profile 的 kMAX batch。

## 5. CUDA 接口使用说明

### 5.1 设备选择

在 `Engine::init` 中使用：

```cpp
cudaGetDeviceCount(&gpu_cnt);
cudaSetDevice(device_id);
```

如果 device id 非法，返回 `GPU_DEVICE_ID_ERR`。

### 5.2 Stream 设计

当前有两类 stream：

- `impl_->stream`：SDK 主推理 stream，使用 `cudaStreamCreate` 创建。
- `handle->upload_stream`：每个 handle 独立的 H2D 上传 stream，使用 `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)` 创建。

这样可以让预处理上传和上一批推理并行，前提是调用侧使用不同 handle 进行流水。

### 5.3 Event 同步

每个 handle 有一个 `input_ready` event：

```cpp
cudaEventCreateWithFlags(&handle->input_ready, cudaEventDisableTiming);
cudaMemcpyAsync(..., handle->upload_stream);
cudaEventRecord(handle->input_ready, handle->upload_stream);
```

推理前主 stream 等待该事件：

```cpp
cudaStreamWaitEvent(impl_->stream, handle->input_ready, 0);
```

这保证 GPU 输入拷贝完成后才开始 TensorRT 推理，同时不强制 CPU 阻塞。

### 5.4 内存分配

输入：

- Host 输入：`cudaMallocHost`，分配 pinned memory，加速 H2D 异步拷贝。
- Device 输入：`cudaMalloc`。

输出：

- Device 输出：`cudaMalloc`。
- Host 输出：`malloc`。当前 D2H 使用 `cudaMemcpyAsync`，但 host 输出不是 pinned memory，因此异步效率可能不如 pinned memory。若进一步优化，可以把 `output_host` 也改为 `cudaMallocHost`。

释放：

- `cudaFree(handle->image_data_d)`
- `cudaFreeHost(handle->input_host)`
- `cudaEventDestroy(handle->input_ready)`
- `cudaStreamDestroy(handle->upload_stream)`
- `cudaFree(output_device[i])`
- `free(output_host[i])`
- `cudaStreamDestroy(impl_->stream)`

### 5.5 拷贝路径

预处理上传：

```cpp
cudaMemcpyAsync(handle->image_data_d,
                host_input,
                image_bytes,
                cudaMemcpyHostToDevice,
                handle->upload_stream);
```

输出下载：

```cpp
cudaMemcpyAsync(output_raw,
                output_d,
                bytes,
                cudaMemcpyDeviceToHost,
                impl_->stream);
cudaStreamSynchronize(impl_->stream);
```

## 6. 图像预处理实现

预处理位于 `image_uniformization`。当前流程：

1. 对每张输入图做 letterbox。
2. 将 BGR 图像转换为 RGB。
3. 将 `uint8` 转为 `float32`。
4. 归一化到 `[0, 1]`。
5. 按模型 layout 写入 NCHW 或 NHWC 输入 buffer。
6. 异步上传到 GPU。

### 6.1 Letterbox

letterbox 逻辑：

- `r = min(input_h / src_h, input_w / src_w)`。
- 按比例 resize 到 `new_unpad_w/new_unpad_h`。
- 计算左右/上下 padding。
- 输出固定大小 `input_w x input_h`。
- padding 颜色为 `(114, 114, 114)`。

SDK 保存 `scale/pad_left/pad_top`，供后处理坐标还原。

### 6.2 SIMD 预处理

当前已使用 OpenCV universal intrinsics：

- `bgr_to_rgb_float_nchw_simd`：输出 NCHW。
- `bgr_to_rgb_float_nhwc_simd`：输出 NHWC。
- `image_preprocess_simd`：根据模型 layout 调用对应实现。

SIMD 做的事情是把以下步骤融合为一次扫描：

```text
BGR uint8 -> RGB float32 -> /255 -> 写入目标 layout
```

相比 `cv::cvtColor + convertTo + blobFromImage/memcpy`，减少了中间 Mat 和多次内存遍历。

当前 OpenCV 构建信息显示已支持：

- Baseline: SSE/SSE2/SSE3/SSSE3/SSE4.1/SSE4.2
- Dispatch: AVX/AVX2
- Intel IPP: enabled

工程 CMake 当前 Release 参数包含：

```cmake
-O3 -Wall -msse4.2 -mavx2 -mfma
```

因此自写 SIMD 预处理和 OpenCV 内部 resize 都具备 SIMD 优化条件。

## 7. 推理流程

`forward` 的 GPU 部分：

1. 如果当前 handle 记录了 `input_ready`，主推理 stream 等待上传完成。
2. 组装 TensorRT I/O buffer：

```cpp
std::vector<void *> buffers = {handle->image_data_d, output_d};
```

3. 调用 `impl_->engine.infer(buffers, handle->batch, impl_->stream)`。
4. 将输出从 GPU 拷贝回 CPU。
5. `cudaStreamSynchronize(impl_->stream)` 等待推理和 D2H 完成。

注意：当前 `forward` 计时如果在外部测量，会包含推理、D2H、同步和 CPU 后处理，不是纯 TensorRT kernel 时间。

## 8. 后处理实现

当前假设输出 shape 末两维为：

```text
[..., channel_num, box_num]
```

其中：

- 前 4 个 channel 为 box：`cx, cy, w, h`。
- 后续 channel 为类别 score。
- `class_num = channel_num - 4`。

处理步骤：

1. 遍历每个 candidate box。
2. 找最大类别分数。
3. 小于 `confidence_thresh` 的过滤掉。
4. 将 `cx,cy,w,h` 转为 `x1,y1,x2,y2`。
5. 调用 `apply_nms` 做按类别 NMS。
6. 根据 letterbox 的 `scale/pad_left/pad_top` 映射回原图。
7. clamp 到原图尺寸。
8. 输出 `DetectionResult`。

输出数据类型支持：

- FP32：直接读取 `float`。
- FP16：通过 `fp16_to_float` 手动转换。

## 9. 错误码

错误码定义在 `include/algorithm_sdk_types.h`：

| 错误码 | 含义 |
| --- | --- |
| `SUCCESS` | 成功 |
| `INIT_ERR` | 初始化错误，如用户 max batch 超过模型最大 batch |
| `INIT_MODEL_ERR` | 模型/SDK 未初始化 |
| `MODEL_PATH_ERR` | 模型路径不存在或无法打开 |
| `MODEL_ERR` | 模型结构、shape、dtype 等不符合预期 |
| `POSTPROCESS_ERR` | 后处理异常 |
| `PARAM_ERR` | 参数错误 |
| `CUDA_RUNTIME_ERR` | CUDA runtime 调用失败 |
| `CUDA_MEMORY_ERR` | CUDA/CPU 内存分配或拷贝失败 |
| `MODEL_INFERENCE_ERR` | TensorRT 推理失败 |
| `BATCHSIZE_ERR` | batch 超出限制 |
| `GPU_DEVICE_ID_ERR` | GPU device id 非法 |
| `INPUT_FORMAT_ERR` | 输入图像为空、通道数不对或 batch 为空 |
| `IMAGE_PREPROCESS_ERR` | 图像预处理异常 |

## 10. 资源生命周期

推荐使用顺序：

```cpp
Algorithm_Sdk::DetectionSDK sdk;
sdk.init(model_path, max_batch, device_id);

Algorithm_Sdk::ImageHandle handle = nullptr;
sdk.image_handle_create(&handle);

sdk.image_uniformization(images, handle);
sdk.forward(handle, &results, conf_thresh, nms_thresh);

sdk.image_handle_free(&handle);
```

注意事项：

- `ImageHandle` 创建后可以复用，多次调用 `image_uniformization + forward`。
- 如果 batch 所需内存变大，内部会重新分配输入 buffer。
- 输出 buffer 当前只在首次 forward 前按当前 batch 分配；如果同一个 handle 后续使用更大的 batch，建议检查并扩展输出 buffer 分配逻辑。
- `DetectionSDK` 禁止拷贝，避免底层 CUDA/TensorRT 资源重复释放。

## 11. 性能相关点

当前已有优化：

- TensorRT engine 反序列化后复用 execution context。
- 输入 host buffer 使用 pinned memory。
- H2D 使用独立 non-blocking stream。
- 使用 event 在上传 stream 和推理 stream 之间同步。
- 支持双 handle 做预处理和推理 overlap。
- 预处理 BGR/RGB/normalize/layout 已融合 SIMD。
- OpenCV 构建启用 AVX/AVX2 和 IPP。

仍可继续优化的点：

- init warmup 当前只有 1 次，可改为 5 到 10 次以稳定前几帧。
- 常用动态 batch 都可以分别 warmup。
- 输出 host buffer 可改为 `cudaMallocHost`，提高 D2H 异步拷贝效率。
- 如果只关心纯推理时间，建议用 CUDA event 包住 `enqueueV3`，不要把 D2H 和后处理混在一起。
- 后处理 NMS 当前为 CPU 实现，大量候选框时会影响 `forward` wall time。

## 12. 构建与依赖

CMake 主要依赖：

- C++17
- CUDA Runtime
- TensorRT 10
- OpenCV 5
- pthread

SDK 编译为共享库：

```cmake
add_library(yolo11_sdk SHARED ${YOLO11_SOURCES})
```

链接库：

```cmake
target_link_libraries(yolo11_sdk
    PRIVATE ${CUDA_LIBRARIES}
            nvinfer
            nvinfer_plugin
            cuda
            ${OpenCV_LIBS}
)
```

测试可执行文件：

```cmake
add_executable(client main.cpp test/*.cpp)
```

## 13. 当前实现边界

- 只支持 3 通道输入。
- 只支持 1 个模型输入和 1 个模型输出的典型 YOLO 检测结构。
- 输出解析假设为 YOLO 风格 `[4 + class_num, box_num]`。
- `Engine::infer` 使用同一个 execution context，并用 mutex 串行保护，因此同一个 SDK 实例内不会并行执行多个 TensorRT enqueue。
- 多 handle 可以 overlap 预处理上传和推理，但不能让同一个 execution context 同时推理多批。
