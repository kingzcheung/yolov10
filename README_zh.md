# YOLOv10 Rust 版目标检测

本项目使用 Rust 编程语言实现了 YOLOv10 目标检测模型推理。支持两种后端框架：ONNX Runtime 和 Candle，可以通过 Cargo features 进行选择。

## 功能特性

- 双后端支持：ONNX Runtime 和 Candle
- 多种硬件加速支持（CUDA、CoreML、DirectML、TensorRT）
- 易于作为库箱集成到其他项目中
- 包含图像预处理和后处理功能
- 提供结果可视化工具
- 支持所有 YOLOv10 模型变体（n、s、m、b、l、x）

## 后端框架

### ONNX Runtime（默认）
- 基于微软的 ONNX Runtime
- 支持多种执行提供程序：CPU、CUDA、CoreML、DirectML、TensorRT
- 使用 ONNX 格式的模型文件

### Candle
- 基于 Candle ML 框架（纯 Rust 实现）
- 轻量级，无外部 C++ 依赖
- 使用 safetensors 格式的模型文件

## 安装

将以下内容添加到你的 `Cargo.toml`：

```toml
[dependencies]
yolov10 = { version = "*" } # * 表示最新版本
```

## Features

该库提供了多个 features 用于自定义后端和硬件加速：

| Feature        | 描述                         | 后端       |
|----------------|-------------------------------|------------|
| `onnx`         | 使用 ONNX Runtime 作为后端    | ONNX       |
| `candle`       | 使用 Candle 作为后端（默认）  | Candle     |
| `onnx-cuda`    | 为 ONNX 启用 CUDA 支持        | ONNX       |
| `onnx-coreml`  | 为 ONNX 启用 CoreML 支持      | ONNX       |
| `onnx-directml`| 为 ONNX 启用 DirectML 支持    | ONNX       |
| `onnx-tensorrt`| 为 ONNX 启用 TensorRT 支持    | ONNX       |
| `candle-cuda`  | 为 Candle 启用 CUDA 支持      | Candle     |

默认启用 `candle` feature。要使用 ONNX Runtime 作为后端：

```toml
[dependencies]
yolov10 = { version = "*", default-features = false, features = ["onnx"] }
```

## 使用方法

### 使用 ONNX Runtime 后端

```rust
use yolov10::{InferenceEngine, draw_labels};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 使用 ONNX 模型创建推理引擎
    let mut engine = InferenceEngine::new("yolov10s.onnx")?;
    
    // 加载图像
    let input_data = std::fs::read("input.jpg")?;
    
    // 运行推理
    let results = engine.run_inference(&input_data, 0.3)?;
    
    // 可视化结果
    let image = image::load_from_memory(&input_data)?;
    let img = draw_labels(&image, &results);
    img.save("result.jpg")?;
    
    Ok(())
}
```

使用 ONNX 后端运行：
```bash
cargo run --features onnx --example basic
```

### 使用 Candle 后端

```rust
use candle_nn::VarBuilder;
use yolov10::{
    candle::{YoloV10, Multiples, Device, Tensor, DType},
    draw_labels, filter_detections
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 加载模型权重
    let device = Device::cuda_if_available(0)?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &["yolov10s.safetensors"],
            DType::F32,
            &device,
        )
    }?;
    
    // 加载并预处理图像
    let original_image = image::load_from_memory(include_bytes!("../testdata/bus.jpg"))?;
    let image_t = {
        let img = original_image.resize_exact(
            640u32,
            640u32,
            image::imageops::FilterType::CatmullRom,
        );
        let data = img.to_rgb8().into_raw();
        Tensor::from_vec(
            data,
            (img.height() as usize, img.width() as usize, 3),
            &device,
        )?
        .permute((2, 0, 1))?
    };
    let image_t = (image_t.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;
    
    // 创建模型并运行推理
    let yolo = YoloV10::load(vb, Multiples::s(), 80)?;
    let output = yolo.forward(&image_t)?;
    
    // 后处理结果
    let output_vec: Vec<f32> = output.flatten_all()?.to_vec1()?;
    let results = filter_detections(
        &output_vec,
        0.3,
        640,
        640,
        original_image.width() as u32,
        original_image.height() as u32,
    );
    
    // 可视化结果
    let img = draw_labels(&original_image, &results);
    img.save_with_format("result.jpg", image::ImageFormat::Jpeg)?;
    
    Ok(())
}
```

使用 Candle 后端运行：
```bash
cargo run --features candle --example candle
```

## 示例

项目提供了多个示例：

1. `basic` - 使用 ONNX 后端的基本目标检测
2. `basic_with_labels` - 使用自定义标签和 ONNX 后端的基本目标检测
3. `candle` - 使用 Candle 后端的目标检测

运行示例：
```bash
# ONNX 示例
cargo run --features onnx --example basic
cargo run --features onnx --example basic_with_labels

# Candle 示例
cargo run --features candle --example candle
```

## 测试

项目为不同模型变体提供了全面的测试：

```bash
# 使用 Candle 后端测试（默认）
cargo test --features candle

# 使用 ONNX 后端测试
cargo test --features onnx
```

特定模型测试：
```bash
# 测试 YOLOv10n 模型
cargo test --features candle test_yolov10n_inference

# 测试 YOLOv10s 模型
cargo test --features candle test_yolov10s_inference

# 测试 YOLOv10m 模型
cargo test --features candle test_yolov10m_inference

# 其他模型变体类似（l、x、b）
```

## 模型准备

### ONNX Runtime 后端

需要 ONNX 格式的模型文件。可以使用提供的脚本转换 PyTorch 模型：

```bash
python export_onnx.py
```

### Candle 后端

需要 safetensors 格式的模型文件。使用以下命令转换 PyTorch 模型：

```bash
python export_to_safetensors.py --model_path yolov10s.pt
```

## 环境要求

- Rust 1.80+ (2024 edition)
- ONNX Runtime 或 CUDA/cuDNN 用于硬件加速（取决于选择的后端）
- 带有 ultralytics 包的 Python (用于模型导出)

## 许可证

本项目的代码采用 Apache License 2.0 许可证 - 详情请见 [LICENSE](LICENSE) 文件。
YOLOv10 模型权重采用 AGPL-3.0 许可证。

## 重要声明

YOLOv10 模型权重采用 AGPL-3.0 许可证，这是一种著佐权许可证。如果你在项目中使用这些权重，你的项目可能需要在同一或兼容的许可证下发布。在使用模型权重之前，请仔细阅读 AGPL-3.0 许可证条款。