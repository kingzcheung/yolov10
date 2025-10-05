# YOLOv10 Rust 版目标检测

本项目使用 Rust 编程语言和 ONNX Runtime 实现了 YOLOv10 目标检测模型推理。

## 使用方法

作为一个库箱（library crate），你可以通过添加依赖项将此功能集成到你自己的 Rust 项目中：

```toml
[dependencies]
yolov10 = {version = "*" } # * means latest
```

然后在你的代码中使用：

```rust
use yolov10::{InferenceEngine, draw_labels};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = InferenceEngine::new("yolov10s.onnx")?;
    let input_data = std::fs::read("input.jpg")?;
    let results = engine.run_inference(&input_data, 0.3)?;
    
    let image = image::load_from_memory(&input_data)?;
    let img = draw_labels(&image, &results);
    img.save("result.jpg")?;
    
    Ok(())
}
```

## 重要声明

YOLOv10 模型权重采用 AGPL-3.0 许可证，这是一种著佐权许可证。如果你在项目中使用这些权重，你的项目可能需要在同一或兼容的许可证下发布。在使用模型权重之前，请仔细阅读 AGPL-3.0 许可证条款。

本项目的代码采用 Apache License 2.0 许可证 - 详情请见 [LICENSE](LICENSE) 文件。
但是，如果你使用 YOLOv10 模型权重，则组合作品可能受 AGPL-3.0 许可证条款的约束。

## 环境要求

- Rust 1.80+ (2024 edition)
- ONNX Runtime
- 带有 ultralytics 包的 Python (用于模型导出)

## 许可证

本项目的代码采用 Apache License 2.0 许可证 - 详情请见 [LICENSE](LICENSE) 文件。
YOLOv10 模型权重采用 AGPL-3.0 许可证。