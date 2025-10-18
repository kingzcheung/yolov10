# YOLOv10 Object Detection in Rust

This project implements YOLOv10 object detection model inference using Rust programming language. It supports two backend frameworks: ONNX Runtime and Candle, which can be selected through Cargo features.

## Features

- Two backend options: ONNX Runtime and Candle
- Multiple hardware acceleration support (CUDA, CoreML, DirectML, TensorRT)
- Easy to integrate as a library crate
- Preprocessing and postprocessing included
- Visualization utilities for drawing detection results
- Support for all YOLOv10 model variants (n, s, m, b, l, x)

## Backend Frameworks

### ONNX Runtime (default)
- Based on Microsoft's ONNX Runtime
- Supports multiple execution providers: CPU, CUDA, CoreML, DirectML, TensorRT
- Works with ONNX model files

### Candle
- Based on the Candle ML framework (Pure Rust)
- Lightweight and no external C++ dependencies
- Works with safetensors model files

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
yolov10 = { version = "*" } # * means latest
```

## Features

The crate provides several features for customizing the backend and hardware acceleration:

| Feature        | Description                        | Backend    |
|----------------|------------------------------------|------------|
| `onnx`         | Use ONNX Runtime as backend        | ONNX       |
| `candle`       | Use Candle as backend (default)    | Candle     |
| `onnx-cuda`    | Enable CUDA support for ONNX       | ONNX       |
| `onnx-coreml`  | Enable CoreML support for ONNX     | ONNX       |
| `onnx-directml`| Enable DirectML support for ONNX   | ONNX       |
| `onnx-tensorrt`| Enable TensorRT support for ONNX   | ONNX       |
| `candle-cuda`  | Enable CUDA support for Candle     | Candle     |

By default, the `candle` feature is enabled. To use ONNX Runtime instead:

```toml
[dependencies]
yolov10 = { version = "*", default-features = false, features = ["onnx"] }
```

## Usage

### Using ONNX Runtime Backend

```rust
use yolov10::{InferenceEngine, draw_labels};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create inference engine with ONNX model
    let mut engine = InferenceEngine::new("yolov10s.onnx")?;
    
    // Load image
    let input_data = std::fs::read("input.jpg")?;
    
    // Run inference
    let results = engine.run_inference(&input_data, 0.3)?;
    
    // Visualize results
    let image = image::load_from_memory(&input_data)?;
    let img = draw_labels(&image, &results);
    img.save("result.jpg")?;
    
    Ok(())
}
```

To run with ONNX backend:
```bash
cargo run --features onnx --example basic
```

### Using Candle Backend

```rust
use candle_nn::VarBuilder;
use yolov10::{
    candle::{YoloV10, Multiples, Device, Tensor, DType},
    draw_labels, filter_detections
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model weights
    let device = Device::cuda_if_available(0)?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &["yolov10s.safetensors"],
            DType::F32,
            &device,
        )
    }?;
    
    // Load and preprocess image
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
    
    // Create model and run inference
    let yolo = YoloV10::load(vb, Multiples::s(), 80)?;
    let output = yolo.forward(&image_t)?;
    
    // Postprocess results
    let output_vec: Vec<f32> = output.flatten_all()?.to_vec1()?;
    let results = filter_detections(
        &output_vec,
        0.3,
        640,
        640,
        original_image.width() as u32,
        original_image.height() as u32,
    );
    
    // Visualize results
    let img = draw_labels(&original_image, &results);
    img.save_with_format("result.jpg", image::ImageFormat::Jpeg)?;
    
    Ok(())
}
```

To run with Candle backend:
```bash
cargo run --features candle --example candle
```

## Examples

The project provides several examples:

1. `basic` - Basic object detection with ONNX backend
2. `basic_with_labels` - Basic object detection with custom labels using ONNX backend
3. `candle` - Object detection using Candle backend

Run examples with:
```bash
# ONNX examples
cargo run --features onnx --example basic
cargo run --features onnx --example basic_with_labels

# Candle example
cargo run --features candle --example candle
```

## Testing

The project includes comprehensive tests for different model variants:

```bash
# Test with Candle backend (default)
cargo test --features candle

# Test with ONNX backend
cargo test --features onnx
```

Specific model tests:
```bash
# Test YOLOv10n model
cargo test --features candle test_yolov10n_inference

# Test YOLOv10s model
cargo test --features candle test_yolov10s_inference

# Test YOLOv10m model
cargo test --features candle test_yolov10m_inference

# And so on for other model variants (l, x, b)
```

## Model Preparation

### For ONNX Runtime Backend

You need ONNX model files. You can convert PyTorch models using the provided script:

```bash
python export_onnx.py
```

### For Candle Backend

You need safetensors model files. Convert PyTorch models using:

```bash
python export_to_safetensors.py --model_path yolov10s.pt
```

## Requirements

- Rust 1.80+ (2024 edition)
- ONNX Runtime or CUDA/cuDNN for hardware acceleration (depending on chosen backend)
- Python with ultralytics package (for model export)

## License

This project's code is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
The YOLOv10 model weights are under AGPL-3.0 license.

## Important Notice

The YOLOv10 model weights are licensed under AGPL-3.0, which is a copyleft license. If you use these weights in your project, your project may need to be released under the same or a compatible license. Please review the AGPL-3.0 license terms carefully before using the model weights.