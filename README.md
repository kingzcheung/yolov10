# YOLOv10 Object Detection in Rust

This project implements YOLOv10 object detection model inference using Rust programming language with ONNX Runtime.

## Usage

As a library crate, you can integrate this into your own Rust projects by adding it as a dependency:

```toml
[dependencies]
yolov10 = { path = "path/to/this/repository" }
```

Then use it in your code:

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

## Important Notice

The YOLOv10 model weights are licensed under AGPL-3.0, which is a copyleft license. If you use these weights in your project, your project may need to be released under the same or a compatible license. Please review the AGPL-3.0 license terms carefully before using the model weights.

This project's code is licensed under the Apache License 2.0. However, if you use the YOLOv10 model weights, the combined work may be subject to the AGPL-3.0 license terms.

## Requirements

- Rust 1.80+ (2024 edition)
- ONNX Runtime
- Python with ultralytics package (for model export)

## License

This project's code is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
The YOLOv10 model weights are under AGPL-3.0 license.