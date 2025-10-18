#![cfg(feature = "onnx")]

use std::error::Error;

use yolov10::{draw_labels, onnx::InferenceEngine};

fn main() -> Result<(), Box<dyn Error>> {
    // 加载模型
    let mut engine = InferenceEngine::new("yolov10s.onnx")?;

    // 加载图像
    let input_data = include_bytes!("../testdata/bus.jpg");

    // 运行推理
    let results = engine.run_inference(input_data, 0.3)?;

    println!("{:?}", &results);

    let image = image::load_from_memory(input_data).unwrap();

    let img = draw_labels(&image, &results);

    img.save_with_format("res.jpg", image::ImageFormat::Jpeg)
        .unwrap();

    Ok(())
}
