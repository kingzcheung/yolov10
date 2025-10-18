#![cfg(feature = "onnx")]

use std::error::Error;
use std::time::Instant;

use yolov10::{draw_labels, onnx::InferenceEngine};

fn main() -> Result<(), Box<dyn Error>> {
    // 记录模型加载开始时间
    let load_start = Instant::now();
    
    // 加载模型
    let mut engine = InferenceEngine::new("yolov10s.onnx")?;
    
    // 计算模型加载耗时
    let load_duration = load_start.elapsed();
    println!("Model loading time: {:.2?}", load_duration);

    // 加载图像
    let input_data = include_bytes!("../testdata/bus.jpg");

    // 记录推理开始时间
    let infer_start = Instant::now();
    
    // 运行推理
    let results = engine.run_inference(input_data, 0.3)?;
    
    // 计算推理耗时
    let infer_duration = infer_start.elapsed();
    println!("Inference time: {:.2?}", infer_duration);
    
    println!("Detected {} objects", results.len());
    println!("{:?}", &results);

    let image = image::load_from_memory(input_data).unwrap();

    let img = draw_labels(&image, &results);

    img.save_with_format("res.jpg", image::ImageFormat::Jpeg)
        .unwrap();

    Ok(())
}