#![cfg(feature = "candle")]

use std::error::Error;
use std::time::Instant;

use candle_nn::VarBuilder;
use yolov10::{
    candle::{DType, Multiples, Tensor, YoloV10},
    draw_labels, filter_detections,
};
fn main() -> Result<(), Box<dyn Error>> {
    let input_data = include_bytes!("../testdata/zidane.jpg");
    let device = yolov10::candle::Device::cuda_if_available(0)?;

    println!("device: {device:?}");

    // 记录模型加载开始时间
    let load_start = Instant::now();
    
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&["yolov10s.safetensors"], DType::F32, &device)
    }?;

    let original_image = image::load_from_memory(input_data)?;

    let image_t = {
        let img =
            original_image.resize_exact(640u32, 640u32, image::imageops::FilterType::CatmullRom);
        let data = img.to_rgb8().into_raw();
        Tensor::from_vec(
            data,
            (img.height() as usize, img.width() as usize, 3),
            &device,
        )?
        .permute((2, 0, 1))?
    };
    let image_t = (image_t.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;
    println!("Input image shape: {:?}", image_t.shape());

    let yolo = YoloV10::load(vb, Multiples::s(), 80)?;
    
    // 计算模型加载耗时
    let load_duration = load_start.elapsed();
    println!("Model loading time: {load_duration:.2?}");

    // 记录推理开始时间
    let infer_start = Instant::now();
    
    // let xs = vec![1f32; 640 * 640 * 3];
    // let image_t = candle_core::Tensor::from_vec(xs, (1, 3, 640, 640), &device)?;
    let output = yolo.forward(&image_t)?;
    println!("Output shape: {:?}", output.shape());

    // 计算推理耗时
    let infer_duration = infer_start.elapsed();
    println!("Inference time: {infer_duration:.2?}");

    // 记录后处理开始时间
    let post_start = Instant::now();
    
    // 将 output 展平为一维张量，并转换为 Vec<f32>
    let output_vec: Vec<f32> = output.flatten_all()?.to_vec1()?;

    let results = filter_detections(
        &output_vec,
        0.3,
        640,
        640,
        original_image.width() as u32,
        original_image.height() as u32,
    );
    
    // 计算后处理耗时
    let post_duration = post_start.elapsed();
    println!("Post-processing time: {:.2?}", post_duration);
    
    // 计算总耗时
    let total_duration = load_start.elapsed();
    println!("Total execution time: {:.2?}", total_duration);
    println!("Detected {} objects", results.len());

    let img = draw_labels(&original_image, &results);

    img.save_with_format("res.jpg", image::ImageFormat::Jpeg)
        .unwrap();

    Ok(())
}