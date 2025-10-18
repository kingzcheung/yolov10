#![cfg(feature = "candle")]

use std::error::Error;
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use yolov10::{
    draw_labels, filter_detections,
    candle::{Multiples, YoloV10},
};

#[test]
fn test_yolov10l_inference() -> Result<(), Box<dyn Error>> {
    // 使用 CPU 设备进行测试
    let device = Device::cuda_if_available(0)?;

    // 加载 yolov10l 模型权重
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &["yolov10l.safetensors"],
            DType::F32,
            &device,
        )
    }?;

    // 加载测试图像
    let input_data = include_bytes!("../testdata/bus.jpg");
    let original_image = image::load_from_memory(input_data)?;

    // 预处理图像
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

    // 创建 YOLOv10l 模型实例
    let yolo = YoloV10::load(vb, Multiples::l(), 80)?;

    // 执行推理
    let output = yolo.forward(&image_t)?;
    
    // 验证输出形状是否正确
    // YOLOv10l 输出形状为 [1, 300, 6]，与 s/m/x 模型相同
    assert_eq!(output.dims(), &[1, 300, 6]);

    // 将输出展平并转换为 Vec<f32>
    let output_vec: Vec<f32> = output.flatten_all()?.to_vec1()?;

    // 过滤检测结果
    let results = filter_detections(
        &output_vec,
        0.3,  // 置信度阈值
        640,  // 输入图像宽度
        640,  // 输入图像高度
        original_image.width() as u32,
        original_image.height() as u32,
    );

    // 验证检测结果不为空
    assert!(!results.is_empty());

    // 验证检测结果格式正确
    for detection in &results {
        let (_x, _y, width, height) = detection.bbox;
        
        // 验证置信度在合理范围内
        assert!(detection.confidence >= 0.3 && detection.confidence <= 1.0);
        // 验证类别 ID 在有效范围内
        assert!(detection.class_id < 80);
        // 验证边界框尺寸为正数
        assert!(width > 0 && height > 0);
    }

    // 保存带标注的结果图像
    let img = draw_labels(&original_image, &results);
    img.save_with_format("./target/yolov10l_test_result.jpg", image::ImageFormat::Jpeg)
        .unwrap();

    println!("YOLOv10l inference test passed!");
    println!("Detected {} objects", results.len());

    Ok(())
}