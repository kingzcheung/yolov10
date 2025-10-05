use ab_glyph::{Font, FontRef, PxScale};
use image::{DynamicImage, GenericImageView, Rgb, imageops::FilterType};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
pub use ndarray;
use ndarray::Array;
pub use ort;

use ort::{
    inputs,
    session::{Session, SessionOutputs, builder::GraphOptimizationLevel},
    value::TensorRef,
};

#[cfg(feature = "coreml")]
use ort::execution_providers::CoreMLExecutionProvider;

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

#[cfg(feature = "directml")]
use ort::execution_providers::DirectMLExecutionProvider;

#[cfg(feature = "tensorrt")]
use ort::execution_providers::TensorRTExecutionProvider;

use std::error::Error;
use std::borrow::Cow;

#[rustfmt::skip]
pub const YOLOV10_CLASS_LABELS: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
	"tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

const COLOR: [image::Rgb<u8>; 20] = [
    Rgb([255, 0, 0]),     // 红色
    Rgb([0, 255, 0]),     // 绿色
    Rgb([0, 0, 255]),     // 蓝色
    Rgb([255, 255, 0]),   // 黄色
    Rgb([255, 0, 255]),   // 紫色
    Rgb([0, 255, 255]),   // 青色
    Rgb([255, 128, 0]),   // 橙色
    Rgb([255, 0, 128]),   // 粉色
    Rgb([128, 255, 0]),   // 黄绿色
    Rgb([0, 128, 255]),   // 天蓝色
    Rgb([128, 0, 255]),   // 紫罗兰
    Rgb([255, 128, 128]), // 浅红色
    Rgb([128, 255, 128]), // 浅绿色
    Rgb([128, 128, 255]), // 浅蓝色
    Rgb([255, 255, 128]), // 浅黄色
    Rgb([255, 128, 255]), // 浅紫色
    Rgb([128, 255, 255]), // 浅青色
    Rgb([192, 192, 192]), // 银色
    Rgb([128, 128, 128]), // 灰色
    Rgb([0, 0, 0]),       // 黑色
];

/// 检测框结构体
#[derive(Debug, Clone)]
pub struct Detection {
    pub confidence: f32,
    pub bbox: (u32, u32, u32, u32), // x, y, width, height
    pub class_id: usize,
    pub class_name: String,
}

/// YOLOv10推理引擎
pub struct InferenceEngine<'a> {
    session: Session,
    class_labels: Option<Vec<Cow<'a, str>>>,
}

impl<'a> InferenceEngine<'a> {
    /// 创建新的推理引擎实例，使用默认的CPU执行提供程序
    pub fn new(model_path: &str) -> Result<Self, Box<dyn Error>> {
        let session = Session::builder()?
            .with_inter_threads(1)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([
                #[cfg(feature = "cuda")]
                CUDAExecutionProvider::default().build(),
                #[cfg(feature = "directml")]
                DirectMLExecutionProvider::default().build(),
                #[cfg(feature = "coreml")]
                CoreMLExecutionProvider::default().build(),
                #[cfg(feature = "tensorrt")]
                TensorRTExecutionProvider::default().build(),
            ])?
            .commit_from_file(model_path)?;

        Ok(Self { 
            session, 
            class_labels: None
        })
    }

    pub fn new_with_labels(model_path: &str, class_labels: &'a[&str]) -> Result<Self, Box<dyn Error>> {
        let session = Session::builder()?
            .with_inter_threads(1)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([
                #[cfg(feature = "cuda")]
                CUDAExecutionProvider::default().build(),
                #[cfg(feature = "directml")]
                DirectMLExecutionProvider::default().build(),
                #[cfg(feature = "coreml")]
                CoreMLExecutionProvider::default().build(),
                #[cfg(feature = "tensorrt")]
                TensorRTExecutionProvider::default().build(),
            ])?
            .commit_from_file(model_path)?;
        
        Ok(Self { 
            session, 
            class_labels: Some(class_labels.iter().map(|&s| Cow::Borrowed(s)).collect()) 
        })
    }


    /// 预处理图像
    #[allow(clippy::type_complexity)]
    fn preprocess_image(
        &self,
        image: &DynamicImage,
    ) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>>, Box<dyn Error>>
    {
        let img = image.resize_exact(640, 640, FilterType::Nearest);
        let mut input = Array::zeros((1, 3, 640, 640));
        for pixel in img.pixels() {
            let x = pixel.0 as _;
            let y = pixel.1 as _;
            let [r, g, b, _] = pixel.2.0;
            input[[0, 0, y, x]] = (r as f32) / 255.;
            input[[0, 1, y, x]] = (g as f32) / 255.;
            input[[0, 2, y, x]] = (b as f32) / 255.;
        }

        Ok(input)
    }

    /// 运行推理
    pub fn run_inference(&mut self, buffer: &[u8], confidence_threshold: f32) -> Result<Vec<Detection>, Box<dyn Error>> {
        // 创建输入张量

        let img = image::load_from_memory(buffer)?;
        let orig_width = img.width();
        let orig_height = img.height();

        let array = self.preprocess_image(&img)?;
        // 运行推理
        let outputs: SessionOutputs = self
            .session
            .run(inputs!["images" => TensorRef::from_array_view(&array)?])?;

        // 获取输出数据

        let (_output_shape, output_data) = outputs["output0"].try_extract_tensor::<f32>()?;
        // println!("输出张量形状: {:?}", _output_shape);
        let output_vec: Vec<f32> = output_data.to_vec();
        // println!("{:?}", output_vec);
        let detections = filter_detections(
            &output_vec, 
            confidence_threshold, 
            640, 
            640, 
            orig_width, 
            orig_height
        );

        Ok(detections)
    }
}

/// 过滤检测结果
pub fn filter_detections(
    results: &[f32],
    confidence_threshold: f32,
    img_width: u32,
    img_height: u32,
    orig_width: u32,
    orig_height: u32,
) -> Vec<Detection> {
    filter_detections_with_labels(results, confidence_threshold, img_width, img_height, orig_width, orig_height, None)
}

/// 过滤检测结果
pub fn filter_detections_with_labels(
    results: &[f32],
    confidence_threshold: f32,
    img_width: u32,
    img_height: u32,
    orig_width: u32,
    orig_height: u32,
    class_labels: Option<&[&str]>,
) -> Vec<Detection> {
    // YOLOv10输出格式: [x1, y1, x2, y2, score, class_id]
    // 每6个元素为一个检测框
    if !results.len().is_multiple_of(6) {
        eprintln!("警告: 模型输出长度不是6的倍数，实际长度: {}", results.len());
    }

    let num_detections = results.len() / 6;
    // println!("检测框数量: {}", num_detections);

    let mut detections = Vec::with_capacity(num_detections);

    // 获取类别标签引用或使用默认标签
    let labels: &[&str] = class_labels.unwrap_or(YOLOV10_CLASS_LABELS.as_slice());

    // 计算缩放和填充因子 
    let scale = (img_width as f32 / orig_width as f32).min(img_height as f32 / orig_height as f32);
    let new_width = (orig_width as f32 * scale) as u32;
    let new_height = (orig_height as f32 * scale) as u32;
    let pad_x = (img_width - new_width) / 2;
    let pad_y = (img_height - new_height) / 2;

    for i in 0..num_detections {
        let base_index = i * 6;

        let left = results[base_index];
        let top = results[base_index + 1];
        let right = results[base_index + 2];
        let bottom = results[base_index + 3];
        let confidence = results[base_index + 4];
        let class_id = results[base_index + 5] as usize;

        // 打印原始值用于调试
        // println!("检测框 {}: left={}, top={}, right={}, bottom={}, 置信度={}, 类别ID={}",
        //          i, left, top, right, bottom, confidence, class_id);

        // 检查置信度是否有效
        if !(0.0..=1.0).contains(&confidence) {
            // println!("跳过无效置信度: {}", confidence);
            continue;
        }

        // 检查类别ID是否有效
        if class_id >= labels.len() {
            // println!("跳过无效类别ID: {}", class_id);
            continue;
        }

        // 应用置信度阈值
        if confidence >= confidence_threshold {
            // 移除填充并缩放到原始图像尺寸
            let left = (left - pad_x as f32) / scale;
            let top = (top - pad_y as f32) / scale;
            let right = (right - pad_x as f32) / scale;
            let bottom = (bottom - pad_y as f32) / scale;

            let x = left as u32;
            let y = top as u32;
            let width = (right - left) as u32;
            let height = (bottom - top) as u32;

            // 确保坐标有效
            if width > 0 && height > 0 && x < orig_width && y < orig_height {
                detections.push(Detection {
                    confidence,
                    bbox: (x, y, width, height),
                    class_id,
                    class_name: labels[class_id].to_string(),
                });
            } else {
                println!(
                    "跳过无效边界框: ({x}, {y}) - 宽度: {width}, 高度: {height}"
                );
            }
        }
    }

    // NMS，但是 yolov10 并不需要 nms
    // nms(&mut detections, 0.5, 0.3);

    // println!("最终有效检测数量: {}", detections.len());
    detections
}

/// 在图像上绘制检测框和标签
pub fn draw_labels(image: &DynamicImage, detections: &[Detection]) -> DynamicImage {
    let mut image = image.to_rgb8();

    // 加载默认字体
    let font = FontRef::try_from_slice(include_bytes!("../testdata/OpenSans-Regular.ttf")).unwrap();
    for detection in detections {
        let (x, y, width, height) = detection.bbox;

        // 确保坐标在图像范围内
        if x >= image.width() || y >= image.height() {
            continue;
        }

        let actual_width = width.min(image.width() - x);
        let actual_height = height.min(image.height() - y);

        let color = COLOR[detection.class_id % COLOR.len()];


        // 绘制多个略微偏移的矩形来创建粗线效果
        for offset_y in -1..=1 {
            for offset_x in -1..=1 {
                let offset_rect = imageproc::rect::Rect::at(x as i32 + offset_x, y as i32 + offset_y).of_size(actual_width, actual_height);
                draw_hollow_rect_mut(&mut image, offset_rect, color);
            }
        }

        // 绘制标签背景
        let label = format!("{}: {:.2}", detection.class_name, detection.confidence);
        let height = 12.4;
        let scale = PxScale {
            x: height * 2.0,
            y: height,
        };
        let text_color = Rgb([255, 255, 255]);

        let text_height = 20;
        let text_width = {
            let mut width = 0.0;
            for glyph_char in label.chars() {
                let glyph_id = font.glyph_id(glyph_char);
                if glyph_id.0 != 0 {
                    width += font.h_advance_unscaled(glyph_id);
                }
            }
            (width * scale.x / font.units_per_em().unwrap_or(1.0)).ceil() as u32
        };
        // println!("文本宽度: {}", text_width);
        // let text_width = 120;
        // 确保文本框在图像范围内
        let label_x = x;
        let label_y = y.saturating_sub(text_height + 2);

        // 绘制标签背景矩形
        for dy in 0..(text_height + 4).min(image.height() - label_y) {
            for dx in 0..(text_width + 4).min(image.width() - label_x) {
                if label_x + dx < image.width() && label_y + dy < image.height() {
                    image.put_pixel(label_x + dx, label_y + dy, color);
                }
            }
        }

        // 绘制文本
        draw_text_mut(
            &mut image,
            text_color,
            (label_x + 2) as i32,
            label_y as i32 + 2,
            scale,
            &font,
            &label,
        );
    }

    DynamicImage::ImageRgb8(image)
}
