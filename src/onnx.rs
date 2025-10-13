use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ort::{
    inputs,
    session::{Session, SessionOutputs, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use ndarray::Array;
use std::borrow::Cow;
use std::error::Error;


#[cfg(feature = "onnx-coreml")]
use ort::execution_providers::CoreMLExecutionProvider;

#[cfg(feature = "onnx-cuda")]
use ort::execution_providers::CUDAExecutionProvider;

#[cfg(feature = "onnx-directml")]
use ort::execution_providers::DirectMLExecutionProvider;

#[cfg(feature = "onnx-tensorrt")]
use ort::execution_providers::TensorRTExecutionProvider;

use crate::{filter_detections, Detection};



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
            class_labels: None,
        })
    }

    pub fn new_with_labels(
        model_path: &str,
        class_labels: &'a [&str],
    ) -> Result<Self, Box<dyn Error>> {
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
            class_labels: Some(class_labels.iter().map(|&s| Cow::Borrowed(s)).collect()),
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
    pub fn run_inference(
        &mut self,
        buffer: &[u8],
        confidence_threshold: f32,
    ) -> Result<Vec<Detection>, Box<dyn Error>> {
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
        println!("输出张量形状: {:?}", _output_shape);
        let output_vec: Vec<f32> = output_data.to_vec();
        // println!("{:?}", output_vec);
        let detections = filter_detections(
            &output_vec,
            confidence_threshold,
            640,
            640,
            orig_width,
            orig_height,
        );

        Ok(detections)
    }
}
