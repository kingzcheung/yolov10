use std::error::Error;

use candle_nn::VarBuilder;
use yolov10::yolov10::{Multiples, YoloV10};
fn main() -> Result<(), Box<dyn Error>> {

    let device = candle_core::Device::cuda_if_available(0)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[
        "yolov10s.safetensors"
    ], candle_core::DType::F32, &device) }?;

    let yolo = YoloV10::load(vb, Multiples::s(), 80)?;
    
    let xs = vec![0.0f32; 640 * 640 * 3];
    let xs = candle_core::Tensor::from_vec(xs, (1, 3, 640, 640), &device)?;
    let output = yolo.forward(&xs)?;

    println!("{:?}", output);
    Ok(())
}
