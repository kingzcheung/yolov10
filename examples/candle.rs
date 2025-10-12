use std::error::Error;

use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use yolov10::yolov10::{Multiples, YoloV10};
fn main() -> Result<(), Box<dyn Error>> {
    let input_data = include_bytes!("../testdata/bus.jpg");
    let device = candle_core::Device::cuda_if_available(0)?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &["yolov10s.safetensors"],
            candle_core::DType::F32,
            &device,
        )
    }?;

    let original_image = image::load_from_memory(input_data)?;
    let (width, height) = {
        let w = original_image.width() as usize;
        let h = original_image.height() as usize;
        if w < h {
            let w = w * 640 / h;
            // Sizes have to be divisible by 32.
            (w / 32 * 32, 640)
        } else {
            let h = h * 640 / w;
            (640, h / 32 * 32)
        }
    };

    let image_t = {
        let img = original_image.resize_exact(
            width as u32,
            height as u32,
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

    let yolo = YoloV10::load(vb, Multiples::s(), 80)?;

    // let xs = vec![0.0f32; 640 * 640 * 3];
    // let xs = candle_core::Tensor::from_vec(xs, (1, 3, 640, 640), &device)?;
    let output = yolo.forward(&image_t)?;


    println!("{:?}", output.shape());
    Ok(())
}
