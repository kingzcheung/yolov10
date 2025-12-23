use candle_core::{DType, Device, Tensor};

fn main() {
    let a = Tensor::zeros(
        1025,
        DType::F32,
        &Device::cuda_if_available(0).unwrap(),
    )
    .unwrap();
    dbg!(&a.arg_sort_last_dim(true));
}