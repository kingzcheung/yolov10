use std::ops::Rem;

use candle_core::{D, Result, Shape, Tensor};

pub struct TopKOutput {
    pub values: Tensor,
    pub indices: Tensor,
}
pub trait TopKLastDimOp {
    fn topk(&self, topk: usize) -> Result<TopKOutput>;
}

impl TopKLastDimOp for Tensor {
    fn topk(&self, topk: usize) -> Result<TopKOutput> {
        // Sorted descending
        let sorted_indices = self.arg_sort_last_dim(false)?;
        // 获取最后一维的大小
        let last_dim_size = sorted_indices.dim(D::Minus1)?;
        // 确保不超过最后一维的实际大小，符合PyTorch的torch.topk行为
        let actual_topk = topk.min(last_dim_size);
        let topk_indices = sorted_indices
            .narrow(D::Minus1, 0, actual_topk)?
            .contiguous()?;
        Ok(TopKOutput {
            values: self.gather(&topk_indices, D::Minus1)?,
            indices: topk_indices,
        })
    }
}

/// Extension trait for Tensor operations
pub trait TensorOps {
    /// Broadcast remainder operation, equivalent to torch.remainder with broadcasting
    fn broadcast_rem(&self, other: &Tensor) -> Result<Tensor>;
}
impl TensorOps for Tensor {
    fn broadcast_rem(&self, other: &Tensor) -> Result<Tensor> {
        // 获取广播后的形状
        let broadcast_shape = broadcast_shape(self.shape(), other.shape())?;

        // 对两个张量进行广播扩展
        let self_expanded = self.expand(&broadcast_shape)?;
        let other_expanded = other.expand(&broadcast_shape)?;

        // 转换为二维数组进行逐元素取模运算
        let self_data = self_expanded.to_vec2::<u32>()?;
        let other_data = other_expanded.to_vec2::<u32>()?;

        // 执行逐元素取模运算
        let result: Vec<Vec<u32>> = self_data
            .into_iter()
            .zip(other_data.into_iter())
            .map(|(row1, row2)| {
                row1.into_iter().zip(row2.into_iter())
                    .map(|(a, b)| a % b)
                    .collect()
            })
            .collect();

        // 将结果展平并重塑为原来的形状
        let flat_result: Vec<u32> = result.into_iter().flatten().collect();
        Tensor::from_vec(flat_result, &broadcast_shape, self.device())
    }
}


/// Helper function to compute broadcast shape following PyTorch's broadcasting rules
fn broadcast_shape(shape1: &Shape, shape2: &Shape) -> Result<Shape> {
    let dims1 = shape1.dims();
    let dims2 = shape2.dims();
    let max_len = dims1.len().max(dims2.len());

    let mut result = Vec::with_capacity(max_len);

    // Pad the shorter dimension list with leading 1s
    for i in 0..max_len {
        let dim1 = if i < max_len - dims1.len() {
            1
        } else {
            dims1[i - (max_len - dims1.len())]
        };
        let dim2 = if i < max_len - dims2.len() {
            1
        } else {
            dims2[i - (max_len - dims2.len())]
        };

        // Check if dimensions are compatible for broadcasting
        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            return Err(candle_core::Error::Msg(format!(
                "Shapes {:?} and {:?} are not broadcastable",
                shape1, shape2
            )));
        }

        result.push(dim1.max(dim2));
    }

    Ok(Shape::from(result))
}
