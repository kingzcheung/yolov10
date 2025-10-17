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

pub trait TensorRemOps {
    fn broadcast_rem(&self, other: &Tensor) -> Result<Tensor>;
}
impl TensorRemOps for Tensor {
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
            .zip(other_data)
            .map(|(row1, row2)| {
                row1.into_iter().zip(row2)
                    .map(|(a, b)| a % b)
                    .collect()
            })
            .collect();

        // 将结果展平并重塑为原来的形状
        let flat_result: Vec<u32> = result.into_iter().flatten().collect();
        Tensor::from_vec(flat_result, &broadcast_shape, self.device())
    }
}



/// 广播两个张量形状，计算广播后的目标形状
/// 
/// 该函数实现了NumPy风格的广播规则，用于确定两个不同形状的张量
/// 在进行元素级操作时应该广播到的目标形状。
/// 
/// # 参数
/// * `shape1` - 第一个张量的形状引用
/// * `shape2` - 第二个张量的形状引用
/// 
/// # 返回值
/// * `Ok(Shape)` - 广播成功时返回目标形状
/// * `Err(Error)` - 当两个形状无法广播时返回错误
/// 
/// # 广播规则
/// 1. 每个维度大小相等，或其中一个为1
/// 2. 结果形状的每个维度取两个输入维度的最大值
/// 3. 维度不足的部分视为1
fn broadcast_shape(shape1: &Shape, shape2: &Shape) -> Result<Shape> {
    let dims1 = shape1.dims();
    let dims2 = shape2.dims();
    let max_len = dims1.len().max(dims2.len());

    let mut result = Vec::with_capacity(max_len);

    // 从右到左对齐两个形状的维度，计算广播后的目标形状
    for i in 0..max_len {
        // 处理维度长度不同的情况，将较短的形状左侧补1
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

        // 检查广播兼容性：相同维度或其中一个为1才能广播
        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            return Err(candle_core::Error::Msg(format!(
                "Shapes {shape1:?} and {shape2:?} are not broadcastable"
            )));
        }

        result.push(dim1.max(dim2));
    }

    Ok(Shape::from(result))
}
