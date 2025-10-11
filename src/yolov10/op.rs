use candle_core::{Result, Tensor, D};

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
        let topk_indices = sorted_indices.narrow(D::Minus1, 0, topk)?.contiguous()?;
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
        // 获取两个张量的形状
        let lhs_shape = self.dims();
        let rhs_shape = other.dims();
        
        // 检查是否可以广播
        if !can_broadcast(lhs_shape, rhs_shape) {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
            }
            .bt());
        }
        
        // 广播两个张量到相同的形状
        let broadcast_shape = broadcast_shape(lhs_shape, rhs_shape)?;
        let lhs_broadcasted = if lhs_shape != broadcast_shape {
            self.broadcast_as(&broadcast_shape)?
        } else {
            self.clone()
        };
        
        let rhs_broadcasted = if rhs_shape != broadcast_shape {
            other.broadcast_as(&broadcast_shape)?
        } else {
            other.clone()
        };
        
        // 执行求余运算
        lhs_broadcasted.rem(&rhs_broadcasted)
    }
}

// 辅助方法：检查两个形状是否可以广播
fn can_broadcast(shape1: &[usize], shape2: &[usize]) -> bool {
    let mut s1 = shape1.iter().rev();
    let mut s2 = shape2.iter().rev();
    
    loop {
        match (s1.next(), s2.next()) {
            (Some(&d1), Some(&d2)) => {
                if d1 != d2 && d1 != 1 && d2 != 1 {
                    return false;
                }
            }
            (Some(_), None) | (None, Some(_)) => break,
            (None, None) => break,
        }
    }
    true
}

// 辅助方法：计算广播后的形状
fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>> {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = len1.max(len2);
    
    let mut result = Vec::with_capacity(max_len);
    
    for i in 0..max_len {
        let d1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
        let d2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };
        
        if d1 == d2 || d1 == 1 || d2 == 1 {
            result.push(d1.max(d2));
        } else {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: shape1.to_vec(),
                rhs: shape2.to_vec(),
            }
            .bt());
        }
    }
    
    result.reverse();
    Ok(result)
}