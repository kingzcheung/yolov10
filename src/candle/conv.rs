use candle_core::Result;
use candle_core::Tensor;
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder, batch_norm, conv2d_no_bias};

#[derive(Clone, Debug)]
pub(crate) struct ConvBlock {
    conv: Conv2d,
    act: bool,
    span: tracing::Span,
}

impl ConvBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: VarBuilder,
        c1: usize,
        c2: usize,
        k: usize,
        stride: usize,
        padding: Option<usize>,
        group: Option<usize>,
        act: bool,
    ) -> Result<Self> {
        let padding = padding.unwrap_or(k / 2);
        let cfg = Conv2dConfig {
            padding,
            stride,
            groups: group.unwrap_or(1),
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        // 先创建卷积层
        let conv = conv2d_no_bias(c1, c2, k, cfg, vb.pp("conv"))?;

        // 然后创建BN层并吸收
        let bn = batch_norm(c2, 1e-3, vb.pp("bn"))?;
        let conv = conv.absorb_bn(&bn)?;
        Ok(Self {
            conv,
            act,
            span: tracing::span!(tracing::Level::TRACE, "conv-block"),
        })
    }
}

impl Module for ConvBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let xs = self.conv.forward(xs)?;
        if self.act {
            candle_nn::ops::silu(&xs)
        } else {
            Ok(xs)
        }
    }
}
