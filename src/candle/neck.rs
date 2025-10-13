use crate::candle::block::C2fCIB;
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::{Module, VarBuilder};

use super::{
    Multiples,
    block::{C2f, SCDown},
    conv::ConvBlock,
};

#[derive(Clone, Debug)]
struct Upsample {
    scale_factor: usize,
}

impl Upsample {
    fn new(scale_factor: usize) -> Result<Self> {
        Ok(Upsample { scale_factor })
    }
}

impl Module for Upsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b_size, _channels, h, w) = xs.dims4()?;
        xs.upsample_nearest2d(self.scale_factor * h, self.scale_factor * w)
    }
}

#[derive(Debug)]
pub struct YoloNeck {
    up: Upsample,
    n1: C2f,
    n2: C2f,
    n3: ConvBlock,
    n4: C2f,
    n5: SCDown,
    n6: C2fCIB,
    span: tracing::Span,
}

impl YoloNeck {
    pub fn load(vb: VarBuilder, m: Multiples) -> Result<Self> {
        let up = Upsample::new(2)?;
        let (w, r, d) = (m.width, m.ratio, m.depth);
        let n = (3. * d).round() as usize;
        let n1 = C2f::load(
            vb.pp("model.13"),
            (512. * w * (1. + r)) as usize,
            (512. * w) as usize,
            n,
            false,
        )?;
        let n2 = C2f::load(
            vb.pp("model.16"),
            (768. * w) as usize,
            (256. * w) as usize,
            n,
            false,
        )?;
        let n3 = ConvBlock::load(
            vb.pp("model.17"),
            (256. * w) as usize,
            (256. * w) as usize,
            3,
            2,
            Some(1),
            Some(1),
            true,
        )?;
        let n4 = C2f::load(
            vb.pp("model.19"),
            (768. * w) as usize,
            (512. * w) as usize,
            n,
            false,
        )?;
        let n5 = SCDown::load(
            vb.pp("model.20"),
            (512. * w) as usize,
            (512. * w) as usize,
            3,
            2,
        )?;
        let n6 = C2fCIB::load(
            vb.pp("model.22"),
            (512. * w * (1. + r)) as usize,
            (512. * w * r) as usize,
            n,
            false,
            true,
            1,
            0.5,
        )?;
        Ok(Self {
            up,
            n1,
            n2,
            n3,
            n4,
            n5,
            n6,
            span: tracing::span!(tracing::Level::TRACE, "neck"),
        })
    }

    pub fn forward(
        &self,
        p3: &Tensor,
        p4: &Tensor,
        p5: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let _enter = self.span.enter();
        let x = self
            .n1
            .forward(&Tensor::cat(&[&self.up.forward(p5)?, p4], 1)?)?;
        let head_1 = self
            .n2
            .forward(&Tensor::cat(&[&self.up.forward(&x)?, p3], 1)?)?;
        let head_2 = self
            .n4
            .forward(&Tensor::cat(&[&self.n3.forward(&head_1)?, &x], 1)?)?;
        let head_3 = self
            .n6
            .forward(&Tensor::cat(&[&self.n5.forward(&head_2)?, p5], 1)?)?;
        
        Ok((head_1, head_2, head_3))
    }
}
