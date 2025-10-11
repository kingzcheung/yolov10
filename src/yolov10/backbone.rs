use super::{block::{C2fCIB, SCDown}, Multiples};
use super::block::C2f;
use super::block::Psa;
use super::block::Sppf;
use super::conv::ConvBlock;
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::{Module, VarBuilder};

#[derive(Debug)]
pub struct Backbone {
    b1_0: ConvBlock,
    b1_1: ConvBlock,
    b2_0: C2f,
    b2_1: ConvBlock,
    b2_2: C2f,
    b3_0: SCDown,
    b3_1: C2f,
    b4_0: SCDown,
    b4_1: C2fCIB,
    b5: Sppf,
    psa: Psa,
    span: tracing::Span,
}

impl Backbone {
    pub fn load(vb: VarBuilder, m: Multiples) -> Result<Self> {
        let (w, r, d) = (m.width, m.ratio, m.depth);
        let b1_0 = ConvBlock::load(
            vb.pp("model.0"),
            3,
            (64. * w) as usize,
            3,
            2,
            Some(1),
            Some(1),
            true,
        )?;
        let b1_1 = ConvBlock::load(
            vb.pp("model.1"),
            (64. * w) as usize,
            (128. * w) as usize,
            3,
            2,
            Some(1),
            Some(1),
            true,
        )?;
        let b2_0 = C2f::load(
            vb.pp("model.2"),
            (128. * w) as usize,
            (128. * w) as usize,
            (3. * d).round() as usize,
            true,
        )?;
        let b2_1 = ConvBlock::load(
            vb.pp("model.3"),
            (128. * w) as usize,
            (256. * w) as usize,
            3,
            2,
            Some(1),
            Some(1),
            true,
        )?;
        let b2_2 = C2f::load(
            vb.pp("model.4"),
            (256. * w) as usize,
            (256. * w) as usize,
            (6. * d).round() as usize,
            true,
        )?;
        let b3_0 = SCDown::load(
            vb.pp("model.5"),
            (256. * w) as usize,
            (512. * w) as usize,
            3,
            2,
        )?;
        let b3_1 = C2f::load(
            vb.pp("model.6"),
            (512. * w) as usize,
            (512. * w) as usize,
            (6. * d).round() as usize,
            true,
        )?;
        let b4_0 = SCDown::load(
            vb.pp("model.7"),
            (512. * w) as usize,
            (512. * w * r) as usize,
            3,
            2,
        )?;
        /* Python version: [-1, 3, C2fCIB, [1024, True, True]]
         * class C2fCIB(C2f):
         * def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
         */
        let b4_1 = C2fCIB::load(
            vb.pp("model.8"),
            (512. * w * r) as usize,
            (512. * w * r) as usize,
            (3. * d).round() as usize,
            true,  /* shortcut */
            true,  /* lk */
            1,     /* g */
            0.5,   /* e */
        )?;
        let b5 = Sppf::load(
            vb.pp("model.9"),
            (512. * w * r) as usize,
            (512. * w * r) as usize,
            5,
        )?;

        let psa = Psa::load(vb.pp("model.10"), (512. * w * r) as usize, (512. * w * r) as usize, 0.5)?;
        Ok(Self {
            b1_0,
            b1_1,
            b2_0,
            b2_1,
            b2_2,
            b3_0,
            b3_1,
            b4_0,
            b4_1,
            b5,
            psa,
            span: tracing::span!(tracing::Level::TRACE, "backbone"),
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let _enter = self.span.enter();
        let x1 = self.b1_1.forward(&self.b1_0.forward(xs)?)?;
        let x2 = self
            .b2_2
            .forward(&self.b2_1.forward(&self.b2_0.forward(&x1)?)?)?;
        let x3 = self.b3_1.forward(&self.b3_0.forward(&x2)?)?;
        let x4 = self.b4_1.forward(&self.b4_0.forward(&x3)?)?;
        let x5 = self.b5.forward(&x4)?;
        let x5 = self.psa.forward(&x5)?; 
        Ok((x2, x3, x5))
    }
}
