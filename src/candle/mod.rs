pub use candle_nn::{VarBuilder};
pub use candle_core::{Result, Tensor,DType,Device};
use crate::candle::{backbone::Backbone, head::V10DetectionHead, neck::YoloNeck};

pub(crate) mod block;
pub(crate) mod conv;
pub mod sequential;
pub mod backbone;
pub mod neck;
pub mod head;
pub mod op;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Multiples {
    depth: f64,
    width: f64,
    ratio: f64,
}

impl Multiples {
    // [depth, width, max_channels]
    // [0.33, 0.25, 1024] 
    pub fn n() -> Self {
        Self {
            depth: 0.33,
            width: 0.25,
            ratio: 2.0,
        }
    }
    // [0.33, 0.50, 1024]
    pub fn s() -> Self {
        Self {
            depth: 0.33,
            width: 0.50,
            ratio: 2.0,
        }
    }
    // [0.67, 0.75, 768] 
    pub fn m() -> Self {
        Self {
            depth: 0.67,
            width: 0.75,
            ratio: 1.5,
        }
    }
    //[1.00, 1.00, 512]
    pub fn l() -> Self {
        Self {
            depth: 1.00,
            width: 1.00,
            ratio: 1.0,
        }
    }
    // [1.00, 1.25, 512]
    pub fn x() -> Self {
        Self {
            depth: 1.00,
            width: 1.25,
            ratio: 1.0,
        }
    }

    fn filters(&self) -> (usize, usize, usize) {
        let f1 = (256. * self.width) as usize;
        let f2 = (512. * self.width) as usize;
        let f3 = (512. * self.width * self.ratio) as usize;
        (f1, f2, f3)
    }
}


pub struct YoloV10 {
    backbone: Backbone,
    neck: YoloNeck,
    head: V10DetectionHead,
}

impl YoloV10 {
    pub fn load(vb: VarBuilder, m: Multiples, num_classes: usize) -> Result<Self> {
        let backbone = Backbone::load(vb.clone(), m)?;
        let neck = YoloNeck::load(vb.clone(), m)?;
        let head = V10DetectionHead::load(vb, num_classes, m.filters())?;
        Ok(Self { backbone, neck, head })
    }
    
    /// Forward pass for inference
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (xs1, xs2, xs3) = self.backbone.forward(xs)?;
        let (xs1, xs2, xs3) = self.neck.forward(&xs1, &xs2, &xs3)?;
        
        self.head.forward(&xs1, &xs2, &xs3)
    }
    
    /// Forward pass for training which returns both one2many and one2one outputs
    pub fn forward_train(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (xs1, xs2, xs3) = self.backbone.forward(xs)?;
        let (xs1, xs2, xs3) = self.neck.forward(&xs1, &xs2, &xs3)?;
        self.head.forward_train(&xs1, &xs2, &xs3)
    }
}