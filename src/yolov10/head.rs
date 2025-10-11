use super::{block::Dfl, conv::ConvBlock};
use candle_core::{D, DType, IndexOp, Result, Tensor};
use candle_nn::{conv2d, seq, Conv2d, Module, VarBuilder};

fn make_anchors(
    xs0: &Tensor,
    xs1: &Tensor,
    xs2: &Tensor,
    (s0, s1, s2): (usize, usize, usize),
    grid_cell_offset: f64,
) -> Result<(Tensor, Tensor)> {
    let dev = xs0.device();
    let mut anchor_points = vec![];
    let mut stride_tensor = vec![];
    for (xs, stride) in [(xs0, s0), (xs1, s1), (xs2, s2)] {
        // xs is only used to extract the h and w dimensions.
        let (_, _, h, w) = xs.dims4()?;
        let sx = (Tensor::arange(0, w as u32, dev)?.to_dtype(DType::F32)? + grid_cell_offset)?;
        let sy = (Tensor::arange(0, h as u32, dev)?.to_dtype(DType::F32)? + grid_cell_offset)?;
        let sx = sx
            .reshape((1, sx.elem_count()))?
            .repeat((h, 1))?
            .flatten_all()?;
        let sy = sy
            .reshape((sy.elem_count(), 1))?
            .repeat((1, w))?
            .flatten_all()?;
        anchor_points.push(Tensor::stack(&[&sx, &sy], D::Minus1)?);
        stride_tensor.push((Tensor::ones(h * w, DType::F32, dev)? * stride as f64)?);
    }
    let anchor_points = Tensor::cat(anchor_points.as_slice(), 0)?;
    let stride_tensor = Tensor::cat(stride_tensor.as_slice(), 0)?.unsqueeze(1)?;
    Ok((anchor_points, stride_tensor))
}

fn dist2bbox(distance: &Tensor, anchor_points: &Tensor) -> Result<Tensor> {
    let chunks = distance.chunk(2, 1)?;
    let lt = &chunks[0];
    let rb = &chunks[1];
    let x1y1 = anchor_points.sub(lt)?;
    let x2y2 = anchor_points.add(rb)?;
    let c_xy = ((&x1y1 + &x2y2)? * 0.5)?;
    let wh = (&x2y2 - &x1y1)?;
    Tensor::cat(&[c_xy, wh], 1)
}

#[derive(Clone, Debug)]
struct Cv3 {
    seq1: (ConvBlock, ConvBlock),
    seq2: (ConvBlock, ConvBlock),
    conv: Conv2d,
}

impl Cv3 {
    fn load(vb: VarBuilder, x: usize, c3: usize, nc: usize) -> Result<Self> {
        // 第一个Sequential: Conv(x, x, 3, g=x), Conv(x, c3, 1)
        let seq1_0 = ConvBlock::load(vb.pp("0.0"), x, x, 3, 1, None, Some(x), true)?;
        let seq1_1 = ConvBlock::load(vb.pp("0.1"), x, c3, 1, 1, None, None, true)?;
        
        // 第二个Sequential: Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)
        let seq2_0 = ConvBlock::load(vb.pp("1.0"), c3, c3, 3, 1, None, Some(c3), true)?;
        let seq2_1 = ConvBlock::load(vb.pp("1.1"), c3, c3, 1, 1, None, None, true)?;
        
        // 最后的卷积层: Conv2d(c3, nc, 1)
        let conv = conv2d(c3, nc, 1, Default::default(), vb.pp("2"))?;
        
        Ok(Self {
            seq1: (seq1_0, seq1_1),
            seq2: (seq2_0, seq2_1),
            conv,
        })
    }
}

impl Module for Cv3 {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out1 = self.seq1.1.forward(&self.seq1.0.forward(xs)?)?;
        let out2 = self.seq2.1.forward(&self.seq2.0.forward(&out1)?)?;
        self.conv.forward(&out2)
    }
}

#[derive(Clone, Debug)]
pub struct V10DetectionHead {
    cv2: [(ConvBlock, ConvBlock, Conv2d); 3],
    cv3: [Cv3; 3],
    dfl: Dfl,
    one2one_cv2: [(ConvBlock, ConvBlock, Conv2d); 3],
    one2one_cv3: [Cv3; 3],
    ch: usize,
    no: usize,
    span: tracing::Span,
}

impl V10DetectionHead {
    pub fn load(vb: VarBuilder, nc: usize, filters: (usize, usize, usize)) -> Result<Self> {
        let reg_max = 16;
        let dfl = Dfl::load(vb.pp("model.23.dfl"), reg_max)?;
        
        let c2: usize = usize::max(usize::max(16,filters.0 / 4), reg_max * 4);
        let c3 = usize::max(filters.0, usize::min(nc, 100));

        let cv3 = [
            Cv3::load(vb.pp("model.23.cv3.0"), filters.0, c3, nc)?,
            Cv3::load(vb.pp("model.23.cv3.1"), filters.1, c3, nc)?,
            Cv3::load(vb.pp("model.23.cv3.2"), filters.2, c3, nc)?,
        ];


        let cv2 = [
            Self::load_cv2(vb.pp("model.23.cv2.0"), c2, reg_max, filters.0)?,
            Self::load_cv2(vb.pp("model.23.cv2.1"), c2, reg_max, filters.1)?,
            Self::load_cv2(vb.pp("model.23.cv2.2"), c2, reg_max, filters.2)?,
        ];
        let one2one_cv2 = [
            Self::load_cv2(vb.pp("model.23.one2one_cv2.0"), c2, reg_max, filters.0)?,
            Self::load_cv2(vb.pp("model.23.one2one_cv2.1"), c2, reg_max, filters.1)?,
            Self::load_cv2(vb.pp("model.23.one2one_cv2.2"), c2, reg_max, filters.2)?,
        ];
         let one2one_cv3 = [
            Cv3::load(vb.pp("model.23.one2one_cv3.0"), filters.0, c3, nc)?,
            Cv3::load(vb.pp("model.23.one2one_cv3.1"), filters.1, c3, nc)?,
            Cv3::load(vb.pp("model.23.one2one_cv3.2"), filters.2, c3, nc)?,
        ];
        let no = nc + reg_max * 4;
        Ok(Self {
            cv2,
            cv3,
            dfl,
            one2one_cv2,
            one2one_cv3,
            ch:reg_max,
            no,
            span: tracing::span!(tracing::Level::TRACE, "detection-head"),
        })
    }
    
    // fn load_cv3(
    //     vb: VarBuilder,
    //     c1: usize,
    //     nc: usize,
    //     filter: usize,
    // ) -> Result<(ConvBlock, ConvBlock, Conv2d)> {
    //     let block0 = ConvBlock::load(vb.pp("0"), filter, c1, 3, 1, None, Some(1), true)?;
    //     let block1 = ConvBlock::load(vb.pp("1"), c1, c1, 3, 1, None, Some(1), true)?;
    //     let conv = conv2d(c1, nc, 1, Default::default(), vb.pp("2"))?;
    //     Ok((block0, block1, conv))
    // }

    fn load_cv2(
        vb: VarBuilder,
        c2: usize,
        ch: usize,
        filter: usize,
    ) -> Result<(ConvBlock, ConvBlock, Conv2d)> {
        let block0 = ConvBlock::load(vb.pp("0"), filter, c2, 3, 1, None, Some(1), true)?;
        let block1 = ConvBlock::load(vb.pp("1"), c2, c2, 3, 1, None, Some(1), true)?;
        let conv = conv2d(c2, 4 * ch, 1, Default::default(), vb.pp("2"))?;
        Ok((block0, block1, conv))
    }

    fn forward_feat(
        &self,
        xs0: &Tensor,
        xs1: &Tensor,
        xs2: &Tensor,
        cv2: [(ConvBlock, ConvBlock, Conv2d); 3],
        cv3: [Cv3; 3],
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let forward_cv = |xs: &Tensor, i: usize| -> Result<Tensor> {
            let xs_2 = cv2[i].0.forward(xs)?;
            let xs_2 = cv2[i].1.forward(&xs_2)?;
            let xs_2 = cv2[i].2.forward(&xs_2)?;

            let xs_3 = cv3[i].forward(xs)?;
            Tensor::cat(&[&xs_2, &xs_3], 1)
        };
        
        let ys0 = forward_cv(xs0, 0)?;
        let ys1 = forward_cv(xs1, 1)?;
        let ys2 = forward_cv(xs2, 2)?;
        Ok((ys0, ys1, ys2))
    }

    fn forward_i(&self, xs0: &Tensor, xs1: &Tensor, xs2: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (anchors, strides) = make_anchors(xs0, xs1, xs2, (8, 16, 32), 0.5)?;
        let anchors = anchors.transpose(0, 1)?.unsqueeze(0)?;
        let strides = strides.transpose(0, 1)?;

        let reshape = |xs: &Tensor| {
            let d = xs.dim(0)?;
            let el = xs.elem_count();
            xs.reshape((d, self.no, el / (d * self.no)))
        };
        let ys0 = reshape(xs0)?;
        let ys1 = reshape(xs1)?;
        let ys2 = reshape(xs2)?;

        let x_cat = Tensor::cat(&[ys0, ys1, ys2], 2)?;
        let box_ = x_cat.i((.., ..self.ch * 4))?;
        let cls = x_cat.i((.., self.ch * 4..))?;

        let dbox = dist2bbox(&self.dfl.forward(&box_)?, &anchors)?;
        let dbox = dbox.broadcast_mul(&strides)?;
        let pred = Tensor::cat(&[dbox, candle_nn::ops::sigmoid(&cls)?], 1)?;
        Ok(pred)
    }

    pub fn forward(&self, xs0: &Tensor, xs1: &Tensor, xs2: &Tensor) -> Result<Tensor> {
        // For inference, we only use the one2one branch
        let one2one = self.forward_feat(xs0, xs1, xs2, self.one2one_cv2.clone(), self.one2one_cv3.clone())?;
        self.forward_i(&one2one.0, &one2one.1, &one2one.2)
    }
    
    // Add a method for training that returns both one2one and one2many branches
    pub fn forward_train(&self, xs0: &Tensor, xs1: &Tensor, xs2: &Tensor) -> Result<(Tensor, Tensor)> {
        let one2many = self.forward_feat(xs0, xs1, xs2, self.cv2.clone(), self.cv3.clone())?;
        let one2many_out = self.forward_i(&one2many.0, &one2many.1, &one2many.2)?;
        
        let one2one = self.forward_feat(xs0, xs1, xs2, self.one2one_cv2.clone(), self.one2one_cv3.clone())?;
        let one2one_out = self.forward_i(&one2one.0, &one2one.1, &one2one.2)?;
        
        Ok((one2many_out, one2one_out))
    }
}