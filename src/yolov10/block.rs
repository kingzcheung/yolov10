use super::sequential::{seq, Sequential};
use super::conv::ConvBlock;
use candle_core::{Result, Tensor, D};
use candle_nn::{conv2d_no_bias, func, ops::softmax, Conv2d, Module, VarBuilder};

#[derive(Debug)]
pub struct SCDown {
    cv1: ConvBlock,
    cv2: ConvBlock,
}

impl SCDown {
    pub fn load(vb: VarBuilder, c1: usize, c2: usize, k: usize, s: usize) -> Result<Self> {
        let cv1 = ConvBlock::load(vb.pp("cv1"), c1, c2, 1, 1, None, None, true)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), c2, c2, k, s, None, Some(c2), false)?;
        Ok(Self { cv1, cv2 })
    }
}

impl Module for SCDown {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let ys = self.cv1.forward(xs)?;
        let ys = self.cv2.forward(&ys)?;
        Ok(ys)
    }
}

pub struct RepVGGDW {
    conv: ConvBlock,
    conv1: ConvBlock,
    dim: usize,
}

impl RepVGGDW {
    pub fn load(vb: VarBuilder, ed: usize) -> Result<Self> {
        let conv = ConvBlock::load(vb.pp("conv"), ed, ed, 7, 1, Some(3), Some(ed), false)?;
        let conv1 = ConvBlock::load(vb.pp("conv1"), ed, ed, 3, 1, Some(1), Some(ed), false)?;
        Ok(Self {
            conv,
            conv1,
            dim: ed,
        })
    }
}

impl Module for RepVGGDW {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let ys1 = self.conv.forward(xs)?;
        let ys2 = self.conv1.forward(xs)?;
        // println!("{:?} {:?}", ys1.shape(), ys2.shape());
        let ys = ys1.broadcast_add(&ys2)?;
        candle_nn::ops::silu(&ys)
    }
}

#[derive(Debug)]
pub struct CIB {
    cv1: Sequential,
    add: bool,
}

impl CIB {
    pub fn load(
        vb: VarBuilder,
        c1: usize,
        c2: usize,
        shortcut: bool,
        e: usize,
        lk: bool,
    ) -> Result<Self> {
        let c_ = c1 * e;
        let mut cv1 = seq();

        // Add first two layers
        cv1 = cv1
            .add(ConvBlock::load(
                vb.pp("cv1.0"),
                c1,
                c1,
                3,
                1,
                None,
                Some(c1),
                true,
            )?)
            .add(ConvBlock::load(
                vb.pp("cv1.1"),
                c1,
                2 * c_,
                1,
                1,
                None,
                None,
                true,
            )?);

        // Add third layer based on lk flag using func wrapper
        if lk {
            let rep_vgg_dw = RepVGGDW::load(vb.pp("cv1.2"), 2 * c_)?;
            cv1 = cv1.add(func(move |xs| rep_vgg_dw.forward(xs)));
        } else {
            let conv_block =
                ConvBlock::load(vb.pp("cv1.2"), 2 * c_, 2 * c_, 3, 1, None, Some(2 * c_), true)?;
            cv1 = cv1.add(func(move |xs| conv_block.forward(xs)));
        }

        // Add last two layers
        cv1 = cv1
            .add(ConvBlock::load(
                vb.pp("cv1.3"),
                2 * c_,
                c2,
                1,
                1,
                None,
                Some(1),
                true,
            )?)
            .add(ConvBlock::load(
                vb.pp("cv1.4"),
                c2,
                c2,
                3,
                1,
                None,
                Some(c2),
                true,
            )?);

        Ok(Self {
            cv1,
            add: shortcut && c1 == c2,
        })
    }
}

impl Module for CIB {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.cv1.forward(xs)
    }
}

#[derive(Clone, Debug)]
struct Bottleneck {
    cv1: ConvBlock,
    cv2: ConvBlock,
    residual: bool,
    span: tracing::Span,
}

impl Bottleneck {
    fn load(vb: VarBuilder, c1: usize, c2: usize, shortcut: bool) -> Result<Self> {
        let channel_factor = 1.;
        let c_ = (c2 as f64 * channel_factor) as usize;
        let cv1 = ConvBlock::load(vb.pp("cv1"), c1, c_, 3, 1, None, Some(1), true)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), c_, c2, 3, 1, None, Some(1), true)?;
        let residual = c1 == c2 && shortcut;
        Ok(Self {
            cv1,
            cv2,
            residual,
            span: tracing::span!(tracing::Level::TRACE, "bottleneck"),
        })
    }
}

impl Module for Bottleneck {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let ys = self.cv2.forward(&self.cv1.forward(xs)?)?;
        if self.residual { xs + ys } else { Ok(ys) }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct C2f {
    cv1: ConvBlock,
    cv2: ConvBlock,
    bottleneck: Vec<Bottleneck>,
    span: tracing::Span,
}

impl C2f {
    /// def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False)
    pub fn load(vb: VarBuilder, c1: usize, c2: usize, n: usize, shortcut: bool) -> Result<Self> {
        let c = (c2 as f64 * 0.5) as usize;
        let cv1 = ConvBlock::load(vb.pp("cv1"), c1, 2 * c, 1, 1, None, Some(1), true)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), (2 + n) * c, c2, 1, 1, None, Some(1), true)?;
        let mut bottleneck = Vec::with_capacity(n);
        for idx in 0..n {
            let b = Bottleneck::load(vb.pp(format!("m.{idx}")), c, c, shortcut)?;
            bottleneck.push(b)
        }
        Ok(Self {
            cv1,
            cv2,
            bottleneck,
            span: tracing::span!(tracing::Level::TRACE, "c2f"),
        })
    }
}

impl Module for C2f {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let ys = self.cv1.forward(xs)?;
        let mut ys = ys.chunk(2, 1)?;
        for m in self.bottleneck.iter() {
            ys.push(m.forward(ys.last().unwrap())?)
        }
        let zs = Tensor::cat(ys.as_slice(), 1)?;
        self.cv2.forward(&zs)
    }
}

#[derive(Debug)]
pub struct C2fCIB {
    cv1: ConvBlock,
    cv2: ConvBlock,
    cib: Vec<CIB>,
    span: tracing::Span,
}

impl C2fCIB {
    /// def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5)
    /// In the Python version:
    /// class C2fCIB(C2f):
    ///    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
    ///        super().__init__(c1, c2, n, shortcut, g, e)
    ///        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: VarBuilder,
        c1: usize,
        c2: usize,
        n: usize,
        shortcut: bool,
        lk: bool,
        g: usize,
        e: f64,
    ) -> Result<Self> {
        // Calculate c as in the C2f implementation
        let c = (c2 as f64 * 0.5) as usize;
        let cv1 = ConvBlock::load(vb.pp("cv1"), c1, 2 * c, 1, 1, None, Some(g), true)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), (2 + n) * c, c2, 1, 1, None, Some(g), true)?;
        let mut cib = Vec::with_capacity(n);
        for idx in 0..n {
            // CIB(self.c, self.c, shortcut, e=1.0, lk=lk)
            let b = CIB::load(vb.pp(format!("m.{idx}")), c, c, shortcut, 1, lk)?;
            cib.push(b)
        }
        Ok(Self {
            cv1,
            cv2,
            cib,
            span: tracing::span!(tracing::Level::TRACE, "c2fcib"),
        })
    }
}

impl Module for C2fCIB {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let ys = self.cv1.forward(xs)?;
        let mut ys = ys.chunk(2, 1)?;
        for m in self.cib.iter() {
            // Pass through the CIB module using forward method
            let output = m.forward(ys.last().unwrap())?;
            ys.push(output)
        }
        let zs = Tensor::cat(ys.as_slice(), 1)?;
        self.cv2.forward(&zs)
    }
}

#[derive(Clone, Debug)]
pub struct Dfl {
    conv: Conv2d,
    num_classes: usize,
    span: tracing::Span,
}

impl Dfl {
    pub fn load(vb: VarBuilder, num_classes: usize) -> Result<Self> {
        let conv = conv2d_no_bias(num_classes, 1, 1, Default::default(), vb.pp("conv"))?;
        Ok(Self {
            conv,
            num_classes,
            span: tracing::span!(tracing::Level::TRACE, "dfl"),
        })
    }
}

impl Module for Dfl {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, _channels, anchors) = xs.dims3()?;
        let xs = xs
            .reshape((b_sz, 4, self.num_classes, anchors))?
            .transpose(2, 1)?;
        let xs = candle_nn::ops::softmax(&xs, 1)?;
        self.conv.forward(&xs)?.reshape((b_sz, 4, anchors))
    }
}


#[derive(Clone, Debug)]
pub struct Attention{
    qkv: ConvBlock,
    proj: ConvBlock,
    pe: ConvBlock,
    num_heads: usize,
    key_dim: usize,
    scale: f64,
    head_dim: usize,
}

impl Attention {
    /// num_heads=8, attn_ratio=0.5
    pub fn load(vb:VarBuilder,dim:usize,num_heads:usize,attn_ratio:f64)->Result<Self> {
        let head_dim = dim / num_heads;
        let key_dim = (head_dim as f64 * attn_ratio) as usize;
        let scale = (key_dim as f64).powf(-0.5);
        let nh_kd = key_dim * num_heads;
        let h = dim + nh_kd * 2;

        let qkv = ConvBlock::load(vb.pp("qkv"), dim, h, 1, 1, None, None, false)?;
        let proj = ConvBlock::load(vb.pp("proj"), dim, dim, 1, 1, None, None, false)?;
        let pe = ConvBlock::load(vb.pp("pe"), dim, dim, 3, 1, None, Some(dim), false)?;

        Ok(
            Self {
                qkv,
                proj,
                pe,
                num_heads,
                key_dim,
                scale,
                head_dim,
            }
        )
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let shape = xs.shape();
        let (b,c,h,w) = shape.dims4()?;

        let n = h * w;
        let qkv = self.qkv.forward(xs)?;

        let rs = qkv.reshape(
            (b,self.num_heads,self.key_dim*2 + self.head_dim,n)
        )?;
        // let (q,k,v) = rs.split(self.key_dim*2 + self.head_dim, 2)?;
        let q = rs.narrow(2, 0, self.key_dim)?;  // 从dim=2的第0个位置开始，取key_dim个元素
        let k = rs.narrow(2, self.key_dim, self.key_dim)?;  // 从dim=2的第key_dim个位置开始，取key_dim个元素
        let v = rs.narrow(2, self.key_dim * 2, self.head_dim)?;  // 从dim=2的第key_dim*2个位置开始，取head_dim个元素

        let attn = (q.transpose(D::Minus2, D::Minus1)?.matmul(&k)? * self.scale)?;

        let attn = softmax(&attn, D::Minus1)?;

        let attn_t = attn.transpose(D::Minus2, D::Minus1)?;
        let mat = v.matmul(&attn_t)?;
        let matmul_reshaped = mat.reshape((b,c,h,w))?;

        let v_reshaped = v.reshape((b,c,h,w))?;
        let pe = self.pe.forward(&v_reshaped)?;

        let x = matmul_reshaped.broadcast_add(&pe)?;
        self.proj.forward(&x) 

    }
}

#[derive(Debug)]
pub struct Psa {
    c: usize,
    cv1: ConvBlock,
    cv2: ConvBlock,
    attn: Attention,
    ffn: Sequential,
}

impl Psa {
    /// e: 0.5
    pub fn load(vb: VarBuilder, c1: usize,c2:usize,e: f64) ->Result<Self> {
        assert_eq!(c1, c2);
        let c = (c1 as f64 * e) as usize;
        let cv1 = ConvBlock::load(vb.pp("cv1"), c1, 2*c, 1, 1, None, None, false)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), 2*c, c1, 1, 1, None, None, false)?;

        let attn = Attention::load(vb.pp("attn"), c, c/64, 0.5)?;

        let ffn = seq()
        .add(
            ConvBlock::load(vb.pp("ffn.0"), c, c*2, 1, 1, None, None, true)?
        )
        .add(ConvBlock::load(vb.pp("ffn.1"), c*2,c, 1, 1, None, None, false)?);


        Ok(
            Self {
                c,
                cv1,
                cv2,
                attn,
                ffn,
            }
        )
    }
}
impl Module for Psa {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let cv1_output = self.cv1.forward(xs)?;
        let a = cv1_output.narrow(1, 0, self.c)?;
        let b = cv1_output.narrow(1, self.c, self.c)?;
        
        let attn = self.attn.forward(&b)?;
        let b = b.broadcast_add(&attn)?;

        let ffn = self.ffn.forward(&b)?;
        let b = b.broadcast_add(&ffn)?;

        let cat = Tensor::cat(&[a,b], 1)?;
        self.cv2.forward(&cat)
    }
}


#[derive(Clone, Debug)]
pub(crate) struct Sppf {
    cv1: ConvBlock,
    cv2: ConvBlock,
    k: usize,
    span: tracing::Span,
}

impl Sppf {
    pub fn load(vb: VarBuilder, c1: usize, c2: usize, k: usize) -> Result<Self> {
        let c_ = c1 / 2;
        let cv1 = ConvBlock::load(vb.pp("cv1"), c1, c_, 1, 1, None,None,true)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), c_ * 4, c2, 1, 1, None,None,true)?;
        Ok(Self {
            cv1,
            cv2,
            k,
            span: tracing::span!(tracing::Level::TRACE, "sppf"),
        })
    }
}

impl Module for Sppf {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_, _, _, _) = xs.dims4()?;
        let xs = self.cv1.forward(xs)?;
        let xs2 = xs
            .pad_with_zeros(2, self.k / 2, self.k / 2)?
            .pad_with_zeros(3, self.k / 2, self.k / 2)?
            .max_pool2d_with_stride(self.k, 1)?;
        let xs3 = xs2
            .pad_with_zeros(2, self.k / 2, self.k / 2)?
            .pad_with_zeros(3, self.k / 2, self.k / 2)?
            .max_pool2d_with_stride(self.k, 1)?;
        let xs4 = xs3
            .pad_with_zeros(2, self.k / 2, self.k / 2)?
            .pad_with_zeros(3, self.k / 2, self.k / 2)?
            .max_pool2d_with_stride(self.k, 1)?;
        self.cv2.forward(&Tensor::cat(&[&xs, &xs2, &xs3, &xs4], 1)?)
    }
}