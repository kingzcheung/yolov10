//! Sequential Layer
//!
//! A sequential layer used to chain multiple layers and closures.
use candle_core::{Module, Result, Tensor};
use std::fmt;

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl fmt::Debug for Sequential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sequential")
            .field("layers", &format!("Vec<Box<dyn Module>> with {} layers", self.layers.len()))
            .finish()
    }
}

/// Creates a new empty sequential layer.
pub fn seq() -> Sequential {
    Sequential { layers: vec![] }
}

impl Sequential {
    /// The number of sub-layers embedded in this layer.
    pub fn len(&self) -> i64 {
        self.layers.len() as i64
    }

    /// Returns true if this layer does not have any sub-layer.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Module for Sequential {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?
        }
        Ok(xs)
    }
}

impl Sequential {
    /// Appends a layer after all the current layers.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Appends a closure after all the current layers.
    pub fn add_fn<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor) -> Result<Tensor> + Send + Sync + Clone,
    {
        self.add(FuncImpl::new(f))
    }

    /// Applies the forward pass and returns the output for each layer.
    pub fn forward_all(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let mut vec = Vec::with_capacity(self.layers.len());
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?;
            vec.push(xs.clone())
        }
        Ok(vec)
    }
}

/// Trait object that can clone itself.
trait CloneableFn: Fn(&Tensor) -> Result<Tensor> + Send + Sync {
    fn clone_box(&self) -> Box<dyn CloneableFn>;
}

impl<F: Fn(&Tensor) -> Result<Tensor> + Send + Sync + Clone + 'static> CloneableFn for F {
    fn clone_box(&self) -> Box<dyn CloneableFn> {
        Box::new(self.clone())
    }
}


struct FuncImpl {
    f: Box<dyn CloneableFn>,
}

impl FuncImpl {
    fn new<F: Fn(&Tensor) -> Result<Tensor> + Send + Sync + Clone + 'static>(f: F) -> Self {
        Self { f: Box::new(f) }
    }
}

impl Module for FuncImpl {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        (self.f)(xs)
    }
}

impl fmt::Debug for FuncImpl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FuncImpl").finish()
    }
}

impl Clone for FuncImpl {
    fn clone(&self) -> Self {
        Self {
            f: self.f.clone_box(),
        }
    }
}