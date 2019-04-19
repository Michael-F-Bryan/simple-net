//! Loss functions measure how good our predictions are.
//!
//! They are used to adjust the parameters of our network.

use crate::Tensor;

pub trait Loss {
    fn loss(&self, prediction: &Tensor, actual: &Tensor) -> f32;
    fn gradient(&self, prediction: &Tensor, actual: &Tensor) -> Tensor;
}

/// Mean squared error.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MeanSquared;

impl Loss for MeanSquared {
    fn loss(&self, prediction: &Tensor, actual: &Tensor) -> f32 {
        let difference = prediction - actual;
        let difference_squared = &difference * &difference;

        difference_squared.iter().map(|(_coord, value)| value).sum()
    }

    fn gradient(&self, prediction: &Tensor, actual: &Tensor) -> Tensor {
        let delta = prediction - actual;
        2.0 * &delta
    }
}
