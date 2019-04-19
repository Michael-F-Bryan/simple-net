use crate::Tensor;

/// The individual components which make up a neural network.
///
/// Each layer needs to pass its inputs forward and propagate gradients
/// backwards.
pub trait Layer {
    /// Produce the outputs corresponding to these inputs.
    fn forward(&self, inputs: &Tensor) -> Tensor;
    /// Back-propagate the gradient through this layer.
    fn backward(
        &mut self,
        original_inputs: &Tensor,
        gradient: &Tensor,
    ) -> Tensor;
}

/// Computes `output = inputs * weights + bias`.
#[derive(Debug, Clone, PartialEq)]
pub struct Linear {
    weights: Tensor,
    bias: Tensor,
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize) -> Linear {
        // inputs are (batch_size, input_size)
        // outputs are (batch_size, output_size)
        Linear {
            weights: Tensor::zero(input_size, output_size),
            bias: Tensor::zero(1, output_size),
        }
    }
}

impl Layer for Linear {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        inputs * &self.weights + &self.bias
    }

    fn backward(
        &mut self,
        _original_inputs: &Tensor,
        gradient: &Tensor,
    ) -> Tensor {
        // if y = f(x) and x = a*b + c
        // then
        //   dy/da = f'(x) * b.T
        //   dy/db = a.T * f'(x)
        //   dy/dc = f'(x)

        // TODO: we need these, but it doesn't look like they're ever used...
        // let bias_gradient =
        //     Tensor::column(gradient.rows().map(|row| row.into_iter().sum()));
        // let weight_gradient = &original_inputs.transposed() * gradient;

        gradient * &self.weights.transposed()
    }
}

/// A layer which applies the provided function element-wise to the provided
/// input.
pub struct Activation<Func, Derivative> {
    func: Func,
    derivative: Derivative,
}

impl<Func, Derivative> Activation<Func, Derivative>
where
    Func: Fn(&Tensor) -> Tensor,
    Derivative: Fn(&Tensor) -> Tensor,
{
    pub fn new(
        func: Func,
        derivative: Derivative,
    ) -> Activation<Func, Derivative> {
        Activation { func, derivative }
    }
}

impl Activation<fn(&Tensor) -> Tensor, fn(&Tensor) -> Tensor> {
    pub fn tanh() -> Self {
        Activation::new(
            |t| t.mapped(|n| n.tanh()),
            |t| t.mapped(|n| 1.0 - n.tanh() * n.tanh()),
        )
    }
}

impl<Func, Derivative> Layer for Activation<Func, Derivative>
where
    Func: Fn(&Tensor) -> Tensor,
    Derivative: Fn(&Tensor) -> Tensor,
{
    fn forward(&self, inputs: &Tensor) -> Tensor {
        (self.func)(inputs)
    }

    fn backward(
        &mut self,
        original_inputs: &Tensor,
        gradient: &Tensor,
    ) -> Tensor {
        &(self.derivative)(original_inputs) * gradient
    }
}
