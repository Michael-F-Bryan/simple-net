use crate::{Layer, Tensor};
use std::iter::FromIterator;

pub struct NeuralNet {
    layers: Vec<Box<dyn Layer>>,
}

impl NeuralNet {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> NeuralNet {
        NeuralNet { layers }
    }

    pub fn forward(&self, inputs: Tensor) -> (Tensor, Vec<Tensor>) {
        let mut intermediate_results = Vec::new();
        intermediate_results.push(inputs);

        for layer in &self.layers {
            let last = intermediate_results.last().unwrap();
            let intermediate = layer.forward(last);
            intermediate_results.push(intermediate);
        }

        let result = intermediate_results.last().unwrap().clone();
        (result, intermediate_results)
    }

    pub fn backward(
        &mut self,
        mut grad: Tensor,
        intermediate_results: &[Tensor],
    ) -> Tensor {
        assert_eq!(self.layers.len(), intermediate_results.len());

        for (layer, intermediate) in self
            .layers
            .iter_mut()
            .zip(intermediate_results.into_iter())
            .rev()
        {
            grad = layer.backward(&grad, intermediate);
        }

        grad
    }
}

impl FromIterator<Box<dyn Layer>> for NeuralNet {
    fn from_iter<T: IntoIterator<Item = Box<dyn Layer>>>(iter: T) -> NeuralNet {
        NeuralNet::new(iter.into_iter().collect())
    }
}
