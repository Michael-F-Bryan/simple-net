use std::ops::{Add, Mul, Sub};

/// A row-major 2D matrix of floats.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    width: usize,
    height: usize,
}

impl Tensor {
    pub fn zero(width: usize, height: usize) -> Tensor {
        Tensor {
            data: vec![0.0; width * height],
            width,
            height,
        }
    }

    pub fn fill(
        width: usize,
        height: usize,
        mut values: impl FnMut(usize, usize) -> f32,
    ) -> Tensor {
        let size = width * height;
        let mut t = Tensor {
            data: Vec::with_capacity(size),
            width,
            height,
        };

        for i in 0..size {
            let (row, col) = t.get_location(i);
            t.data.push(values(row, col));
        }

        t
    }

    pub fn get(&self, row: usize, column: usize) -> Option<f32> {
        if row >= self.height || column >= self.width {
            None
        } else {
            let ix = self.get_ix(row, column);
            Some(self.data[ix])
        }
    }

    pub fn cells(&self) -> impl Iterator<Item = (usize, usize)> {
        let width = self.width;
        (0..self.height)
            .flat_map(move |row| (0..width).map(move |col| (row, col)))
    }

    pub fn iter<'this>(
        &'this self,
    ) -> impl Iterator<Item = ((usize, usize), f32)> + 'this {
        self.cells().map(move |loc| {
            (loc, self.get(loc.0, loc.1).expect("Always within bounds"))
        })
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn get_ix(&self, row: usize, column: usize) -> usize {
        row * self.width + column
    }

    fn get_location(&self, index: usize) -> (usize, usize) {
        let column = index % self.width;
        let row = index / self.width;
        (row, column)
    }
}

impl<'a> Add for &'a Tensor {
    type Output = Tensor;

    fn add(self, other: &'a Tensor) -> Self::Output {
        assert_eq!(self.dimensions(), other.dimensions());
        assert_eq!(self.data.len(), other.data.len());

        let mut result = self.clone();
        for i in 0..self.data.len() {
            result.data[i] += other.data[i];
        }

        result
    }
}

impl<'a> Sub for &'a Tensor {
    type Output = Tensor;

    fn sub(self, other: &'a Tensor) -> Self::Output {
        assert_eq!(self.dimensions(), other.dimensions());
        assert_eq!(self.data.len(), other.data.len());

        let mut result = self.clone();
        for i in 0..self.data.len() {
            result.data[i] += other.data[i];
        }

        result
    }
}

impl<'a> Mul for &'a Tensor {
    type Output = Tensor;

    fn mul(self, other: &'a Tensor) -> Self::Output {
        unimplemented!()
    }
}

impl<'a> Mul<f32> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, other: f32) -> Self::Output {
        let mut c = self.clone();
        for item in &mut c.data {
            *item *= other;
        }

        c
    }
}

impl<'a> Mul<&'a Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, other: &'a Tensor) -> Self::Output {
        other * self
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Self::Output {
        &self * &other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_are_actually_row_major() {
        let tensor = Tensor::zero(100, 100);
        let inputs = vec![
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, tensor.width),
            (50, 30, 50 * tensor.width + 30),
        ];

        for (row, column, expected) in inputs {
            let got = tensor.get_ix(row, column);
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn add_two_tensors() {
        let left = Tensor::fill(50, 100, |row, col| (row * 50 + col) as f32);
        let right = Tensor::fill(50, 100, |row, col| (row * 50 + col) as f32);
        let expected =
            Tensor::fill(50, 100, |row, col| (row * 50 + col) as f32 * 2.0);

        let got = &left + &right;

        assert_eq!(got, expected);
    }

    #[test]
    fn multiply_by_scale_factor() {
        let t = Tensor::fill(50, 100, |row, col| (row * 50 + col) as f32);
        let factor = 123.4;
        let expected =
            Tensor::fill(50, 100, |row, col| (row * 50 + col) as f32 * factor);

        let got = factor * &t;

        assert_eq!(got, expected);
    }
}
