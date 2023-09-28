//! Implements necessary methods on *square* matrices.

use std::{
    ops::{
        Add,
        Index,
        IndexMut,
    },
};

use super::Vector;

#[derive(Clone, Copy, PartialEq, Debug)]
/// Abstracts over a square matrix of arbitrary dimension.
pub struct Matrix<const N: usize> {
    /// Contains the values of this matrix.
    values: [[f64; N]; N],
}

/// Implements necessary behaviors of a matrix.
impl<const N: usize> Matrix<N> {
    /// Constructs a zero matrix.
    pub fn zero() -> Self {
        Self {
            values: [[0.0; N]; N],
        }
    }

    /// Constructs an identity matrix.
    pub fn identity() -> Self {
        let mut values = [[0.0; N]; N];

        for i in 0..N {
            values[i][i] = 1.0;
        }

        Self {
            values
        }
    }

    /// Constructs a matrix of provided values.
    pub fn new(values: [[f64; N]; N]) -> Self {
        Self {
            values,
        }
    }

    /// Right-multiplies this matrix by the provided vector, returning the result.
    pub fn mult(&self, vector: Vector<N>) -> Vector<N> {
        let mut output = Vector::<N>::zero();

        for i in 0..N {
            for j in 0..N {
                output[i] += self[(i, j)] * vector[j];
            }
        }

        output
    }

    /// Swap rows `i` and `j`.
    fn swap(&mut self, i: usize, j: usize) {
        let temp = self.values[i];
        self.values[i] = self.values[j];
        self.values[j] = temp;
    }

    /// Scale row `i` by factor `s`.
    fn scale(&mut self, i: usize, s: f64) {
        for j in 0..N {
            self[(i, j)] *= s;
        }
    }

    /// Subtract `s` times row `j` from row `i`.
    fn sub(&mut self, i: usize, j: usize, s: f64) {
        for k in 0..N {
            self[(i, k)] -= s * self[(j, k)];
        }
    }

    /// Returns the inverse of this matrix.
    pub fn inverse(&self) -> Self {
        let mut output = self.clone();
        let mut inverse = Self::identity();

        for i in 0..N {
            // Determine the index of the row with the largest pivot
            // Start from the working row
            let mut j = i;
            for k in i..N {
                if output[(k, i)] > output[(i, i)] {
                    j = k;
                }
            }

            // Swap largest pivot to working row
            output.swap(i, j);
            inverse.swap(i, j);

            // Normalize this row
            let s = 1.0 / output[(i, i)];
            output.scale(i, s);
            inverse.scale(i, s);

            // Subtract this row from all lower rows
            for k in (i + 1)..N {
                let s = output[(k, i)];
                output.sub(k, i, s);
                inverse.sub(k, i, s);
            }
        }

        // We're now in upper triangular, let's get to GJ normal form

        for i in 0..N {
            for j in (i + 1)..N {
                let s = output[(i, j)];
                output.sub(i, j, s);
                inverse.sub(i, j, s);
            }
        }

        inverse
    }
}

impl<const N: usize> Index<(usize, usize)> for Matrix<N> {
    type Output = f64;

    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        &self.values[idx.0][idx.1]
    }
}

impl<const N: usize> IndexMut<(usize, usize)> for Matrix<N> {
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut Self::Output {
        &mut self.values[idx.0][idx.1]
    }
}

impl<const N: usize> Add for Matrix<N> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut new = Self::zero();
    
        for i in 0..N {
            for j in 0..N {
                new[(i, j)] = self[(i, j)] + other[(i, j)];   
            }
        }

        new
    }
}