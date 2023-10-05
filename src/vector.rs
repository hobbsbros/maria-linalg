//! Implements necessary methods on vectors.

use std::{
    fmt,
    ops::{
        Add,
        Sub,
        Index,
        IndexMut,
    },
};

use rand::{
    random,
    thread_rng,
};

use rand_distr::{
    Distribution,
    Normal,
};

use super::Matrix;

#[derive(Clone, Copy, PartialEq, Debug)]
/// Abstracts over a vector of arbitary dimension.
pub struct Vector<const N: usize> {
    /// Contains the values of this vector.
    values: [f64; N],
}

/// Implements necessary behaviors of a vector.
impl<const N: usize> Vector<N> {
    /// Constructs a zero vector.
    pub fn zero() -> Self {
        Self {
            values: [0.0; N],
        }
    }

    /// Constructs a vector of provided values.
    pub fn new(values: [f64; N]) -> Self {
        Self {
            values,
        }
    }

    /// Constructs an orthogonal basis of vectors.
    pub fn basis() -> [Self; N] {
        let mut basis = [Self::zero(); N];

        for i in 0..N {
            basis[i][i] = 1.0;
        }

        basis
    }

    /// Scales a vector by a provided scalar, returning the new vector.
    pub fn scale(&self, scalar: f64) -> Self {
        let mut newvalues = [0.0; N];
        for i in 0..N {
            newvalues[i] = scalar * self[i];
        }

        Self {
            values: newvalues,
        }
    }
    
    /// Dots this vector with another vector.
    pub fn dot(&self, other: Self) -> f64 {
        let mut output = 0.0;

        for i in 0..N {
            output += self[i] * other[i];
        }

        output
    }

    /// Crosses this vector with another vector.
    /// 
    /// *Note*: this is only implemented for `Vector<3>`.
    /// Otherwise, this returns a zero vector.
    pub fn cross(&self, other: Self) -> Self {
        let mut output = Self::zero();

        if N == 3 {
            output[0] = self[1] * other[2] - self[2] * other[1];
            output[1] = self[2] * other[0] - self[0] * other[2];
            output[2] = self[0] * other[1] - self[1] * other[0];
        }

        output
    }

    /// Takes the norm of a vector.
    pub fn norm(&self) -> f64 {
        let mut output = 0.0;

        for i in 0..N {
            output += self[i].powf(2.0);
        }

        output.sqrt()
    }

    /// Left-multiplies the provided matrix by the transpose of this vector, returning the result.
    pub fn mult(&self, matrix: Matrix<N>) -> Self {
        let mut output = Self::zero();

        for i in 0..N {
            for j in 0..N {
                output[i] += matrix[(i, j)] * self[j];
            }
        }

        output
    }

    /// Given two vectors, generate a "child" vector.
    /// This function is useful for genetic optimization algorithms.
    pub fn child(mother: &Self, father: &Self, stdev: f64) -> Self {
        let mut child = Self::zero();

        for i in 0..N {
            // Select gene for child
            child[i] = if random::<f64>() < 0.5 {
                mother[i]
            } else {
                father[i]
            };

            // Mutate this gene
            // NOTE: it's ok to use `unwrap` here because we
            // know that we will always be able to create a normal
            // distribution of type N(0, `stdev`)
            let normal = Normal::new(0.0, stdev).unwrap();
            let v = normal.sample(&mut thread_rng());
            child[i] += v;
        }

        child
    }

    /// Determines if this vector is within the element-wise contraints.
    pub fn check(&self, lower: [Option<f64>; N], upper: [Option<f64>; N]) -> bool {
        for i in 0..N {
            if let Some (l) = lower[i] {
                if self[i] < l {
                    return false;
                }
            } else if let Some (u) = upper[i] {
                if self[i] > u {
                    return false;
                }
            }
        }

        true
    }
}

impl<const N: usize> From<[f64; N]> for Vector<N> {
    fn from(values: [f64; N]) -> Self {
        Self::new(values)
    }
}

impl<const N: usize> Index<usize> for Vector<N> {
    type Output = f64;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.values[idx]
    }
}

impl<const N: usize> IndexMut<usize> for Vector<N> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.values[idx]
    }
}

impl<const N: usize> Add for Vector<N> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut new = Self::zero();
        
        for i in 0..self.values.len() {
            new[i] = self[i] + other[i];
        }

        new
    }
}

impl<const N: usize> Sub for Vector<N> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut new = Self::zero();
        
        for i in 0..N {
            new[i] = self[i] - other[i];
        }

        new
    }
}

impl<const N: usize> fmt::Display for Vector<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut rows = Vec::new();
        let mut maxlen = 0;
        for i in 0..N {
            let row = format!("{:.8}", self[i]);
            let l = row.len();
            rows.push(row);
            if l > maxlen {
                maxlen = l;
            }
        }

        let mut output = String::new();
        for i in 0..N {
            output.push_str("[");
            output.push_str(
                &format!("{:^i$}", rows[i], i = maxlen + 2)
            );
            output.push_str("]\n");
        }

        write!(f, "{}", output)
    }
}