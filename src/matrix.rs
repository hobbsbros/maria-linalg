//! Implements necessary methods on *square* matrices.

use std::{
    fmt,
    ops::{
        Add,
        Index,
        IndexMut,
        Sub,
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

    /// Returns a 3D rotation matrix representing a right-handed rotation about the
    ///     provided axis by the provided angle.
    /// 
    /// *Note*: the provided angle is in radians.
    pub fn rotation(
        axis: Vector<3>,
        angle: f64,
    ) -> Matrix<3> {
        let basis = Vector::<3>::basis();
        let mut r = [Vector::<3>::zero(); 3];

        for i in 0..3 {
            r[i] = basis[i].rotate(axis, angle);
        }

        Matrix::<3>::new([
            [r[0][0], r[1][0], r[2][0]],
            [r[0][1], r[1][1], r[2][1]],
            [r[0][2], r[1][2], r[2][2]],
        ])
    }

    /// Decomposes this matrix into its columns.
    /// 
    /// This is useful for determining the axes of a rotated coordinate system.
    pub fn decompose(&self) -> [Vector<N>; N] {
        let mut basis = [Vector::zero(); N];

        for i in 0..N {
            for j in 0..N {
                basis[j][i] = self[(i, j)];
            }
        }

        basis
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

    /// Right-multiplies this matrix by the provided matrix, returning the result.
    pub fn matmult(&self, matrix: Matrix<N>) -> Matrix<N> {
        let mut output = Matrix::<N>::zero();

        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    output[(i, j)] += self[(i, k)] * matrix[(k, j)];
                }
            }
        }

        output
    }

    /// Swap rows `i` and `j`.
    fn swaprow(&mut self, i: usize, j: usize) {
        let temp = self.values[i];
        self.values[i] = self.values[j];
        self.values[j] = temp;
    }

    /// Scale row `i` by factor `s`.
    fn scalerow(&mut self, i: usize, s: f64) {
        for j in 0..N {
            self[(i, j)] *= s;
        }
    }

    /// Subtract `s` times row `j` from row `i`.
    fn subrow(&mut self, i: usize, j: usize, s: f64) {
        for k in 0..N {
            self[(i, k)] -= s * self[(j, k)];
        }
    }

    /// Returns the inverse of this matrix.
    pub fn inverse(&self) -> Self {
        let mut output = *self;
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
            output.swaprow(i, j);
            inverse.swaprow(i, j);

            // Normalize this row
            let s = 1.0 / output[(i, i)];
            output.scalerow(i, s);
            inverse.scalerow(i, s);

            // Subtract this row from all lower rows
            for k in (i + 1)..N {
                let s = output[(k, i)];
                output.subrow(k, i, s);
                inverse.subrow(k, i, s);
            }
        }

        // We're now in upper triangular, let's get to GJ normal form

        for i in 0..N {
            for j in (i + 1)..N {
                let s = output[(i, j)];
                output.subrow(i, j, s);
                inverse.subrow(i, j, s);
            }
        }

        inverse
    }

    /// Scales a matrix by a provided scalar, returning the new matrix.
    pub fn scale(&self, scalar: f64) -> Self {
        let mut newvalues = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                newvalues[i][j] = scalar * self[(i, j)];
            }
        }

        Self {
            values: newvalues,
        }
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

impl<const N: usize> Sub for Matrix<N> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut new = Self::zero();
    
        for i in 0..N {
            for j in 0..N {
                new[(i, j)] = self[(i, j)] - other[(i, j)];   
            }
        }

        new
    }
}

impl<const N: usize> fmt::Display for Matrix<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut values = Vec::new();
        let mut maxlen = 0;
        for i in 0..N {
            for j in 0..N {
                let value = self[(i, j)];
                let row = if value >= 0.0 {
                    format!(" {:.8}", value)
                } else {
                    format!("{:.8}", value)
                };
                let l = row.len();
                values.push(row);
                if l > maxlen {
                    maxlen = l;
                }
            }
        }

        let mut output = String::new();
        for i in 0..N {
            output.push_str("[");
            for j in 0..N {
                output.push_str(
                    &format!("{:^i$}", values[j + N*i], i = maxlen + 2)
                );
            }
            output.push_str("]\n");
        }

        write!(f, "{}", output)
    }
}

#[test]
fn matrix_multiply() {
    let a = Matrix::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]);

    let b = Matrix::new([
        [9.0, 8.0, 7.0],
        [6.0, 5.0, 4.0],
        [3.0, 2.0, 1.0],
    ]);

    let c = Matrix::new([
        [ 30.0,  24.0,  18.0],
        [ 84.0,  69.0,  54.0],
        [138.0, 114.0,  90.0],
    ]);

    println!("{}", c);

    assert_eq!(a.matmult(b), c);
}

#[test]
fn decompose() {
    let a = Matrix::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]);

    let basis: [Vector<3>; 3] = [
        [1.0, 4.0, 7.0].into(),
        [2.0, 5.0, 8.0].into(),
        [3.0, 6.0, 9.0].into(),
    ];

    assert_eq!(a.decompose(), basis);
}

#[test]
fn z_rotation_matrix() {
    let axis = [0.0, 0.0, 1.0].into();

    let rotation = Matrix::<3>::rotation(axis, 30.0 * 3.141592653 / 180.0);

    println!("{}", rotation);
}

#[test]
fn decomposition() {
    let basis = Vector::<3>::basis();

    println!("{:#?}", basis);
}