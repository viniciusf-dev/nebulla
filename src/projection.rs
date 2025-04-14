use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectionMatrix {
    pub matrix: Vec<Vec<f32>>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl ProjectionMatrix {
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut m = Vec::with_capacity(input_dim);
        for _ in 0..input_dim {
            let mut row = Vec::with_capacity(output_dim);
            for _ in 0..output_dim {
                row.push(rng.gen::<f32>() * 2.0 - 1.0);
            }
            m.push(row);
        }
        Self { matrix: m, input_dim, output_dim }
    }

    pub fn project(&self, sparse: &HashMap<usize, f32>) -> Vec<f32> {
        let mut r = vec![0.0; self.output_dim];
        for (&idx, &val) in sparse {
            if idx < self.input_dim {
                for (i, &proj) in self.matrix[idx].iter().enumerate() {
                    r[i] += val * proj;
                }
            }
        }
        let mag = r.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if mag > 1e-10 {
            for v in &mut r {
                *v /= mag;
            }
        }
        r
    }
}
