use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand::distributions::{Distribution, Uniform};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectionMatrix {
    pub matrix: Vec<Vec<f32>>,
    pub input_dim: usize,
    pub output_dim: usize,
    pub dropout_rate: f32,
}

impl ProjectionMatrix {
    pub fn new(input_dim: usize, output_dim: usize, seed: u64, orthogonal_init: bool) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut m = Vec::with_capacity(input_dim);
        
        
        if orthogonal_init {
           
            let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
            
            for _ in 0..input_dim {
                let mut row = Vec::with_capacity(output_dim);
                for _ in 0..output_dim {
                    row.push(normal.sample(&mut rng));
                }
                
                let magnitude = row.iter().map(|v| v * v).sum::<f32>().sqrt();
                if magnitude > 1e-8 {
                    for v in &mut row {
                        *v /= magnitude;
                    }
                }
                m.push(row);
            }
        } else {
            
            let scale = (6.0 / (input_dim + output_dim) as f32).sqrt();
            let uniform = Uniform::new(-scale, scale);
            
            for _ in 0..input_dim {
                let mut row = Vec::with_capacity(output_dim);
                for _ in 0..output_dim {
                    row.push(uniform.sample(&mut rng));
                }
                m.push(row);
            }
        }
        
        Self { 
            matrix: m, 
            input_dim, 
            output_dim,
            dropout_rate: 0.1, 
        }
    }

    pub fn project(&self, sparse: &HashMap<usize, f32>) -> Vec<f32> {
        let mut r = vec![0.0; self.output_dim];
        let mut rng = StdRng::from_entropy();
        
        for (&idx, &val) in sparse {
            if idx < self.input_dim {
                
                if rng.gen::<f32>() >= self.dropout_rate {
                    let scale = 1.0 / (1.0 - self.dropout_rate);
                    for (i, &proj) in self.matrix[idx].iter().enumerate() {
                        r[i] += val * proj * scale;
                    }
                }
            }
        }
        
        r
    }
    
    pub fn set_dropout_rate(&mut self, rate: f32) {
        self.dropout_rate = rate.max(0.0).min(0.9);
    }
}