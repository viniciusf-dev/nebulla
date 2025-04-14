use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Embedding(pub Vec<f32>);

impl Embedding {
    pub fn new(values: Vec<f32>) -> Self {
        Self(values)
    }

    pub fn values(&self) -> &[f32] {
        &self.0
    }

    pub fn dimension(&self) -> usize {
        self.0.len()
    }

    pub fn normalize(&self) -> Self {
        let m = self.0.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if m > 1e-10 {
            let n = self.0.iter().map(|&x| x / m).collect();
            Self(n)
        } else {
            self.clone()
        }
    }

    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        if self.dimension() != other.dimension() {
            panic!("Different dimensions");
        }
        let dot = self.0.iter().zip(other.0.iter()).map(|(&a, &b)| a * b).sum::<f32>();
        let ma = self.0.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let mb = other.0.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if ma > 1e-10 && mb > 1e-10 {
            dot / (ma * mb)
        } else {
            0.0
        }
    }
}
