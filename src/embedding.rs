use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul};

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
    
    pub fn distance(&self, other: &Embedding) -> f32 {
        if self.dimension() != other.dimension() {
            panic!("Different dimensions");
        }
        
        self.0.iter()
            .zip(other.0.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    
    pub fn top_k_similarity(&self, others: &[Embedding], k: usize) -> Vec<(usize, f32)> {
        let mut similarities: Vec<(usize, f32)> = others.iter()
            .enumerate()
            .map(|(i, other)| (i, self.cosine_similarity(other)))
            .collect();
            
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(k);
        similarities
    }
}

impl Add for &Embedding {
    type Output = Embedding;
    
    fn add(self, other: &Embedding) -> Embedding {
        if self.dimension() != other.dimension() {
            panic!("Cannot add embeddings of different dimensions");
        }
        
        let values = self.0.iter()
            .zip(other.0.iter())
            .map(|(&a, &b)| a + b)
            .collect();
            
        Embedding(values)
    }
}

impl Mul<f32> for &Embedding {
    type Output = Embedding;
    
    fn mul(self, scalar: f32) -> Embedding {
        let values = self.0.iter()
            .map(|&v| v * scalar)
            .collect();
            
        Embedding(values)
    }
}