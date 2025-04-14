use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vocabulary {
    pub word_to_index: HashMap<String, usize>,
    pub idf_values: Vec<f32>,
}

impl Vocabulary {
    pub fn new(word_to_index: HashMap<String, usize>, idf_values: Vec<f32>) -> Self {
        Self { word_to_index, idf_values }
    }

    pub fn size(&self) -> usize {
        self.word_to_index.len()
    }

    pub fn get_index(&self, w: &str) -> Option<usize> {
        self.word_to_index.get(w).copied()
    }

    pub fn get_idf(&self, w: &str) -> f32 {
        self.get_index(w).map(|i| self.idf_values[i]).unwrap_or(0.0)
    }
}
