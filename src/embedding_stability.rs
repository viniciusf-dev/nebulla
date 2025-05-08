use crate::{NebulaEmbeddings, Embedding};
use rand::prelude::*;

pub struct StabilityMetrics {
    pub average_similarity: f32,
    pub min_similarity: f32,
    pub max_similarity: f32,
    pub standard_deviation: f32,
}

pub fn generate_variant(text: &str, mutation_rate: f32) -> String {
    let mut rng = rand::thread_rng();
    
   
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }
    
    let mut result = Vec::new();
    
    for word in &words {
        if rng.gen::<f32>() < mutation_rate {
           
            let mutation_type = rng.gen_range(0..4);
            match mutation_type {
                0 => {
                    
                    continue;
                },
                1 => {
                    
                    result.push(word.to_string());
                    result.push(word.to_string());
                },
                2 => {
                   
                    if !result.is_empty() {
                        let prev = result.pop().unwrap();
                        result.push(word.to_string());
                        result.push(prev);
                    } else {
                        result.push(word.to_string());
                    }
                },
                3 => {
                    
                    let mut chars: Vec<char> = word.chars().collect();
                    if chars.len() > 2 {
                        let idx = rng.gen_range(0..chars.len());
                        chars.remove(idx);
                    }
                    result.push(chars.iter().collect());
                },
                _ => result.push(word.to_string()),
            }
        } else {
            result.push(word.to_string());
        }
    }
    
    result.join(" ")
}

pub fn evaluate_stability(
    model: &NebulaEmbeddings,
    texts: &[String],
    num_variants: usize,
    mutation_rate: f32,
) -> StabilityMetrics {
    let mut similarities = Vec::new();
    
    for text in texts {
        let original_embedding = model.embed(text);
        
        for _ in 0..num_variants {
            let variant = generate_variant(text, mutation_rate);
            let variant_embedding = model.embed(&variant);
            
            let similarity = original_embedding.cosine_similarity(&variant_embedding);
            similarities.push(similarity);
        }
    }
    
    if similarities.is_empty() {
        return StabilityMetrics {
            average_similarity: 0.0,
            min_similarity: 0.0,
            max_similarity: 0.0,
            standard_deviation: 0.0,
        };
    }
    
    let avg = similarities.iter().sum::<f32>() / similarities.len() as f32;
    let min = *similarities.iter().fold(&1.0, |a, b| if b < a { b } else { a });
    let max = *similarities.iter().fold(&0.0, |a, b| if b > a { b } else { a });
    
    let variance = similarities.iter()
        .map(|&s| (s - avg).powi(2))
        .sum::<f32>() / similarities.len() as f32;
    
    let std_dev = variance.sqrt();
    
    StabilityMetrics {
        average_similarity: avg,
        min_similarity: min,
        max_similarity: max,
        standard_deviation: std_dev,
    }
}