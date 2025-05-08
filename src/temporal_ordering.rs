use crate::{NebulaEmbeddings, Embedding};
use std::collections::HashMap;

pub struct TemporalOrderingMetrics {
    pub kendall_tau: f32,
    pub correlation_coefficient: f32,
    pub average_distance: f32,
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}

pub fn evaluate_temporal_ordering(
    model: &NebulaEmbeddings,
    document_sequences: &[Vec<String>],
) -> TemporalOrderingMetrics {
    let mut total_kendall_tau = 0.0;
    let mut total_correlation = 0.0;
    let mut total_avg_distance = 0.0;
    
    for sequence in document_sequences {
        if sequence.len() < 2 {
            continue;
        }
        
        let embeddings: Vec<Embedding> = sequence.iter()
            .map(|doc| model.embed(doc))
            .collect();
        
        let mut consecutive_similarities = Vec::new();
        for i in 0..embeddings.len() - 1 {
            let similarity = cosine_similarity(
                embeddings[i].values(),
                embeddings[i + 1].values(),
            );
            consecutive_similarities.push(similarity);
        }
        
        let mut non_consecutive_similarities = Vec::new();
        for i in 0..embeddings.len() {
            for j in i + 2..embeddings.len() {
                let similarity = cosine_similarity(
                    embeddings[i].values(),
                    embeddings[j].values(),
                );
                non_consecutive_similarities.push(similarity);
            }
        }
        
        let avg_consecutive_sim = if !consecutive_similarities.is_empty() {
            consecutive_similarities.iter().sum::<f32>() / consecutive_similarities.len() as f32
        } else {
            0.0
        };
        
        let avg_non_consecutive_sim = if !non_consecutive_similarities.is_empty() {
            non_consecutive_similarities.iter().sum::<f32>() / non_consecutive_similarities.len() as f32
        } else {
            0.0
        };
        

        let mut concordant = 0;
        let mut discordant = 0;
        
        for &consec_sim in &consecutive_similarities {
            for &non_consec_sim in &non_consecutive_similarities {
                if consec_sim > non_consec_sim {
                    concordant += 1;
                } else if consec_sim < non_consec_sim {
                    discordant += 1;
                }
            }
        }
        
        let kendall_tau = if concordant + discordant > 0 {
            (concordant as f32 - discordant as f32) / (concordant + discordant) as f32
        } else {
            0.0
        };
        
        let correlation = avg_consecutive_sim - avg_non_consecutive_sim;
        
        let mut avg_distance = 0.0;
        for i in 0..embeddings.len() - 1 {
            avg_distance += embeddings[i].distance(&embeddings[i + 1]);
        }
        avg_distance /= (embeddings.len() - 1) as f32;
        
        total_kendall_tau += kendall_tau;
        total_correlation += correlation;
        total_avg_distance += avg_distance;
    }
    
    let num_sequences = document_sequences.len() as f32;
    
    TemporalOrderingMetrics {
        kendall_tau: total_kendall_tau / num_sequences,
        correlation_coefficient: total_correlation / num_sequences,
        average_distance: total_avg_distance / num_sequences,
    }
}

pub fn generate_test_sequences() -> Vec<Vec<String>> {
    vec![
        vec![
            "Introduction to machine learning".to_string(),
            "Linear regression and gradient descent".to_string(),
            "Neural networks and backpropagation".to_string(),
            "Convolutional neural networks for image processing".to_string(),
            "Transformers and attention mechanisms".to_string(),
        ],
        vec![
            "The beginning of the novel".to_string(),
            "Character development and plot progression".to_string(),
            "Conflict escalation and challenges".to_string(),
            "Resolution and climax of the story".to_string(),
            "Epilogue and aftermath".to_string(),
        ],
        vec![
            "Introduction to programming basics".to_string(),
            "Variables, data types, and control structures".to_string(),
            "Functions and object-oriented programming".to_string(),
            "Data structures and algorithms".to_string(),
            "Advanced topics and design patterns".to_string(),
        ],
    ]
}