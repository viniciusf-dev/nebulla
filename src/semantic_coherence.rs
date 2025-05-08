use std::collections::HashMap;
use crate::{NebulaEmbeddings, Embedding};

pub struct SemanticPair {
    pub word1: String,
    pub word2: String,
    pub related: bool,
}

pub struct SemanticCoherenceResult {
    pub accuracy: f32,
    pub average_related_similarity: f32,
    pub average_unrelated_similarity: f32,
    pub contrast_score: f32,
}

pub fn evaluate_semantic_coherence(
    model: &NebulaEmbeddings,
    pairs: &[SemanticPair],
) -> SemanticCoherenceResult {
    let mut related_similarities = Vec::new();
    let mut unrelated_similarities = Vec::new();
    let mut correct_predictions = 0;
    
    for pair in pairs {
        let similarity = model.similarity(&pair.word1, &pair.word2);
        
        if pair.related {
            related_similarities.push(similarity);
        } else {
            unrelated_similarities.push(similarity);
        }
        
        let predicted_related = similarity > 0.5;
        if predicted_related == pair.related {
            correct_predictions += 1;
        }
    }
    
    let avg_related = if !related_similarities.is_empty() {
        related_similarities.iter().sum::<f32>() / related_similarities.len() as f32
    } else {
        0.0
    };
    
    let avg_unrelated = if !unrelated_similarities.is_empty() {
        unrelated_similarities.iter().sum::<f32>() / unrelated_similarities.len() as f32
    } else {
        0.0
    };
    
    let accuracy = if !pairs.is_empty() {
        correct_predictions as f32 / pairs.len() as f32
    } else {
        0.0
    };
    
    let contrast_score = avg_related - avg_unrelated;
    
    SemanticCoherenceResult {
        accuracy,
        average_related_similarity: avg_related,
        average_unrelated_similarity: avg_unrelated,
        contrast_score,
    }
}

pub fn generate_test_pairs() -> Vec<SemanticPair> {
    vec![
        SemanticPair {
            word1: "dog".to_string(),
            word2: "cat".to_string(),
            related: true,
        },
        SemanticPair {
            word1: "happy".to_string(),
            word2: "sad".to_string(),
            related: true,
        },
        SemanticPair {
            word1: "big".to_string(),
            word2: "large".to_string(),
            related: true,
        },
        SemanticPair {
            word1: "computer".to_string(),
            word2: "laptop".to_string(),
            related: true,
        },
        SemanticPair {
            word1: "run".to_string(),
            word2: "sprint".to_string(),
            related: true,
        },
        SemanticPair {
            word1: "dog".to_string(),
            word2: "algorithm".to_string(),
            related: false,
        },
        SemanticPair {
            word1: "happy".to_string(),
            word2: "computer".to_string(),
            related: false,
        },
        SemanticPair {
            word1: "big".to_string(),
            word2: "quickly".to_string(),
            related: false,
        },
        SemanticPair {
            word1: "laptop".to_string(),
            word2: "elephant".to_string(),
            related: false,
        },
        SemanticPair {
            word1: "sprint".to_string(),
            word2: "keyboard".to_string(),
            related: false,
        },
    ]
}