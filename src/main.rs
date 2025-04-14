// Nebula Embeddings: A lightweight, high-performance text embedding model in Rust
//
// This implementation provides a minimal but functional embedding model that:
// - Uses a bag-of-words approach with TF-IDF weighting
// - Applies dimensionality reduction via random projections
// - Processes text through tokenization and normalization
// - Offers an immutable API design with thread safety
// - Includes cosine similarity calculation for comparing embeddings

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// Core data structures
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Embedding(Vec<f32>);

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Vocabulary {
    word_to_index: HashMap<String, usize>,
    idf_values: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProjectionMatrix {
    matrix: Vec<Vec<f32>>,
    input_dim: usize,
    output_dim: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NebulaModel {
    vocabulary: Vocabulary,
    projection: ProjectionMatrix,
    config: ModelConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    embedding_dim: usize,
    min_token_frequency: usize,
    max_vocabulary_size: usize,
    random_seed: u64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            min_token_frequency: 5,
            max_vocabulary_size: 20000,
            random_seed: 42,
        }
    }
}

// Builder pattern for configuration
pub struct ModelConfigBuilder {
    config: ModelConfig,
}

impl ModelConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: ModelConfig::default(),
        }
    }

    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.config.embedding_dim = dim;
        self
    }

    pub fn min_token_frequency(mut self, freq: usize) -> Self {
        self.config.min_token_frequency = freq;
        self
    }

    pub fn max_vocabulary_size(mut self, size: usize) -> Self {
        self.config.max_vocabulary_size = size;
        self
    }

    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = seed;
        self
    }

    pub fn build(self) -> ModelConfig {
        self.config
    }
}

// Implementation for the Embedding type
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

    // Normalize the embedding to unit length
    pub fn normalize(&self) -> Self {
        let magnitude: f32 = self.0.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if magnitude > 1e-10 {
            let normalized = self.0.iter().map(|&x| x / magnitude).collect();
            Self(normalized)
        } else {
            self.clone()
        }
    }

    // Calculate cosine similarity between two embeddings
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        if self.dimension() != other.dimension() {
            panic!("Cannot compare embeddings of different dimensions");
        }

        let dot_product: f32 = self.0.iter()
            .zip(other.0.iter())
            .map(|(&a, &b)| a * b)
            .sum();

        let mag_a: f32 = self.0.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = other.0.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if mag_a > 1e-10 && mag_b > 1e-10 {
            dot_product / (mag_a * mag_b)
        } else {
            0.0
        }
    }
}

// Vocabulary implementation
impl Vocabulary {
    fn new(word_to_index: HashMap<String, usize>, idf_values: Vec<f32>) -> Self {
        Self {
            word_to_index,
            idf_values,
        }
    }

    fn size(&self) -> usize {
        self.word_to_index.len()
    }

    fn contains(&self, word: &str) -> bool {
        self.word_to_index.contains_key(word)
    }

    fn get_index(&self, word: &str) -> Option<usize> {
        self.word_to_index.get(word).copied()
    }

    fn get_idf(&self, word: &str) -> f32 {
        if let Some(idx) = self.get_index(word) {
            self.idf_values[idx]
        } else {
            0.0
        }
    }
}

// Projection matrix implementation
impl ProjectionMatrix {
    fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut matrix = Vec::with_capacity(input_dim);

        for _ in 0..input_dim {
            let mut row = Vec::with_capacity(output_dim);
            for _ in 0..output_dim {
                // Generate values from a normal distribution
                let val = rng.gen::<f32>() * 2.0 - 1.0;
                row.push(val);
            }
            matrix.push(row);
        }

        Self {
            matrix,
            input_dim,
            output_dim,
        }
    }

    fn project(&self, sparse_vector: &HashMap<usize, f32>) -> Vec<f32> {
        let mut result = vec![0.0; self.output_dim];

        for (&word_idx, &word_weight) in sparse_vector.iter() {
            if word_idx < self.input_dim {
                for (dim, &proj_value) in self.matrix[word_idx].iter().enumerate() {
                    result[dim] += word_weight * proj_value;
                }
            }
        }

        // Normalize result
        let magnitude: f32 = result.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if magnitude > 1e-10 {
            for value in &mut result {
                *value /= magnitude;
            }
        }

        result
    }
}

// Text preprocessing utilities
mod preprocessing {
    use std::collections::HashSet;
    use unicode_normalization::UnicodeNormalization;

    lazy_static::lazy_static! {
        static ref STOP_WORDS: HashSet<&'static str> = {
            let words = vec![
                "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "to", "from",
                "by", "with", "in", "out", "is", "are", "am", "was", "were", "be", "being", "been",
                "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should",
                "may", "might", "must", "can", "could", "of", "this", "that", "these", "those",
                "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
                "its", "our", "their", "what", "which", "who", "whom", "whose", "when", "where",
                "why", "how"
            ];
            words.into_iter().collect()
        };
    }

    pub fn tokenize(text: &str) -> Vec<String> {
        text.unicode_normalization()
            .nfc()
            .collect::<String>()
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() >= 2 && !STOP_WORDS.contains(s))
            .map(|s| s.to_string())
            .collect()
    }
}

// Model implementation
impl NebulaModel {
    // Build a new model from a collection of texts
    pub fn train<I, S>(texts: I, config: ModelConfig) -> Result<Self, io::Error>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        // Count word frequencies and document frequencies
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        let mut doc_counts: HashMap<String, usize> = HashMap::new();
        let mut document_count = 0;

        for text in texts {
            document_count += 1;
            let tokens = preprocessing::tokenize(text.as_ref());
            let unique_tokens: HashSet<_> = tokens.iter().cloned().collect();

            for token in tokens {
                *word_counts.entry(token).or_insert(0) += 1;
            }

            for token in unique_tokens {
                *doc_counts.entry(token).or_insert(0) += 1;
            }
        }

        // Filter vocabulary based on frequency
        let mut vocab_words: Vec<(String, usize)> = word_counts
            .into_iter()
            .filter(|(_, count)| *count >= config.min_token_frequency)
            .collect();

        // Sort by frequency (descending)
        vocab_words.sort_by(|a, b| b.1.cmp(&a.1));

        // Limit vocabulary size
        if vocab_words.len() > config.max_vocabulary_size {
            vocab_words.truncate(config.max_vocabulary_size);
        }

        // Create word-to-index mapping and IDF values
        let mut word_to_index = HashMap::new();
        let mut idf_values = Vec::new();

        for (i, (word, _)) in vocab_words.iter().enumerate() {
            word_to_index.insert(word.clone(), i);
            let doc_freq = doc_counts.get(word).copied().unwrap_or(0);
            let idf = (document_count as f32 / (doc_freq as f32 + 1.0)).ln() + 1.0;
            idf_values.push(idf);
        }

        // Create vocabulary
        let vocabulary = Vocabulary::new(word_to_index, idf_values);

        // Create projection matrix
        let projection = ProjectionMatrix::new(
            vocabulary.size(),
            config.embedding_dim,
            config.random_seed,
        );

        Ok(Self {
            vocabulary,
            projection,
            config,
        })
    }

    // Generate an embedding for a text
    pub fn embed(&self, text: &str) -> Embedding {
        let tokens = preprocessing::tokenize(text);
        let mut tf_values: HashMap<usize, f32> = HashMap::new();
        
        // Calculate term frequencies
        for token in tokens {
            if let Some(idx) = self.vocabulary.get_index(&token) {
                *tf_values.entry(idx).or_insert(0.0) += 1.0;
            }
        }
        
        // Apply TF-IDF weighting
        for (idx, tf) in tf_values.iter_mut() {
            let idf = self.vocabulary.idf_values[*idx];
            *tf = (*tf).sqrt() * idf; // Using sqrt of TF to dampen the effect of frequent terms
        }
        
        // Project to lower-dimensional space
        let embedding_values = self.projection.project(&tf_values);
        Embedding::new(embedding_values)
    }

    // Get the vocabulary size
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.size()
    }

    // Get the embedding dimension
    pub fn embedding_dimension(&self) -> usize {
        self.config.embedding_dim
    }

    // Save the model to a file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let serialized = serde_json::to_string(self)?;
        let mut file = File::create(path)?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    // Load a model from a file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let model = serde_json::from_reader(reader)?;
        Ok(model)
    }
}

// A facade for working with embeddings
pub struct NebulaEmbeddings {
    model: Arc<NebulaModel>,
}

impl NebulaEmbeddings {
    pub fn new(model: NebulaModel) -> Self {
        Self {
            model: Arc::new(model),
        }
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, io::Error> {
        let model = NebulaModel::load(path)?;
        Ok(Self::new(model))
    }

    pub fn builder() -> ModelConfigBuilder {
        ModelConfigBuilder::new()
    }

    pub fn embed(&self, text: &str) -> Embedding {
        self.model.embed(text)
    }

    pub fn similarity(&self, text1: &str, text2: &str) -> f32 {
        let embedding1 = self.embed(text1);
        let embedding2 = self.embed(text2);
        embedding1.cosine_similarity(&embedding2)
    }

    pub fn batch_embed<I, S>(&self, texts: I) -> Vec<Embedding>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let texts_vec: Vec<_> = texts.into_iter().collect();
        
        texts_vec.par_iter()
            .map(|text| self.embed(text.as_ref()))
            .collect()
    }

    pub fn vocabulary_size(&self) -> usize {
        self.model.vocabulary_size()
    }

    pub fn embedding_dimension(&self) -> usize {
        self.model.embedding_dimension()
    }
}

// Simple example application to demonstrate usage
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example texts for training
    let training_texts = vec![
        "Rust is a systems programming language focused on safety, speed, and concurrency.",
        "Embeddings are dense vector representations of data in a continuous vector space.",
        "Nebula is a lightweight text embedding model implemented in Rust.",
        "Vector representations help machines understand semantic similarity between texts.",
        "Systems programming requires careful memory management and performance optimization.",
        "Dense vector representations map discrete objects to continuous vector spaces.",
        "Performance optimization is key to building efficient software systems.",
        "Semantic similarity can be measured using cosine distance between vectors.",
        "Memory management in Rust is handled through ownership and borrowing.",
        "Continuous vector spaces allow for mathematical operations on discrete objects.",
    ];

    // Configure and train the model
    let config = ModelConfigBuilder::new()
        .embedding_dim(64)
        .min_token_frequency(2)
        .max_vocabulary_size(1000)
        .random_seed(42)
        .build();

    println!("Training Nebula embeddings model...");
    let model = NebulaModel::train(training_texts, config)?;
    println!("Model trained with {} vocabulary items", model.vocabulary_size());

    // Create the facade
    let nebula = NebulaEmbeddings::new(model);

    // Test embedding generation
    let test_texts = vec![
        "Rust programming language provides memory safety",
        "Vector embeddings represent semantic meaning",
        "Completely unrelated topic about cooking recipes",
    ];

    println!("\nTesting embeddings:");
    for text in &test_texts {
        let embedding = nebula.embed(text);
        println!("Embedded '{}' to {} dimensions", text, embedding.dimension());
    }

    // Test similarity calculation
    println!("\nTesting similarity:");
    for i in 0..test_texts.len() {
        for j in i+1..test_texts.len() {
            let similarity = nebula.similarity(&test_texts[i], &test_texts[j]);
            println!("Similarity between '{}' and '{}': {:.4}", 
                     test_texts[i], test_texts[j], similarity);
        }
    }

    // Save the model
    let save_path = "nebula_model.json";
    nebula.model.save(save_path)?;
    println!("\nModel saved to {}", save_path);

    // Example of loading model (commented out for example purposes)
    // let loaded_nebula = NebulaEmbeddings::from_file(save_path)?;
    // println!("Model loaded with {} vocabulary items", loaded_nebula.vocabulary_size());

    Ok(())
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_embedding_operations() {
        let values = vec![1.0, 2.0, 3.0];
        let embedding = Embedding::new(values);
        
        assert_eq!(embedding.dimension(), 3);
        
        let normalized = embedding.normalize();
        let expected_norm = (1.0f32 + 4.0 + 9.0).sqrt();
        
        assert!((normalized.values()[0] - 1.0 / expected_norm).abs() < 1e-6);
        assert!((normalized.values()[1] - 2.0 / expected_norm).abs() < 1e-6);
        assert!((normalized.values()[2] - 3.0 / expected_norm).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let emb1 = Embedding::new(vec![1.0, 0.0, 0.0]);
        let emb2 = Embedding::new(vec![0.0, 1.0, 0.0]);
        let emb3 = Embedding::new(vec![1.0, 1.0, 0.0]);
        
        assert!((emb1.cosine_similarity(&emb1) - 1.0).abs() < 1e-6);
        assert!((emb1.cosine_similarity(&emb2) - 0.0).abs() < 1e-6);
        assert!((emb1.cosine_similarity(&emb3) - 1.0 / 2.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_tokenization() {
        let text = "Hello, world! This is a TEST.";
        let tokens = preprocessing::tokenize(text);
        
        assert_eq!(tokens, vec!["hello", "world", "test"]);
    }

    #[test]
    fn test_model_training_and_embedding() {
        let texts = vec![
            "test document one",
            "test document two",
            "completely different content",
        ];
        
        let config = ModelConfigBuilder::new()
            .embedding_dim(10)
            .min_token_frequency(1)
            .build();
            
        let model = NebulaModel::train(texts, config).unwrap();
        let embeddings = NebulaEmbeddings::new(model);
        
        let emb1 = embeddings.embed("test document");
        let emb2 = embeddings.embed("different content");
        
        assert_eq!(emb1.dimension(), 10);
        assert_eq!(emb2.dimension(), 10);
        
        // Similar documents should have higher similarity
        let sim1 = embeddings.similarity("test document one", "test document two");
        let sim2 = embeddings.similarity("test document one", "completely different");
        
        assert!(sim1 > sim2);
    }

    #[test]
    fn test_model_save_and_load() {
        let texts = vec!["test document for saving and loading"];
        let config = ModelConfig::default();
        let model = NebulaModel::train(texts, config).unwrap();
        
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_model.json");
        
        // Save model
        model.save(&file_path).unwrap();
        
        // Load model
        let loaded_model = NebulaModel::load(&file_path).unwrap();
        
        // Check that the models have the same properties
        assert_eq!(model.vocabulary_size(), loaded_model.vocabulary_size());
        assert_eq!(model.embedding_dimension(), loaded_model.embedding_dimension());
    }
}