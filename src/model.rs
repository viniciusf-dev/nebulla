use crate::preprocessing::tokenize;
use crate::{Vocabulary, ProjectionMatrix, ModelConfig};
use std::collections::{HashMap, HashSet};
use std::io;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::path::Path;
use std::io::{BufReader, Write};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NebulaModel {
    pub vocabulary: Vocabulary,
    pub projection: ProjectionMatrix,
    pub config: ModelConfig,
}

impl NebulaModel {
    pub fn train<I, S>(texts: I, config: ModelConfig) -> Result<Self, io::Error>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        let mut doc_counts: HashMap<String, usize> = HashMap::new();
        let mut doc_total = 0;

        for text in texts {
            doc_total += 1;
            let tokens = tokenize(text.as_ref());
            let unique_tokens: HashSet<_> = tokens.iter().cloned().collect();

            for token in tokens {
                *word_counts.entry(token).or_insert(0) += 1;
            }
            for token in unique_tokens {
                *doc_counts.entry(token).or_insert(0) += 1;
            }
        }

        let mut vocab: Vec<(String, usize)> = word_counts
            .into_iter()
            .filter(|(_, count)| *count >= config.min_token_frequency)
            .collect();
        vocab.sort_by(|a, b| b.1.cmp(&a.1));

        if vocab.len() > config.max_vocabulary_size {
            vocab.truncate(config.max_vocabulary_size);
        }

        let mut word_to_index = HashMap::new();
        let mut idf_values = Vec::new();

        for (i, (word, _)) in vocab.iter().enumerate() {
            word_to_index.insert(word.clone(), i);
            let doc_freq = doc_counts.get(word).copied().unwrap_or(0);
            let val = (doc_total as f32 / (doc_freq as f32 + 1.0)).ln() + 1.0;
            idf_values.push(val);
        }

        let vocabulary = Vocabulary::new(word_to_index, idf_values);
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

    pub fn embed(&self, text: &str) -> crate::Embedding {
        let tokens = tokenize(text);
        let mut tf: HashMap<usize, f32> = HashMap::new();

        for token in tokens {
            if let Some(idx) = self.vocabulary.get_index(&token) {
                *tf.entry(idx).or_insert(0.0) += 1.0;
            }
        }

        for (idx, val) in tf.iter_mut() {
            let idf = self.vocabulary.idf_values[*idx];
            *val = (*val).sqrt() * idf;
        }

        let embedding_values = self.projection.project(&tf);
        crate::Embedding::new(embedding_values)
    }

    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.size()
    }

    pub fn embedding_dimension(&self) -> usize {
        self.config.embedding_dim
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let serialized = serde_json::to_string(self)?;
        let mut file = File::create(path)?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let model = serde_json::from_reader(reader)?;
        Ok(model)
    }
}
