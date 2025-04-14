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
        let mut word_counts = HashMap::new();
        let mut doc_counts = HashMap::new();
        let mut doc_total = 0;
        for t in texts {
            doc_total += 1;
            let toks = tokenize(t.as_ref());
            let unique: HashSet<_> = toks.iter().cloned().collect();
            for x in toks {
                *word_counts.entry(x).or_insert(0) += 1;
            }
            for x in unique {
                *doc_counts.entry(x).or_insert(0) += 1;
            }
        }
        let mut vocab: Vec<(String, usize)> = word_counts
            .into_iter()
            .filter(|(_, c)| *c >= config.min_token_frequency)
            .collect();
        vocab.sort_by(|a, b| b.1.cmp(&a.1));
        if vocab.len() > config.max_vocabulary_size {
            vocab.truncate(config.max_vocabulary_size);
        }
        let mut w2i = HashMap::new();
        let mut idf = Vec::new();
        for (i, (w, _)) in vocab.iter().enumerate() {
            w2i.insert(w.clone(), i);
            let df = doc_counts.get(w).copied().unwrap_or(0);
            let val = (doc_total as f32 / (df as f32 + 1.0)).ln() + 1.0;
            idf.push(val);
        }
        let v = Vocabulary::new(w2i, idf);
        let p = ProjectionMatrix::new(v.size(), config.embedding_dim, config.random_seed);
        Ok(Self { vocabulary: v, projection: p, config })
    }

    pub fn embed(&self, text: &str) -> crate::Embedding {
        let toks = tokenize(text);
        let mut tf = HashMap::new();
        for t in toks {
            if let Some(i) = self.vocabulary.get_index(&t) {
                *tf.entry(i).or_insert(0.0) += 1.0;
            }
        }
        for (i, val) in tf.iter_mut() {
            let idf = self.vocabulary.idf_values[*i];
            *val = (*val).sqrt() * idf;
        }
        let emb = self.projection.project(&tf);
        crate::Embedding::new(emb)
    }

    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.size()
    }

    pub fn embedding_dimension(&self) -> usize {
        self.config.embedding_dim
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let s = serde_json::to_string(self)?;
        let mut f = File::create(path)?;
        f.write_all(s.as_bytes())?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, io::Error> {
        let f = File::open(path)?;
        let r = BufReader::new(f);
        let m = serde_json::from_reader(r)?;
        Ok(m)
    }
}
