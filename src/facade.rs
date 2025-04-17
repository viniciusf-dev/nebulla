use std::sync::Arc;
use std::collections::HashMap;
use crate::{NebulaModel, Embedding, ModelConfigBuilder};
use rayon::prelude::*;

pub struct NebulaEmbeddings {
    pub model: Arc<NebulaModel>,
    pub cache: HashMap<String, Embedding>,
}

impl NebulaEmbeddings {
    pub fn new(model: NebulaModel) -> Self {
        Self { 
            model: Arc::new(model),
            cache: HashMap::new(),
        }
    }

    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, std::io::Error> {
        let m = NebulaModel::load(path)?;
        Ok(Self::new(m))
    }

    pub fn builder() -> ModelConfigBuilder {
        ModelConfigBuilder::new()
    }

    pub fn embed(&self, text: &str) -> Embedding {
        if let Some(cached) = self.cache.get(text) {
            return cached.clone();
        }
        
        let embedding = self.model.embed(text);
        
        embedding
    }
    
    pub fn cache_embedding(&mut self, text: &str, embedding: Embedding) {
        if self.cache.len() < 10000 {
            self.cache.insert(text.to_string(), embedding);
        }
    }
    
    pub fn similarity(&self, a: &str, b: &str) -> f32 {
        let e1 = self.embed(a);
        let e2 = self.embed(b);
        e1.cosine_similarity(&e2)
    }
    
    pub fn batch_embed<I, S>(&self, texts: I) -> Vec<Embedding>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str> + Send + Sync,
    {
        let v: Vec<S> = texts.into_iter().collect();
        v.par_iter().map(|t| self.embed(t.as_ref())).collect()
    }
    
    pub fn nearest_neighbors<Q, C>(
        &self,
        query: Q,
        candidates: &[C],
        k: usize,
    ) -> Vec<(usize, f32)>
    where
        Q: AsRef<str> + Sync,
        C: AsRef<str> + Sync,
    {
        let query_embedding = self.embed(query.as_ref());
        let candidate_embeddings: Vec<Embedding> = self.batch_embed(candidates);
        query_embedding.top_k_similarity(&candidate_embeddings, k)
    }
    
    pub fn analogy<Q, C>(
        &self,
        a: Q,
        b: Q,
        c: Q,
        candidates: &[C],
        k: usize,
    ) -> Vec<(usize, f32)>
    where
        Q: AsRef<str> + Sync,
        C: AsRef<str> + Sync,
    {
        let embedding_a = self.embed(a.as_ref());
        let embedding_b = self.embed(b.as_ref());
        let embedding_c = self.embed(c.as_ref());
    
        let relationship = &embedding_b - &embedding_a;
        let target = &relationship + &embedding_c;
        let normalized_target = target.normalize();
    
        let candidate_embeddings: Vec<Embedding> = self.batch_embed(candidates);
        normalized_target.top_k_similarity(&candidate_embeddings, k)
    }
    
    pub fn vocabulary_size(&self) -> usize {
        self.model.vocabulary_size()
    }

    pub fn embedding_dimension(&self) -> usize {
        self.model.embedding_dimension()
    }
    
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Drop for NebulaEmbeddings {
    fn drop(&mut self) {
        self.clear_cache();
    }
}