use std::sync::Arc;
use crate::{NebulaModel, Embedding, ModelConfigBuilder};
use rayon::prelude::*;

pub struct NebulaEmbeddings {
    pub model: Arc<NebulaModel>,
}

impl NebulaEmbeddings {
    pub fn new(model: NebulaModel) -> Self {
        Self { model: Arc::new(model) }
    }

    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, std::io::Error> {
        let m = NebulaModel::load(path)?;
        Ok(Self::new(m))
    }

    pub fn builder() -> ModelConfigBuilder {
        ModelConfigBuilder::new()
    }

    pub fn embed(&self, text: &str) -> Embedding {
        self.model.embed(text)
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

    pub fn vocabulary_size(&self) -> usize {
        self.model.vocabulary_size()
    }

    pub fn embedding_dimension(&self) -> usize {
        self.model.embedding_dimension()
    }
}
