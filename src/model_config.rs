#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    pub embedding_dim: usize,
    pub min_token_frequency: usize,
    pub max_vocabulary_size: usize,
    pub random_seed: u64,
    pub use_ngrams: bool,
    pub use_orthogonal_init: bool,
    pub projection_dropout: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 256,
            min_token_frequency: 3,
            max_vocabulary_size: 50000,
            random_seed: 42,
            use_ngrams: true,
            use_orthogonal_init: true,
            projection_dropout: 0.1,
        }
    }
}

pub struct ModelConfigBuilder {
    config: ModelConfig,
}

impl ModelConfigBuilder {
    pub fn new() -> Self {
        Self { config: ModelConfig::default() }
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

    pub fn use_ngrams(mut self, use_ngrams: bool) -> Self {
        self.config.use_ngrams = use_ngrams;
        self
    }

    pub fn use_orthogonal_init(mut self, use_orthogonal: bool) -> Self {
        self.config.use_orthogonal_init = use_orthogonal;
        self
    }

    pub fn projection_dropout(mut self, dropout: f32) -> Self {
        let clamped = dropout.max(0.0).min(0.9);
        self.config.projection_dropout = clamped;
        self
    }

    pub fn build(self) -> ModelConfig {
        self.config
    }
}