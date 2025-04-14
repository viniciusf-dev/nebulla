#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    pub embedding_dim: usize,
    pub min_token_frequency: usize,
    pub max_vocabulary_size: usize,
    pub random_seed: u64,
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

    pub fn build(self) -> ModelConfig {
        self.config
    }
}
