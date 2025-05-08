<div align="center">

<img src="https://github.com/viniciusf-dev/nebulla/blob/main/public/nebulla_icon.png" width="1000" height="350">

[![][version]](https://github.com/viniciusf-dev/nebulla)
[![][license]](https://github.com/viniciusf-dev/nebulla/blob/master/LICENSE)
[![][stars]](https://github.com/viniciusf-dev/nebulla)
[![][commit]](https://github.com/viniciusf-dev/nebulla)

</div>

A lightweight, high-performance text embedding model implemented in Rust.

Nebulla is designed to efficiently convert text into numerical vector representations (embeddings) with a simple, modular architecture. Built in Rust for speed and safety, it provides a clean API to integrate into your own projects or use as a standalone command-line utility.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
  - [Training Process](#training-process)
  - [Embedding Computation](#embedding-computation)
  - [Similarity Comparisons](#similarity-comparisons)
  - [Vector Operations](#vector-operations)
- [Architecture and Code Structure](#architecture-and-code-structure)
  - [Preprocessing](#preprocessing)
  - [Vocabulary Management](#vocabulary-management)
  - [Embedding Computation](#embedding-computation)
  - [Model Core and Configuration](#model-core-and-configuration)
  - [Projection Layer](#projection-layer)
  - [Facade Pattern](#facade-pattern)
- [Installation](#installation)
- [Usage](#usage)
  - [Library Usage](#library-usage)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
  - [Dataset](#dataset)
- [Advanced Features](#advanced-features)
  - [Nearest Neighbors Search](#nearest-neighbors-search)
  - [Vector Analogies](#vector-analogies)
  - [Recall Evaluation](#recall-evaluation)
  - [BM-25 Weighting](#bm-25-weighting)
  - [Embedding Operations](#embedding-operations)
  - [Performance Benchmarking](#performance-benchmarking)
- [Evaluation](#evaluation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

Nebulla is built to efficiently transform raw text into numerical embeddings. The process includes:

1. **Preprocessing** – Normalizing and tokenizing input text.
2. **Vocabulary Mapping** – Converting tokens into numerical indices.
3. **Embedding Lookup & Computation** – Transforming token indices into dense vector representations.
4. **Projection** – Optionally reducing or transforming the embedding dimensionality.
5. **Facade Interface** – Providing a simple API that abstracts complexity for both library consumers and CLI users.

The model is tuned for performance in real-world applications while keeping the codebase modular and easy to extend.

---

## Features

- **High Performance:** Written in Rust to leverage speed and memory safety.
- **Modular Design:** Separate modules for preprocessing, vocabulary management, model computation, and projection.
- **Simple API:** A facade layer offers a unified interface to access the text embedding functionality.
- **Lightweight:** Minimal dependencies with an emphasis on speed and low memory footprint.
- **Easy Integration:** Use as a standalone CLI tool or integrate the library into your application.
- **Advanced Algorithms:** Implements BM-25 weighting for better semantic understanding.
- **Vector Operations:** Supports mathematical operations on embeddings like addition, subtraction, and scaling.
- **Nearest Neighbors:** Efficiently find semantically similar content using cosine similarity.
- **Vector Analogies:** Solve word analogy problems (A is to B as C is to ?) using vector operations.
- **Performance Metrics:** Built-in evaluation using Recall@K and benchmarking tools.
- **Parallel Processing:** Leverages Rayon for parallel computation of batch embeddings.

---

## How It Works

### Training Process

When you run `cargo run`, Nebulla performs the following training steps:

1. **Data Loading**: Reads text data from a Parquet file (`dataset.parquet`) containing WikiText data.
2. **Vocabulary Building**: Processes the texts, counts token frequencies, and builds a vocabulary of the most common tokens.
3. **IDF Calculation**: Computes inverse document frequency (IDF) values for each token, which measure how informative a token is.
4. **Embedding Initialization**: Creates embedding vectors for each token in the vocabulary using orthogonal initialization (optional) for better performance.
5. **Model Configuration**: Uses parameters like embedding dimension, minimum token frequency, and n-gram settings to customize the model.

### Embedding Computation

When computing embeddings for texts, Nebulla:

1. **Tokenizes** the input text into words and optionally generates n-grams.
2. **Computes term frequencies** (TF) for each token in the input.
3. **Applies BM-25 weighting** to balance term frequency with document length and IDF values.
4. **Projects** the sparse TF-IDF vector into a dense embedding space using the projection matrix.
5. **Normalizes** the final embedding vector to enable reliable similarity calculations.

### Similarity Comparisons

Nebulla compares texts by:

1. **Computing embeddings** for each text.
2. **Calculating cosine similarity** between the normalized embedding vectors, which measures the angle between them.
3. **Ranking results** by similarity score for nearest neighbor searches.

### Vector Operations

The model supports vector operations that enable semantic reasoning:

1. **Addition**: Combining semantic concepts (e.g., "king" + "woman").
2. **Subtraction**: Finding relationships between concepts (e.g., "king" - "man").
3. **Scaling**: Emphasizing or de-emphasizing certain embedding dimensions.
4. **Normalization**: Ensuring vectors have unit length for consistent similarity calculations.

---

## Architecture and Code Structure

The repository is organized into several modules located in the `src/` directory. Here's an in-depth look at each component:

### Preprocessing

- **File:** `preprocessing.rs`
- **Purpose:**  
  - Normalizes input text by converting to lowercase, removing punctuation, and trimming whitespace.
  - Tokenizes the normalized text into individual tokens.
- **How It Works:**  
  The module defines helper functions that process the raw input string before it's fed into the model. This standardization ensures consistency in tokenization and subsequent embedding lookup.
- **Key Functions:**  
  - `normalize(text: &str) -> String`
  - `tokenize(text: &str) -> Vec<String>`

### Vocabulary Management

- **File:** `vocabulary.rs`
- **Purpose:**  
  - Manages the mapping between tokens and their unique indices.
  - Stores a vocabulary (often pre-built or generated from training data).
- **How It Works:**  
  It uses an efficient data structure (e.g., a hash map) to resolve token-to-index conversion. It may include routines to add new tokens or maintain frequency counts if extended.
- **Key Functions / Structures:**  
  - `Vocabulary` struct with methods like `get_index(token: &str) -> Option<usize>` and `from_file(path: &str) -> Result<Vocabulary, Error>`.

### Embedding Computation

- **File:** `embedding.rs`
- **Purpose:**  
  - Contains the logic to look up or compute embedding vectors for given tokens.
- **How It Works:**  
  The module leverages the preprocessed tokens and uses the vocabulary mappings to retrieve associated vector representations. In many cases, this is a simple lookup from a pre-initialized embedding matrix.
- **Key Functions / Structures:**  
  - `embed(tokens: &[String]) -> Vec<f32>` – This function retrieves or computes embedding vectors for the input tokens.
  
### Model Core and Configuration

- **Files:** `model.rs` and `model_config.rs`
- **Purpose:**  
  - `model.rs`: Implements the core text embedding model including forward propagation.
  - `model_config.rs`: Handles configuration parameters such as embedding dimensions, layer sizes, and other hyperparameters.
- **How It Works:**  
  The model is implemented as a Rust struct (e.g., `struct Model`) which contains weights (or lookup tables), configuration parameters, and methods to perform forward passes on the tokenized input. The configuration file ensures that parameters can be adjusted without changing code.
- **Key Functions / Structures:**  
  - In `model.rs`: A `Model` struct with a method like `forward(&self, input: &[usize]) -> Vec<f32>` which computes the embedding.
  - In `model_config.rs`: A `ModelConfig` struct with methods such as `load(config_path: &str) -> Result<ModelConfig, Error>`.

### Projection Layer

- **File:** `projection.rs`
- **Purpose:**  
  - Applies a linear transformation to the output embeddings.
  - Optionally reduces dimensionality or adjusts embedding space.
- **How It Works:**  
  Implements a simple matrix multiplication function that projects the original embeddings into another space. This is useful when integrating with downstream tasks that require a fixed output dimension.
- **Key Functions:**  
  - `project(embedding: &Vec<f32>, projection_matrix: &[[f32]]) -> Vec<f32>`.

### Facade Pattern

- **File:** `facade.rs`
- **Purpose:**  
  - Provides a simplified interface to the various modules.
  - Abstracts preprocessing, embedding computation, and projection steps behind a single API.
- **How It Works:**  
  The facade encapsulates the complexity by internally calling the other modules. Users can simply call something like `Nebulla::embed(text: &str)` to receive the processed and computed embedding.
- **Key Functions / Structures:**  
  - A `Nebulla` struct with a public method `embed(&self, text: &str) -> Vec<f32>` that runs the entire pipeline.

### Library and Entry Point

- **Files:** `lib.rs` and `main.rs`
- **Purpose:**  
  - `lib.rs`: Exposes public interfaces and re-exports the key modules to be used as a library.
  - `main.rs`: Acts as the executable entry point. It parses command-line arguments, loads configuration, and demonstrates usage of the model.
- **How It Works:**  
  - `lib.rs` aggregates modules so that external projects can import Nebulla as a dependency.
  - `main.rs` contains a simple CLI that may accept text input via arguments or standard input and then outputs the computed embedding.
- **Key Functions:**  
  - In `main.rs`:  
    ```rust
    fn main() {
        // Parse command-line input
        let input_text = std::env::args().nth(1).expect("Please provide input text.");
        let embedding = Nebulla::new(/* config or defaults */).embed(&input_text);
        println!("Embedding: {:?}", embedding);
    }
    ```

---

### Dataset
The current code version uses a parquet dataset from [WikiText Hugging Face](https://huggingface.co/datasets/Salesforce/wikitext)

## Installation

### Prerequisites

- **Rust** (1.65 or higher recommended)
- **Cargo** (included with Rust)

### Building the Project

1. **Clone the repository:**

   ```bash
   git clone https://github.com/viniciusf-dev/nebulla.git
   cd nebulla
   ```

2. **Build the project:**

   ```bash
   cargo build --release
   ```

3. **Run tests (if available):**

   ```bash
   cargo test
   ```

---

## Advanced Features

### Nearest Neighbors Search

Nebulla can efficiently find the most semantically similar texts from a collection:

```rust
// Find the top 5 most similar candidates to a query
let results = nebula.nearest_neighbors(query_text, &candidate_texts, 5);

// Results contain indices and similarity scores
for (idx, score) in results {
    println!("Similar text: {}, Score: {}", candidate_texts[idx], score);
}
```

The nearest neighbor functionality uses cosine similarity between normalized embedding vectors and returns the top K results.

### Vector Analogies

Nebulla can solve analogy problems using vector arithmetic:

```rust
// If a is to b as c is to what?
let results = nebula.analogy(a, b, c, &candidates, k);
```

This works by:
1. Computing the relationship vector between `a` and `b` (b - a)
2. Applying this relationship to `c` (c + (b - a))
3. Finding candidates most similar to the resulting vector

This makes it possible to answer questions like "king is to queen as man is to ?" (answer: woman).

### Recall Evaluation

The model includes a built-in recall evaluation system:

```rust
let recall = compute_recall_at_k(&nebula, &test_texts, k);
```

This measures the model's ability to retrieve known relevant texts within the top K results. This is a common metric in information retrieval systems.

### BM-25 Weighting

Unlike basic TF-IDF weighting, Nebulla uses the advanced BM-25 algorithm for token weighting:

```rust
// BM-25 formula implementation from model.rs
let k1 = 1.2;
let b = 0.75;
let normalized_freq = freq / total_tokens;
let term_saturation = ((k1 + 1.0) * normalized_freq) / 
                      (k1 * (1.0 - b + b) + normalized_freq);
            
*val = term_saturation * idf;
```

BM-25 addresses the saturation problem in basic TF weighting, preventing very frequent terms from dominating the embedding.

### Embedding Operations

Nebulla supports various mathematical operations on embeddings:

```rust
// Add two embeddings
let combined = &embedding_a + &embedding_b;

// Find the difference between embeddings
let difference = &embedding_a - &embedding_b;

// Scale an embedding
let scaled = &embedding * 0.5;

// Calculate distance between embeddings
let distance = embedding_a.distance(&embedding_b);
```

These operations enable semantic reasoning and exploration of the embedding space.

### Performance Benchmarking

The library includes tools for performance measurement:

```rust
let benchmark_result = benchmark_model(&mut nebula, &texts, runs);
println!("Throughput: {} texts/sec", benchmark_result.throughput);
println!("Average embedding time: {} ms", benchmark_result.avg_embedding_time);
```

This helps when optimizing the model for production use cases.

---

## Usage

### Library Usage

Include Nebulla in your Rust project by adding it to your `Cargo.toml`:

```toml
[dependencies]
nebulla = { git = "https://github.com/viniciusf-dev/nebulla.git" }
```

Then import and use the API:

```rust
use nebulla::facade::NebulaEmbeddings;

fn main() {
    // Initialize Nebulla with default configuration or a custom one from `model_config`
    let model = NebulaEmbeddings::new();
    let text = "The quick brown fox jumps over the lazy dog.";
    let embedding = model.embed(text);
    println!("Computed Embedding: {:?}", embedding);
}
```

### Command-Line Interface (CLI)

After building the project, you can run the CLI:

```bash
cargo run
```

This will:
1. Load the WikiText dataset from `dataset.parquet`
2. Train a new embedding model with the specified configuration
3. Run similarity comparisons between sample texts
4. Evaluate the model using Recall@K metric
5. Demonstrate vector analogies
6. Save the trained model to `nebula_model.json`

---

## Evaluation

Nebulla includes several evaluation modules to measure different aspects of embedding quality and performance. These modules help quantify how well the embeddings capture semantic relationships and perform on specific tasks.

### Clustering Quality

- **File:** `clustering_quality.rs`
- **Purpose:** Evaluates how well the embedding model clusters semantically related texts.
- **How It Works:** The module applies k-means clustering to embeddings of labeled texts and compares the resulting clusters with the ground truth labels.
- **Key Metrics:**
  - **Purity:** Measures how homogeneous the clusters are with respect to the true labels.
  - **Normalized Mutual Information:** Quantifies the mutual dependence between the clustering assignment and the ground truth.
  - **Rand Index:** Measures the similarity between two data clusterings by considering all pairs of samples.
- **Example Usage:**
```rust
let clusters = generate_test_clusters();
let metrics = evaluate_clustering(&model, &clusters);
println!("Clustering Metrics: Purity={:.4}, NMI={:.4}, RI={:.4}", 
         metrics.purity, metrics.normalized_mutual_information, metrics.rand_index);
```

### Document Retrieval

- **File:** `document_retrieval.rs`
- **Purpose:** Assesses how well the model retrieves relevant documents based on a query.
- **How It Works:** For each query, the module ranks documents by embedding similarity and compares the retrieved set with known relevant documents.
- **Key Metrics:**
  - **Precision@K:** The proportion of retrieved documents that are relevant.
  - **Recall@K:** The proportion of relevant documents that are retrieved.
  - **F1 Score:** The harmonic mean of precision and recall.
  - **Mean Average Precision (MAP):** Average of precision values at positions where relevant documents are found.
  - **Mean Reciprocal Rank (MRR):** Average of the reciprocal of the rank at which the first relevant document is found.
- **Example Usage:**
```rust
let documents = vec!["doc1 content", "doc2 content", ...].iter().map(|s| s.to_string()).collect();
let queries = generate_test_queries(&documents);
let metrics = evaluate_document_retrieval(&model, &documents, &queries, 5);
println!("Retrieval Metrics: P@K={:.4}, R@K={:.4}, F1={:.4}, MAP={:.4}, MRR={:.4}",
         metrics.precision_at_k, metrics.recall_at_k, metrics.f1_score, 
         metrics.mean_average_precision, metrics.mean_reciprocal_rank);
```

### Embedding Stability

- **File:** `embedding_stability.rs`
- **Purpose:** Measures how robust embeddings are to small variations in input text.
- **How It Works:** The module creates variants of input texts with controlled mutations (word deletion, duplication, reordering, character removal) and measures similarity between original and variant embeddings.
- **Key Metrics:**
  - **Average Similarity:** Mean cosine similarity between original and variant embeddings.
  - **Min/Max Similarity:** Range of cosine similarities observed.
  - **Standard Deviation:** Variation in similarity across different mutations.
- **Example Usage:**
```rust
let texts = vec!["sample text one", "sample text two"].iter().map(|s| s.to_string()).collect();
let metrics = evaluate_stability(&model, &texts, 10, 0.3); // 10 variants per text, 30% mutation rate
println!("Stability Metrics: Avg={:.4}, Min={:.4}, Max={:.4}, StdDev={:.4}",
         metrics.average_similarity, metrics.min_similarity, 
         metrics.max_similarity, metrics.standard_deviation);
```

### Semantic Coherence

- **File:** `semantic_coherence.rs`
- **Purpose:** Evaluates how well the model captures semantic relationships between words.
- **How It Works:** The module tests word pairs labeled as either semantically related or unrelated and measures if the embedding similarities align with these expectations.
- **Key Metrics:**
  - **Accuracy:** Proportion of correct predictions (high similarity for related pairs, low for unrelated).
  - **Average Related/Unrelated Similarity:** Mean similarity scores for each group.
  - **Contrast Score:** Difference between related and unrelated similarities (higher is better).
- **Example Usage:**
```rust
let pairs = generate_test_pairs();
let result = evaluate_semantic_coherence(&model, &pairs);
println!("Semantic Coherence: Accuracy={:.4}, Related={:.4}, Unrelated={:.4}, Contrast={:.4}",
         result.accuracy, result.average_related_similarity, 
         result.average_unrelated_similarity, result.contrast_score);
```

### Temporal Ordering

- **File:** `temporal_ordering.rs`
- **Purpose:** Assesses how well the embedding space preserves temporal relationships in sequential texts.
- **How It Works:** The module analyzes sequences of texts (like chapters or ordered paragraphs) to determine if consecutive texts are more similar than non-consecutive ones.
- **Key Metrics:**
  - **Kendall Tau:** Measures the correlation between the ordering of embedding similarities and the expected ordering.
  - **Correlation Coefficient:** Difference between the average similarity of consecutive vs. non-consecutive pairs.
  - **Average Distance:** Mean distance between consecutive embeddings in the sequence.
- **Example Usage:**
```rust
let sequences = generate_test_sequences();
let metrics = evaluate_temporal_ordering(&model, &sequences);
println!("Temporal Ordering: Kendall Tau={:.4}, Correlation={:.4}, Avg Distance={:.4}",
         metrics.kendall_tau, metrics.correlation_coefficient, metrics.average_distance);
```

These evaluation modules provide comprehensive insights into different aspects of embedding quality, helping to benchmark and improve the Nebulla model across various natural language processing tasks.

## Examples

### Example: Using the Facade to Compute an Embedding

```rust
use nebulla::facade::NebulaEmbeddings;

fn main() {
    let text = "Rust is fast and memory efficient.";
    let model = NebulaEmbeddings::new(); // Uses default configuration
    let embedding = model.embed(text);
    println!("Embedding for the input text: {:?}", embedding);
}
```

### Example: Finding Similar Texts

```rust
use nebulla::facade::NebulaEmbeddings;

fn main() {
    let model = NebulaEmbeddings::new();
    let query = "Artificial intelligence and machine learning";
    let candidates = vec![
        "Deep neural networks are transforming AI research".to_string(),
        "Efficient memory management in Rust".to_string(),
        "Natural language processing models".to_string(),
        "Web development frameworks comparison".to_string(),
    ];
    
    let results = model.nearest_neighbors(query, &candidates, 2);
    for (idx, score) in results {
        println!("Similar text: {}, Score: {:.4}", candidates[idx], score);
    }
}
```

### Example: Solving Analogies

```rust
use nebulla::facade::NebulaEmbeddings;

fn main() {
    let model = NebulaEmbeddings::new();
    let a = "king";
    let b = "queen";
    let c = "man";
    let candidates = vec![
        "woman".to_string(),
        "person".to_string(),
        "royal".to_string(),
        "girl".to_string(),
    ];
    
    let results = model.analogy(a, b, c, &candidates, 1);
    println!("'{}' is to '{}' as '{}' is to '{}'", a, b, c, candidates[results[0].0]);
}
```

---

## Contributing

Contributions are welcome! To contribute:

1. **Fork the repository.**
2. **Create your feature branch:**  
   `git checkout -b feature/your-feature`
3. **Commit your changes:**  
   `git commit -am 'Add some feature'`
4. **Push to the branch:**  
   `git push origin feature/your-feature`
5. **Open a Pull Request**

Please follow the existing coding style and write tests for new features where appropriate.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Thanks to the Rust community for continuous support and innovation.
- Inspired by modern text embedding methodologies and efficient systems design.

---

*Nebulla* is a project dedicated to high performance, clean code, and ease of use in text embedding tasks. We hope this README provides clarity on both how to get started and the inner workings of the model.


[version]: https://img.shields.io/github/v/release/viniciusf-dev/nebulla?label=Version&color=1AD1A5
[license]: https://img.shields.io/github/license/viniciusf-dev/nebulla?label=License&color=1AD1A5
[stars]: https://badgen.net/github/stars/viniciusf-dev/nebulla?label=GitHub%20stars&color=1AD1A5
[commit]: https://img.shields.io/github/last-commit/viniciusf-dev/nebulla?label=Last%20commit&color=1AD1A5
[logo-url]: https://github.com/viniciusf-dev/nebulla/blob/main/public/nebulla_icon.png
