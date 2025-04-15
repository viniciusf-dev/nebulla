# Nebulla 🌌

A lightweight, high-performance text embedding model implemented in Rust.

Nebulla is designed to efficiently convert text into numerical vector representations (embeddings) with a simple, modular architecture. Built in Rust for speed and safety, it provides a clean API to integrate into your own projects or use as a standalone command-line utility.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
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

---

## Architecture and Code Structure

The repository is organized into several modules located in the `src/` directory. Here’s an in-depth look at each component:

### Preprocessing

- **File:** `preprocessing.rs`
- **Purpose:**  
  - Normalizes input text by converting to lowercase, removing punctuation, and trimming whitespace.
  - Tokenizes the normalized text into individual tokens.
- **How It Works:**  
  The module defines helper functions that process the raw input string before it’s fed into the model. This standardization ensures consistency in tokenization and subsequent embedding lookup.
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

## Usage

### Library Usage

Include Nebulla in your Rust project by adding it to your `Cargo.toml`:

```toml
[dependencies]
nebulla = { git = "https://github.com/viniciusf-dev/nebulla.git" }
```

Then import and use the API:

```rust
use nebulla::facade::Nebulla;

fn main() {
    // Initialize Nebulla with default configuration or a custom one from `model_config`
    let model = Nebulla::new();
    let text = "The quick brown fox jumps over the lazy dog.";
    let embedding = model.embed(text);
    println!("Computed Embedding: {:?}", embedding);
}
```

### Command-Line Interface (CLI)

After building the project, you can run the CLI:

```bash
cargo run -- "Your text to embed goes here."
```

This command will output the computed embedding vector to the terminal.

---

## Examples

### Example: Using the Facade to Compute an Embedding

```rust
use nebulla::facade::Nebulla;

fn main() {
    let text = "Rust is fast and memory efficient.";
    let model = Nebulla::new(); // Uses default configuration
    let embedding = model.embed(text);
    println!("Embedding for the input text: {:?}", embedding);
}
```

### Example: Inspecting Vocabulary

If you wish to examine or extend the vocabulary, you may use methods from the `vocabulary` module:

```rust
use nebulla::vocabulary::Vocabulary;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vocab = Vocabulary::from_file("path/to/vocab.txt")?;
    if let Some(index) = vocab.get_index("rust") {
        println!("The token 'rust' is mapped to index: {}", index);
    }
    Ok(())
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

---

### Explanation of How the Model Works

1. **Preprocessing:**  
   The text input is first normalized (lowercase, punctuation removal, etc.) and tokenized. This process ensures that the tokens match those expected in the vocabulary.

2. **Vocabulary Mapping:**  
   The tokenized text is then mapped to indices using the vocabulary manager. This is crucial for consistent lookup of embeddings.

3. **Embedding Lookup & Computation:**  
   The indices retrieved are fed into the embedding module, where each index is associated with a pre-trained (or initialized) vector. The result is a set of vector representations corresponding to the tokens.

4. **Projection:**  
   An optional projection layer performs a linear transformation on the combined embeddings. This may involve dimensionality reduction or adaptation to a different representation space required for downstream tasks.

5. **Facade Integration:**  
   Finally, the `facade.rs` module acts as the public API. It orchestrates the preprocessing, lookup, and projection steps so that the user simply receives the final embedding vector.

This modular design not only promotes high performance (by leveraging Rust’s efficiency) but also ensures that each stage of the text embedding pipeline is easy to test, extend, and maintain.

