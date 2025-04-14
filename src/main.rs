use nebula_embeddings::{NebulaModel, NebulaEmbeddings, ModelConfigBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
        "Machine learning involves training models on labeled datasets.",
        "Neural networks use layers of interconnected nodes to process input data.",
        "Large language models are capable of generating coherent text.",
        "Parallel processing can accelerate data analysis in modern systems.",
        "Concurrency allows tasks to run simultaneously, improving throughput.",
    ];

    let config = ModelConfigBuilder::new()
        .embedding_dim(64)
        .min_token_frequency(2)
        .max_vocabulary_size(2000)
        .random_seed(42)
        .build();

    let model = NebulaModel::train(training_texts, config)?;
    let nebula = NebulaEmbeddings::new(model);

    let test_texts = vec![
        "Rust programming language provides memory safety",
        "Vector embeddings represent semantic meaning",
        "Completely unrelated topic about cooking recipes",
        "Machine learning techniques with neural networks",
        "Parallel processing in modern computing architectures"
    ];

    for t in &test_texts {
        let e = nebula.embed(t);
        println!("'{}' -> {} dimensions", t, e.dimension());
    }

    for i in 0..test_texts.len() {
        for j in i+1..test_texts.len() {
            let s = nebula.similarity(&test_texts[i], &test_texts[j]);
            println!("Similarity '{}' vs '{}': {:.4}", test_texts[i], test_texts[j], s);
        }
    }

    nebula.model.save("nebula_model.json")?;
    println!("Model saved");

    Ok(())
}
