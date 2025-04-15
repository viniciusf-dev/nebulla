use nebula_embeddings::{NebulaModel, NebulaEmbeddings, ModelConfigBuilder};
use polars::prelude::*;
use polars::io::parquet::ParquetReader;
use std::fs::File;
use rand::seq::SliceRandom;
use std::time::Instant;
use rayon::prelude::*;

fn read_texts_from_parquet(path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let df = ParquetReader::new(file).finish()?;
    
    let texts: Vec<String> = df
        .column("text")?
        .utf8()?
        .into_iter()
        .filter_map(|opt_val| {
            opt_val.map(|s| s.trim().to_string())
                   .filter(|s| !s.is_empty())
        })
        .collect();
        
    Ok(texts)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let training_texts = read_texts_from_parquet("dataset.parquet")?;
    println!("Loaded {} texts from the dataset.", training_texts.len());

    let config = ModelConfigBuilder::new()
        .embedding_dim(64)
        .min_token_frequency(2)
        .max_vocabulary_size(2000)
        .random_seed(42)
        .build();

    let model = NebulaModel::train(&training_texts, config)?;
    let nebula = NebulaEmbeddings::new(model);

    let mut rng = rand::thread_rng();
    let mut sample_texts = training_texts.clone();
    sample_texts.shuffle(&mut rng);
    let test_texts: Vec<String> = sample_texts.into_iter().take(100).collect();

    println!("\nSelected {} test texts.", test_texts.len());

    let pairs: Vec<(usize, usize)> = (0..test_texts.len())
        .flat_map(|i| (i+1..test_texts.len()).map(move |j| (i, j)))
        .collect();

    let start = Instant::now();
    
    pairs.par_iter().for_each(|(i, j)| {
        let similarity = nebula.similarity(&test_texts[*i], &test_texts[*j]);
        println!("Similarity between '{}' and '{}' : {:.4}",
                 test_texts[*i],
                 test_texts[*j],
                 similarity);
    });

    let duration = start.elapsed();
    println!("\nSimilarity computation took: {:.2?}", duration);

    nebula.model.save("nebula_model.json")?;
    println!("Model saved to nebula_model.json");

    Ok(())
}
