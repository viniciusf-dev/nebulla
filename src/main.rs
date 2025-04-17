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

fn compute_recall_at_k(embeddings: &NebulaEmbeddings, test_data: &[String], k: usize) -> f32 {
    let mut total_hits = 0;
    let mut total_queries = 0;
    
    for (i, query) in test_data.iter().enumerate() {
        if i % 10 == 0 && i < test_data.len() - 5 {  
            total_queries += 1;
            
            let mut candidates: Vec<String> = Vec::new();
            candidates.push(test_data[i+1].clone());  
            
            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..test_data.len()).collect();
            indices.shuffle(&mut rng);
            
            for idx in indices.iter().take(19) {  
                if *idx != i && *idx != i+1 {
                    candidates.push(test_data[*idx].clone());
                }
            }
            
            candidates.shuffle(&mut rng);
            
            let results = embeddings.nearest_neighbors(query, &candidates[..], k);
            
            for (idx, _) in results {
                if candidates[idx] == test_data[i+1] {
                    total_hits += 1;
                    break;
                }
            }
        }
    }
    
    if total_queries > 0 {
        total_hits as f32 / total_queries as f32
    } else {
        0.0
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let training_texts = read_texts_from_parquet("dataset.parquet")?;
    println!("Loaded {} texts from the dataset.", training_texts.len());

    let config = ModelConfigBuilder::new()
        .embedding_dim(256)
        .min_token_frequency(3)
        .max_vocabulary_size(10000)
        .random_seed(42)
        .use_ngrams(true)
        .use_orthogonal_init(true)
        .projection_dropout(0.1)
        .build();

    println!("Training the model...");
    let start_training = Instant::now();
    let model = NebulaModel::train(&training_texts, config)?;
    let training_duration = start_training.elapsed();
    
    println!("Model trained in {:.2?}", training_duration);
    println!("Vocabulary size: {}", model.vocabulary_size());
    
    let nebula = NebulaEmbeddings::new(model);

    let mut rng = rand::thread_rng();
    let mut sample_texts = training_texts.clone();
    sample_texts.shuffle(&mut rng);
    
    let test_texts: Vec<String> = sample_texts.into_iter().take(100).collect();
    println!("\nSelected {} test texts for evaluation.", test_texts.len());

    let start = Instant::now();
    
    let pairs: Vec<(usize, usize)> = (0..test_texts.len().min(10))
        .flat_map(|i| (i+1..test_texts.len().min(i+5)).map(move |j| (i, j)))
        .collect();

    pairs.par_iter().for_each(|(i, j)| {
        let similarity = nebula.similarity(&test_texts[*i], &test_texts[*j]);
        println!("Similarity between '{}' and '{}' : {:.4}",
                 test_texts[*i],
                 test_texts[*j],
                 similarity);
    });

    let similarity_duration = start.elapsed();
    println!("\nSimilarity computation took: {:.2?}", similarity_duration);

    let recall_k = 5;
    println!("\nEvaluating model with Recall@{} metric...", recall_k);
    let start_eval = Instant::now();
    let recall = compute_recall_at_k(&nebula, &test_texts, recall_k);
    let eval_duration = start_eval.elapsed();
    
    println!("Recall@{}: {:.2} (computed in {:.2?})", recall_k, recall, eval_duration);

    println!("\nDemonstrating vector analogies:");
    
    if test_texts.len() >= 20 {
        let a = &test_texts[0];
        let b = &test_texts[1];
        let c = &test_texts[10];
        
        println!("If '{}' is to '{}' as '{}' is to what?", a, b, c);
        
        let candidates: Vec<String> = test_texts[12..20].to_vec();
        let analogies = nebula.analogy(a, b, c, &candidates[..], 3);

        for (idx, score) in analogies {
            println!("  - '{}' (similarity: {:.4})", candidates[idx], score);
        }
    }

    
    nebula.model.save("nebula_model.json")?;
    println!("\nModel saved to nebula_model.json");

    Ok(())
}