use std::time::{Duration, Instant};
use crate::NebulaEmbeddings;

pub struct BenchmarkResult {
    pub throughput: f64,         
    pub avg_embedding_time: f64,  
    pub total_duration: Duration,
    pub num_texts: usize,
}

pub fn benchmark_model(model: &mut NebulaEmbeddings, texts: &[String], num_runs: usize) -> BenchmarkResult {
    let mut total_time = Duration::new(0, 0);
    let mut embedding_times = Vec::with_capacity(texts.len() * num_runs);
    
    for _ in 0..num_runs {
        let start = Instant::now();
        
        for text in texts {
            let embedding_start = Instant::now();
            let embedding = model.embed(text);
            let embedding_time = embedding_start.elapsed();
            embedding_times.push(embedding_time);
            
            model.cache_embedding(text, embedding);
        }
        
        total_time += start.elapsed();
    }
    
    let total_texts = texts.len() * num_runs;
    let throughput = total_texts as f64 / total_time.as_secs_f64();
    
    let total_embedding_ms: f64 = embedding_times
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .sum();
    
    let avg_embedding_time = total_embedding_ms / total_texts as f64;
    
    BenchmarkResult {
        throughput,
        avg_embedding_time,
        total_duration: total_time,
        num_texts: total_texts,
    }
}