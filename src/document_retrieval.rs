use crate::NebulaEmbeddings;

pub struct DocumentRetrievalQuery {
    pub query: String,
    pub relevant_docs: Vec<usize>,
}

pub struct RetrievalMetrics {
    pub precision_at_k: f32,
    pub recall_at_k: f32,
    pub f1_score: f32,
    pub mean_average_precision: f32,
    pub mean_reciprocal_rank: f32,
}

pub fn evaluate_document_retrieval(
    model: &NebulaEmbeddings,
    documents: &[String],
    queries: &[DocumentRetrievalQuery],
    k: usize,
) -> RetrievalMetrics {
    let mut total_precision = 0.0;
    let mut total_recall = 0.0;
    let mut total_map = 0.0;
    let mut total_mrr = 0.0;
    
    for query in queries {
        let results = model.nearest_neighbors(&query.query, documents, k);
        
        let relevant_set: std::collections::HashSet<usize> = query.relevant_docs.iter().cloned().collect();
        let mut relevant_found = 0;
        
        let mut ap_sum = 0.0;
        let mut first_relevant_rank = None;
        
        for (i, (doc_idx, _)) in results.iter().enumerate() {
            if relevant_set.contains(doc_idx) {
                relevant_found += 1;
                
                ap_sum += relevant_found as f32 / (i + 1) as f32;
                
                if first_relevant_rank.is_none() {
                    first_relevant_rank = Some(i + 1);
                }
            }
        }
        
        let precision = if k > 0 {
            relevant_found as f32 / k as f32
        } else {
            0.0
        };
        
        let recall = if !query.relevant_docs.is_empty() {
            relevant_found as f32 / query.relevant_docs.len() as f32
        } else {
            0.0
        };
        
        let ap = if relevant_found > 0 {
            ap_sum / relevant_found as f32
        } else {
            0.0
        };
        
        let rr = match first_relevant_rank {
            Some(rank) => 1.0 / rank as f32,
            None => 0.0,
        };
        
        total_precision += precision;
        total_recall += recall;
        total_map += ap;
        total_mrr += rr;
    }
    
    let num_queries = queries.len() as f32;
    let avg_precision = total_precision / num_queries;
    let avg_recall = total_recall / num_queries;
    
    let f1 = if avg_precision + avg_recall > 0.0 {
        2.0 * avg_precision * avg_recall / (avg_precision + avg_recall)
    } else {
        0.0
    };
    
    RetrievalMetrics {
        precision_at_k: avg_precision,
        recall_at_k: avg_recall,
        f1_score: f1,
        mean_average_precision: total_map / num_queries,
        mean_reciprocal_rank: total_mrr / num_queries,
    }
}

pub fn generate_test_queries(documents: &[String]) -> Vec<DocumentRetrievalQuery> {

    
    let mut queries = Vec::new();
    if documents.len() < 20 {
        return queries;
    }
    
    queries.push(DocumentRetrievalQuery {
        query: "information retrieval system".to_string(),
        relevant_docs: vec![3, 7, 12],
    });
    
    queries.push(DocumentRetrievalQuery {
        query: "neural networks machine learning".to_string(),
        relevant_docs: vec![2, 9, 15],
    });
    
    queries.push(DocumentRetrievalQuery {
        query: "database management systems".to_string(),
        relevant_docs: vec![5, 11, 17],
    });
    
    queries
}