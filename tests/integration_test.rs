use nebula_embeddings::{NebulaModel, NebulaEmbeddings, ModelConfigBuilder, Embedding, preprocessing};
use tempfile::tempdir;

#[test]
fn test_embedding_operations() {
    let v = vec![1.0, 2.0, 3.0];
    let e = Embedding::new(v);
    assert_eq!(e.dimension(), 3);
    let n = e.normalize();
    let m = (1.0f32 + 4.0 + 9.0).sqrt();
    assert!((n.values()[0] - 1.0 / m).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity() {
    let e1 = Embedding::new(vec![1.0, 0.0]);
    let e2 = Embedding::new(vec![0.0, 1.0]);
    let s = e1.cosine_similarity(&e2);
    assert!((s - 0.0).abs() < 1e-6);
}

#[test]
fn test_tokenization() {
    let t = "Hello, world! This is a TEST.";
    let toks = preprocessing::tokenize(t);
    assert_eq!(toks, vec!["hello", "world", "test"]);
}

#[test]
fn test_training() {
    let txt = vec!["test document one", "test document two", "completely different content"];
    let c = ModelConfigBuilder::new().embedding_dim(8).min_token_frequency(1).build();
    let m = NebulaModel::train(txt, c).unwrap();
    let e = NebulaEmbeddings::new(m);
    let x = e.embed("test document");
    let y = e.embed("different content");
    assert_eq!(x.dimension(), 8);
    assert_eq!(y.dimension(), 8);
    let s1 = e.similarity("test document one", "test document two");
    let s2 = e.similarity("test document one", "completely different");
    assert!(s1 > s2);
}

#[test]
fn test_save_load() {
    let txt = vec!["test save load"];
    let c = ModelConfigBuilder::new().build();
    let m = NebulaModel::train(txt, c).unwrap();
    let d = tempdir().unwrap();
    let f = d.path().join("model.json");
    m.save(&f).unwrap();
    let l = NebulaModel::load(&f).unwrap();
    assert_eq!(m.vocabulary_size(), l.vocabulary_size());
}
