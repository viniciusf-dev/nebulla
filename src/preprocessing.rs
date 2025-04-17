use std::collections::HashSet;
use unicode_normalization::UnicodeNormalization;
use lazy_static::lazy_static;

lazy_static! {
    static ref STOP_WORDS: HashSet<&'static str> = {
        let w = vec![
            "a","an","the","and","but","or","for","nor","on","at","to","from","by","with","in","out",
            "is","are","am","was","were","be","being","been","have","has","had","do","does","did",
            "will","would","shall","should","may","might","must","can","could","of","this","that",
            "these","those","i","you","he","she","it","we","they","me","him","her","us","them","its",
            "our","their","what","which","who","whom","whose","when","where","why","how"
        ];
        w.into_iter().collect()
    };
}

pub fn normalize(text: &str) -> String {
    text.nfc()
        .collect::<String>()
        .to_lowercase()
}

pub fn tokenize(text: &str) -> Vec<String> {
    let normalized = normalize(text);
    normalized
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() >= 2 && !STOP_WORDS.contains(s))
        .map(|s| s.to_string())
        .collect()
}

pub fn ngrams(tokens: &[String], n: usize) -> Vec<String> {
    if tokens.is_empty() || n > tokens.len() {
        return Vec::new();
    }
    
    (0..=tokens.len() - n)
        .map(|i| tokens[i..i + n].join("_"))
        .collect()
}

pub fn process_text(text: &str, use_ngrams: bool) -> Vec<String> {
    let tokens = tokenize(text);
    
    if !use_ngrams || tokens.len() < 2 {
        return tokens;
    }
    
    let mut result = tokens.clone();
    let bigrams = ngrams(&tokens, 2);
    result.extend(bigrams);
    
    result
}