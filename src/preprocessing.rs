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

pub fn tokenize(text: &str) -> Vec<String> {
    text.nfc()
        .collect::<String>()
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() >= 2 && !STOP_WORDS.contains(s))
        .map(|s| s.to_string())
        .collect()
}
