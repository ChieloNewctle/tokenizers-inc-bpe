use std::{convert::TryFrom, iter::FromIterator, ops::Deref, sync::Arc};

use ahash::AHashSet;
use mtc_incremental_bpe::{
    DictBuildError, Dictionary, IncBpeTokenizer, NormalizedDict, NormalizedDictBuildError,
    Token as Bytes, TokenId, Vocab, VocabBuildError,
};
use scc::HashCache;
use serde::{Deserialize, Serialize};

use crate::{
    models::bpe::{BpeTrainer, BPE},
    utils::cache::{DEFAULT_CACHE_CAPACITY, MAX_LENGTH},
    Model,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "BPE")]
pub struct IncrementalBpe {
    #[serde(flatten)]
    pub(crate) bpe: BPE,
    #[serde(skip)]
    pub(crate) cache_disabled: bool,
    #[serde(skip)]
    inner: Arc<IncrementalBpeRef>,
}

impl IncrementalBpe {
    pub fn bpe(&self) -> &BPE {
        &self.bpe
    }
}

type FallbackBytes = Option<Arc<[u32; 1 << 8]>>;
type SccHashCache = HashCache<String, Arc<[crate::Token]>, ahash::RandomState>;

#[derive(Debug)]
pub struct IncrementalBpeRef {
    pub(crate) tokenizer: IncBpeTokenizer,
    pub(crate) unk_token_id: u32,
    pub(crate) fallback_bytes: FallbackBytes,
    pub(crate) cache: SccHashCache,
}

impl IncrementalBpeRef {
    pub fn unk_token_id(&self) -> u32 {
        self.unk_token_id
    }
}

impl Deref for IncrementalBpeRef {
    type Target = IncBpeTokenizer;

    fn deref(&self) -> &Self::Target {
        &self.tokenizer
    }
}

impl PartialEq for IncrementalBpe {
    fn eq(&self, other: &Self) -> bool {
        self.bpe == other.bpe
    }
}

impl Deref for IncrementalBpe {
    type Target = IncrementalBpeRef;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl TryFrom<BPE> for IncrementalBpe {
    type Error = Error;

    fn try_from(bpe: BPE) -> Result<Self, Self::Error> {
        let cache_size = bpe.cache_capacity();
        Self::new(bpe, Some(cache_size))
    }
}

#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("vocab error: {0}")]
    VocabError(#[from] VocabBuildError),
    #[error("dict error: {0}")]
    DictError(#[from] DictBuildError),
    #[error("normalize error: {0}")]
    NormalizedDictError(#[from] NormalizedDictBuildError),
    #[error("byte fallback token `<{byte:#04X}>` not found")]
    ByteFallbackTokenRequired { byte: u8 },
    #[error("incompatible config: {fields:?}")]
    IncompatibleSettings { fields: Vec<&'static str> },
    #[error("invalid rule merging {pre} and {suc} to {id}: {reason}")]
    InvalidRule {
        pre: u32,
        suc: u32,
        id: u32,
        reason: &'static str,
    },
}

fn convert_byte_fallback(s: &str) -> Option<u8> {
    const LEN: usize = "<0x00>".len();
    if s.len() == LEN && (s.starts_with("<0x") || s.starts_with("<0X")) && s.ends_with(">") {
        u8::from_str_radix(&s[3..5], 16).ok()
    } else {
        None
    }
}

fn build_dict_from_bpe(bpe: &BPE) -> Result<(NormalizedDict, FallbackBytes), Error> {
    let mut singletons = AHashSet::new();
    let mut fallback_bytes = Box::new([u32::MAX; 1 << 8]);
    let get_token = |token_id| {
        let token = bpe
            .vocab_r
            .get(&token_id)
            .map(|s| s.to_owned())
            .unwrap_or_default();
        if let Some(byte) = convert_byte_fallback(&token).filter(|_| bpe.byte_fallback) {
            fallback_bytes[byte as usize] = token_id;
        } else if token.len() <= 4 && token.chars().count() == 1 {
            singletons.insert(token_id);
        }
        Bytes::from(token)
    };
    let vocab = Vocab::new((0..bpe.get_vocab_size() as u32).map(get_token))?;

    let singletons = singletons;
    let fallback_bytes = if bpe.byte_fallback {
        let find = fallback_bytes
            .iter()
            .copied()
            .enumerate()
            .find(|(_, id)| *id == u32::MAX);
        if let Some((byte, _)) = find {
            return Err(Error::ByteFallbackTokenRequired { byte: byte as _ });
        }
        Some(Arc::from(fallback_bytes))
    } else {
        None
    };

    let mut rules = bpe
        .merges
        .iter()
        .map(|(&u, &v)| (v.0, v.1, u.0, u.1))
        .collect::<Vec<_>>();
    rules.sort();

    for (_, id, pre, suc) in rules.iter().copied() {
        let e = |reason| Error::InvalidRule {
            pre,
            suc,
            id,
            reason,
        };
        let pre = vocab.get_token(pre).ok_or(e("left token out of bound"))?;
        let suc = vocab.get_token(suc).ok_or(e("right token out of bound"))?;
        let merged = vocab.get_token(id).ok_or(e("merged token out of bound"))?;
        let expected = Bytes::from_iter(pre.iter().copied().chain(suc.iter().copied()));
        if merged.as_ref() != expected {
            return Err(e("merged token does not match"));
        }
    }

    let dict =
        Dictionary::new_from_id_pair(vocab, rules.into_iter().map(|(_, _, pre, suc)| (pre, suc)))?;
    let normalized_dict = NormalizedDict::new(dict, |_, token_id, _| {
        singletons.contains(&token_id.inner())
    })?;

    Ok((normalized_dict, fallback_bytes))
}

impl IncrementalBpe {
    pub fn new(bpe: BPE, cache_capacity: Option<usize>) -> Result<Self, Error> {
        let cache_capacity = cache_capacity.unwrap_or(DEFAULT_CACHE_CAPACITY);
        let cache =
            HashCache::with_capacity_and_hasher(cache_capacity, cache_capacity, Default::default());

        let unk_token_id = bpe
            .unk_token
            .as_ref()
            .and_then(|t| bpe.vocab.get(t))
            .copied()
            .unwrap_or(u32::MAX);

        let mut fields = Vec::new();
        if bpe.dropout.is_some() {
            fields.push("dropout");
        }
        if bpe
            .continuing_subword_prefix
            .as_ref()
            .is_some_and(|s| !s.is_empty())
        {
            fields.push("continuing_subword_prefix");
        }
        if bpe
            .end_of_word_suffix
            .as_ref()
            .is_some_and(|s| !s.is_empty())
        {
            fields.push("end_of_word_suffix");
        }
        if !fields.is_empty() {
            return Err(Error::IncompatibleSettings { fields });
        }

        let (dict, fallback_bytes) = build_dict_from_bpe(&bpe)?;
        let tokenizer = IncBpeTokenizer::new(dict);

        Ok(Self {
            bpe,
            cache_disabled: cache_capacity == 0,
            inner: Arc::new(IncrementalBpeRef {
                tokenizer,
                unk_token_id,
                fallback_bytes,
                cache,
            }),
        })
    }
}

#[derive(Debug)]
struct WordSeq {
    id: TokenId,
    start: u32,
}

impl WordSeq {
    fn new(id: u32, start: usize) -> Self {
        Self {
            id: TokenId::new(id),
            start: start as _,
        }
    }
}

impl IncrementalBpe {
    fn split_text(&self, text: &str) -> Vec<WordSeq> {
        if self.bpe.ignore_merges {
            if let Some(&id) = self.bpe.vocab.get(text) {
                return vec![WordSeq::new(id, 0)];
            }
        }
        let mut res = Vec::with_capacity(text.len());
        let mut left = 0;
        for right in 1..text.len() + 1 {
            if !text.is_char_boundary(right) {
                continue;
            }
            let seq = &text[left..right];
            if let Some(&id) = self.bpe.vocab.get(seq) {
                res.push(WordSeq::new(id, left))
            } else if let Some(fallback_bytes) = &self.fallback_bytes {
                for pos in left..right {
                    res.push(WordSeq::new(
                        fallback_bytes[text.as_bytes()[pos] as usize],
                        pos,
                    ))
                }
            } else if !self.bpe.fuse_unk
                || res.last().is_none_or(|w| w.id.inner() != self.unk_token_id)
            {
                res.push(WordSeq::new(self.unk_token_id, left))
            }
            left = right;
        }
        res
    }

    fn token_res(&self, id: u32, left: usize, right: usize) -> crate::Token {
        crate::Token::new(id, self.bpe.vocab_r[&id].clone(), (left, right))
    }

    pub fn tokenize_without_cache(&self, sequence: &str) -> Vec<crate::Token> {
        let words = self.split_text(sequence);
        let mut state = self.tokenization();
        state.reserve(words.len());
        for word in &words {
            state.feed(word.id);
        }
        let mut res = Vec::with_capacity(words.len());
        let mut last_pos = sequence.len();
        for (idx, token) in state.current_token_chain() {
            let pos = words[(idx + 1).saturating_sub(token.skip_len as usize)].start as usize;
            res.push(self.token_res(token.token_id.inner(), pos, last_pos));
            last_pos = pos;
        }
        res.reverse();
        res
    }

    pub(crate) fn resize_cache(&mut self, capacity: usize) {
        self.cache_disabled = capacity == 0;
    }

    pub fn clear_cache(&self) {
        self.cache.clear_sync();
    }
}

impl Model for IncrementalBpe {
    type Trainer = BpeTrainer;

    fn tokenize(&self, sequence: &str) -> crate::Result<Vec<crate::Token>> {
        if sequence.is_empty() {
            return Ok(Default::default());
        }
        if self.cache_disabled || sequence.len() >= MAX_LENGTH {
            return Ok(self.tokenize_without_cache(sequence));
        }
        if let Some(seq) = self.cache.read_sync(sequence, |_, v| v.clone()) {
            return Ok(seq.iter().cloned().collect());
        }
        let seq = self.tokenize_without_cache(sequence);
        let _ = self
            .cache
            .put_sync(sequence.to_owned(), Arc::from(seq.as_slice()));
        Ok(seq)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.bpe.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.bpe.id_to_token(id)
    }

    fn get_vocab(&self) -> std::collections::HashMap<String, u32> {
        self.bpe.get_vocab()
    }

    fn get_vocab_size(&self) -> usize {
        self.bpe.get_vocab_size()
    }

    fn save(
        &self,
        folder: &std::path::Path,
        prefix: Option<&str>,
    ) -> crate::Result<Vec<std::path::PathBuf>> {
        self.bpe.save(folder, prefix)
    }

    fn get_trainer(&self) -> <Self as Model>::Trainer {
        self.bpe.get_trainer()
    }
}
