// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub capture: CaptureConfig,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub retrieval: RetrievalConfig,
    #[serde(default)]
    pub bellman: BellmanConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub dream: DreamConfig,
}

/// Settings for the v0.7+ dream cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamConfig {
    /// Days of `Merged` time before `verify_advance` promotes an
    /// episode to `StableNoRevert`. 30 by default — about a full
    /// release cycle for most projects.
    #[serde(default = "default_stable_threshold_days")]
    pub stable_threshold_days: u32,
    /// Default dollar cap for a single `tempera dream` invocation.
    /// Overridable with `--max-usd`. Free phases ignore this.
    #[serde(default = "default_dream_max_usd")]
    pub default_max_usd: f32,
    /// Anthropic model used by `triage_day` (v0.7.2). Cheap + fast.
    #[serde(default = "default_triage_model")]
    pub triage_model: String,
    /// Anthropic model used by `author_reflection` (v0.7.3). Higher
    /// quality, more expensive (~$0.05/reflection vs ~$0.001/triage).
    #[serde(default = "default_reflect_model")]
    pub reflect_model: String,
    /// Max output tokens for a reflection body. 1500 ≈ ~1000 words.
    #[serde(default = "default_reflect_max_tokens")]
    pub reflect_max_tokens: u32,
    /// Days of recent reflections fed into the patterns phase (v0.7.4).
    #[serde(default = "default_patterns_lookback_days")]
    pub patterns_lookback_days: u32,
    /// Minimum cluster size before a pattern page gets authored.
    #[serde(default = "default_patterns_min_evidence")]
    pub patterns_min_evidence: usize,
    /// Cosine-similarity threshold for single-linkage clustering.
    /// 0.75 ≈ "clearly related theme"; lower = more permissive.
    #[serde(default = "default_patterns_cluster_threshold")]
    pub patterns_cluster_threshold: f32,
    /// Top N most-retrieved episodes considered for contradiction
    /// pairing (v0.7.5). Default 50.
    #[serde(default = "default_contradict_top_n")]
    pub contradict_top_n: usize,
    /// Pair similarity must fall in [low, high]: too low = unrelated,
    /// too high = near-duplicate (not actually contradictions).
    #[serde(default = "default_contradict_min_similarity")]
    pub contradict_min_similarity: f32,
    #[serde(default = "default_contradict_max_similarity")]
    pub contradict_max_similarity: f32,
    /// Maximum number of pairs to send to the judge per run.
    #[serde(default = "default_contradict_max_pairs")]
    pub contradict_max_pairs: usize,
    /// Probe is gated on the judge's self-reported confidence — verdicts
    /// below this don't get stored (treated as ambiguous).
    #[serde(default = "default_contradict_min_confidence")]
    pub contradict_min_confidence: f32,
    /// Minimum cluster size before a reasoning template gets authored
    /// (v0.8.3). 3 = "show the pattern at least three times before I
    /// trust it as repeatable."
    #[serde(default = "default_templates_min_evidence")]
    pub templates_min_evidence: usize,
    /// Minimum verification weight for an episode to qualify as
    /// template evidence. 0.30 = `Untested` (accept everything that's
    /// at least a self-declared success); 0.60 = `Merged` (only count
    /// what survived review). Starts lenient — tighten as verification
    /// adoption grows.
    #[serde(default = "default_templates_min_verification_weight")]
    pub templates_min_verification_weight: f32,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            stable_threshold_days: default_stable_threshold_days(),
            default_max_usd: default_dream_max_usd(),
            triage_model: default_triage_model(),
            reflect_model: default_reflect_model(),
            reflect_max_tokens: default_reflect_max_tokens(),
            patterns_lookback_days: default_patterns_lookback_days(),
            patterns_min_evidence: default_patterns_min_evidence(),
            patterns_cluster_threshold: default_patterns_cluster_threshold(),
            contradict_top_n: default_contradict_top_n(),
            contradict_min_similarity: default_contradict_min_similarity(),
            contradict_max_similarity: default_contradict_max_similarity(),
            contradict_max_pairs: default_contradict_max_pairs(),
            contradict_min_confidence: default_contradict_min_confidence(),
            templates_min_evidence: default_templates_min_evidence(),
            templates_min_verification_weight: default_templates_min_verification_weight(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureConfig {
    #[serde(default = "default_true")]
    pub auto_capture: bool,
    #[serde(default = "default_true")]
    pub extract_intent_llm: bool,
    #[serde(default = "default_true")]
    pub capture_diffs: bool,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            auto_capture: true,
            extract_intent_llm: true,
            capture_diffs: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    #[serde(default = "default_embedding_model")]
    pub model: String,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: default_embedding_model(),
            batch_size: default_batch_size(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RetrievalMode {
    /// Vector (semantic) search only — pre-v0.4.2 behavior.
    Vector,
    /// BM25 keyword search only.
    Keyword,
    /// Both, fused via reciprocal-rank fusion (v0.4.2+ default).
    Hybrid,
}

impl Default for RetrievalMode {
    fn default() -> Self {
        Self::Hybrid
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    #[serde(default = "default_limit")]
    pub default_limit: usize,
    /// Which retrieval pipeline to use. "vector" | "keyword" | "hybrid".
    #[serde(default)]
    pub mode: RetrievalMode,
    /// RRF constant `k` — higher values smooth out ranking differences between
    /// the two input rankings. Lucene/Vespa default is 60.
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f32,
    /// In hybrid mode, the RRF-normalized score replaces cosine similarity in
    /// the ranking blend. Its distribution is compressed (top doc = 1.0,
    /// second often ≈ 0.5), so the vector-mode weights would let utility
    /// dominate. Use a more similarity-biased default here.
    #[serde(default = "default_hybrid_similarity_weight")]
    pub hybrid_similarity_weight: f32,
    #[serde(default = "default_hybrid_utility_weight")]
    pub hybrid_utility_weight: f32,
    #[serde(default = "default_similarity_weight")]
    pub similarity_weight: f32,
    #[serde(default = "default_utility_weight")]
    pub utility_weight: f32,
    #[serde(default = "default_min_similarity")]
    pub min_similarity: f32,
    /// MMR lambda: 0.0 = pure diversity, 1.0 = pure relevance
    #[serde(default = "default_mmr_lambda")]
    pub mmr_lambda: f32,
    /// Weight for recency in scoring (0.0 = off, opt-in).
    /// **Deprecated in v0.7.7** — recency is now folded into the
    /// `salience` term. Kept for backward-compat deserialization;
    /// new code reads `salience_weight` instead. Removing this field
    /// would break old config.toml files.
    #[serde(default = "default_recency_weight")]
    pub recency_weight: f32,
    /// Half-life for recency decay in days. Still used by `salience_score`.
    #[serde(default = "default_recency_halflife_days")]
    pub recency_halflife_days: f32,
    /// v0.7.7: weight of the unified `salience` term in `combined_score`.
    /// Salience = utility × verification × recency × inv_freq. Default
    /// 0.7 keeps the same blend ratio the old `utility_weight = 0.7`
    /// + `similarity_weight = 0.3` combination produced.
    #[serde(default = "default_salience_weight")]
    pub salience_weight: f32,
    /// v0.7.7: denominator for the inv_freq factor — at
    /// `retrieval_history.len() == salience_freq_normalizer`, the
    /// inv_freq factor is 0.5. Smaller values penalize heavy retrieval
    /// harder.
    #[serde(default = "default_salience_freq_normalizer")]
    pub salience_freq_normalizer: f32,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            default_limit: default_limit(),
            mode: RetrievalMode::default(),
            rrf_k: default_rrf_k(),
            hybrid_similarity_weight: default_hybrid_similarity_weight(),
            hybrid_utility_weight: default_hybrid_utility_weight(),
            similarity_weight: default_similarity_weight(),
            utility_weight: default_utility_weight(),
            min_similarity: default_min_similarity(),
            mmr_lambda: default_mmr_lambda(),
            recency_weight: default_recency_weight(),
            recency_halflife_days: default_recency_halflife_days(),
            salience_weight: default_salience_weight(),
            salience_freq_normalizer: default_salience_freq_normalizer(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BellmanConfig {
    #[serde(default = "default_gamma")]
    pub gamma: f32,
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    #[serde(default = "default_propagate_interval")]
    pub propagate_interval: String,
    /// Decay rate per day for unused episodes (0.0 - 1.0)
    #[serde(default = "default_decay_rate")]
    pub decay_rate: f64,
    /// Minimum similarity threshold for Bellman propagation
    #[serde(default = "default_propagation_threshold")]
    pub propagation_threshold: f32,
    /// Maximum propagation depth (hops)
    #[serde(default = "default_max_propagation_depth")]
    pub max_propagation_depth: u32,
    /// Temporal credit lookback window in hours
    #[serde(default = "default_temporal_credit_window_hours")]
    pub temporal_credit_window_hours: i64,
    /// v0.6.2: only episodes whose Claim has `falsifiability >= this` may
    /// propagate utility through the similarity graph. Episodes below the
    /// threshold are still stored and retrievable — they just don't shape
    /// the ranking surface for their neighbors.
    #[serde(default = "default_min_falsifiability")]
    pub min_falsifiability: f32,
    /// If true, episodes without any Claim at all (older files, or captures
    /// where LLM extraction was skipped) are also excluded from propagation.
    /// Default `false` is lenient — pre-v0.6.2 episodes keep working.
    #[serde(default)]
    pub require_claim_for_propagation: bool,
}

impl Default for BellmanConfig {
    fn default() -> Self {
        Self {
            gamma: default_gamma(),
            alpha: default_alpha(),
            propagate_interval: default_propagate_interval(),
            decay_rate: default_decay_rate(),
            propagation_threshold: default_propagation_threshold(),
            max_propagation_depth: default_max_propagation_depth(),
            temporal_credit_window_hours: default_temporal_credit_window_hours(),
            min_falsifiability: default_min_falsifiability(),
            require_claim_for_propagation: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    #[serde(default = "default_max_age_days")]
    pub max_age_days: u32,
    #[serde(default = "default_min_utility_threshold")]
    pub min_utility_threshold: f32,
    #[serde(default = "default_min_retrievals")]
    pub min_retrievals: u32,
    /// Similarity threshold for BKM consolidation during capture
    #[serde(default = "default_consolidation_threshold")]
    pub consolidation_threshold: f32,
    /// Similarity threshold for duplicate clustering during review
    #[serde(default = "default_cluster_threshold")]
    pub cluster_threshold: f32,
    /// Episodes older than this (days) with low utility are considered stale
    #[serde(default = "default_stale_age_days")]
    pub stale_age_days: u32,
    /// Utility below this marks an episode as stale (when also old enough)
    #[serde(default = "default_stale_utility_threshold")]
    pub stale_utility_threshold: f32,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            max_age_days: default_max_age_days(),
            min_utility_threshold: default_min_utility_threshold(),
            min_retrievals: default_min_retrievals(),
            consolidation_threshold: default_consolidation_threshold(),
            cluster_threshold: default_cluster_threshold(),
            stale_age_days: default_stale_age_days(),
            stale_utility_threshold: default_stale_utility_threshold(),
        }
    }
}

// Default value functions
fn default_true() -> bool {
    true
}

fn default_embedding_model() -> String {
    "bge-small-en-v1.5".to_string()
}

fn default_batch_size() -> usize {
    32
}

fn default_limit() -> usize {
    3
}

fn default_similarity_weight() -> f32 {
    0.3
}

fn default_utility_weight() -> f32 {
    0.7
}

fn default_min_similarity() -> f32 {
    0.5
}

fn default_gamma() -> f32 {
    0.9
}

fn default_alpha() -> f32 {
    0.1
}

fn default_propagate_interval() -> String {
    "daily".to_string()
}

fn default_max_age_days() -> u32 {
    180
}

fn default_min_utility_threshold() -> f32 {
    0.05
}

fn default_min_retrievals() -> u32 {
    2
}

fn default_mmr_lambda() -> f32 {
    0.7
}

fn default_recency_weight() -> f32 {
    0.0 // Off by default — opt-in
}

fn default_recency_halflife_days() -> f32 {
    30.0
}

fn default_salience_weight() -> f32 {
    // Matches the pre-v0.7.7 utility_weight default. Combined with
    // similarity_weight=0.3 this preserves the historical 70/30 blend.
    0.7
}

fn default_salience_freq_normalizer() -> f32 {
    // At retrieval_count == 100, inv_freq drops to 0.5 (a 50% rank
    // damping for very heavily retrieved episodes).
    100.0
}

fn default_rrf_k() -> f32 {
    60.0
}

fn default_hybrid_similarity_weight() -> f32 {
    0.85
}

fn default_hybrid_utility_weight() -> f32 {
    0.15
}

fn default_decay_rate() -> f64 {
    0.01
}

fn default_propagation_threshold() -> f32 {
    0.5
}

fn default_max_propagation_depth() -> u32 {
    2
}

fn default_temporal_credit_window_hours() -> i64 {
    1
}

fn default_min_falsifiability() -> f32 {
    0.7
}

fn default_stable_threshold_days() -> u32 {
    30
}

fn default_dream_max_usd() -> f32 {
    // v0.7.1 phases are free; this default exists for v0.7.2 onward.
    0.50
}

fn default_triage_model() -> String {
    // Haiku 4.5: cheap + fast, perfect for gating.
    "claude-haiku-4-5-20251001".to_string()
}

fn default_reflect_model() -> String {
    // Sonnet 4.6: highest quality / cost ratio for authorship.
    "claude-sonnet-4-6".to_string()
}

fn default_reflect_max_tokens() -> u32 {
    1500
}

fn default_patterns_lookback_days() -> u32 {
    30
}

fn default_patterns_min_evidence() -> usize {
    3
}

fn default_patterns_cluster_threshold() -> f32 {
    0.75
}

fn default_contradict_top_n() -> usize {
    50
}

fn default_contradict_min_similarity() -> f32 {
    0.6
}

fn default_contradict_max_similarity() -> f32 {
    0.95
}

fn default_contradict_max_pairs() -> usize {
    30
}

fn default_contradict_min_confidence() -> f32 {
    0.7
}

fn default_templates_min_evidence() -> usize {
    3
}

fn default_templates_min_verification_weight() -> f32 {
    // 0.30 = Untested. Lenient default so templates kick in early.
    // Bump to 0.60 (Merged) when verify_advance is widely adopted.
    0.30
}

fn default_consolidation_threshold() -> f32 {
    0.85
}

fn default_cluster_threshold() -> f32 {
    0.85
}

fn default_stale_age_days() -> u32 {
    30
}

fn default_stale_utility_threshold() -> f32 {
    0.2
}

impl Default for Config {
    fn default() -> Self {
        Self {
            capture: CaptureConfig::default(),
            embedding: EmbeddingConfig::default(),
            retrieval: RetrievalConfig::default(),
            bellman: BellmanConfig::default(),
            storage: StorageConfig::default(),
            dream: DreamConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from ~/.tempera/config.toml
    /// Falls back to defaults if file doesn't exist
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;

        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read config from {}", config_path.display()))?;
            let config: Config =
                toml::from_str(&content).with_context(|| "Failed to parse config.toml")?;
            Ok(config)
        } else {
            // Return default config if file doesn't exist
            Ok(Config::default())
        }
    }

    /// Save configuration to ~/.tempera/config.toml
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;

        // Ensure parent directory exists
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = toml::to_string_pretty(self)?;
        std::fs::write(&config_path, content)
            .with_context(|| format!("Failed to write config to {}", config_path.display()))?;

        Ok(())
    }

    /// Get the path to the config file
    pub fn config_path() -> Result<PathBuf> {
        let home = dirs::home_dir().context("Could not find home directory")?;
        Ok(home.join(".tempera").join("config.toml"))
    }

    /// Get the tempera data directory (~/.tempera)
    pub fn data_dir() -> Result<PathBuf> {
        let home = dirs::home_dir().context("Could not find home directory")?;
        Ok(home.join(".tempera"))
    }

    /// Get the episodes directory (~/.tempera/episodes)
    pub fn episodes_dir() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("episodes"))
    }

    /// Get the database path (~/.tempera/tempera.db)
    pub fn database_path() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("tempera.db"))
    }

    /// Get the feedback log path (~/.tempera/feedback.log)
    pub fn feedback_log_path() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("feedback.log"))
    }

    /// Get today's episode directory
    pub fn today_episodes_dir() -> Result<PathBuf> {
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        Ok(Self::episodes_dir()?.join(today))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.capture.auto_capture);
        assert_eq!(config.retrieval.default_limit, 3);
        assert_eq!(config.bellman.gamma, 0.9);
        // New fields have correct defaults
        assert_eq!(config.retrieval.mmr_lambda, 0.7);
        assert!((config.bellman.decay_rate - 0.01).abs() < f64::EPSILON);
        assert_eq!(config.bellman.propagation_threshold, 0.5);
        assert_eq!(config.bellman.max_propagation_depth, 2);
        assert_eq!(config.bellman.temporal_credit_window_hours, 1);
        assert_eq!(config.storage.consolidation_threshold, 0.85);
        assert_eq!(config.storage.cluster_threshold, 0.85);
        assert_eq!(config.storage.stale_age_days, 30);
        assert_eq!(config.storage.stale_utility_threshold, 0.2);
        // Recency defaults
        assert_eq!(config.retrieval.recency_weight, 0.0);
        assert_eq!(config.retrieval.recency_halflife_days, 30.0);
        // Hybrid retrieval defaults
        assert_eq!(config.retrieval.mode, RetrievalMode::Hybrid);
        assert_eq!(config.retrieval.rrf_k, 60.0);
    }

    #[test]
    fn test_retrieval_mode_serialization() {
        // Make sure lowercase tags round-trip
        for &m in &[
            RetrievalMode::Vector,
            RetrievalMode::Keyword,
            RetrievalMode::Hybrid,
        ] {
            let json = serde_json::to_string(&m).unwrap();
            let parsed: RetrievalMode = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, m);
        }
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(
            config.retrieval.default_limit,
            parsed.retrieval.default_limit
        );
        assert_eq!(config.retrieval.mmr_lambda, parsed.retrieval.mmr_lambda);
        assert!((config.bellman.decay_rate - parsed.bellman.decay_rate).abs() < f64::EPSILON);
        assert_eq!(
            config.storage.consolidation_threshold,
            parsed.storage.consolidation_threshold
        );
    }

    #[test]
    fn test_config_backward_compat() {
        // Old config without new fields should deserialize with defaults
        let old_toml = r#"
[capture]
auto_capture = true

[retrieval]
default_limit = 5

[bellman]
gamma = 0.8

[storage]
max_age_days = 90
"#;
        let config: Config = toml::from_str(old_toml).unwrap();
        assert_eq!(config.retrieval.default_limit, 5);
        assert_eq!(config.bellman.gamma, 0.8);
        // New fields get defaults
        assert_eq!(config.retrieval.mmr_lambda, 0.7);
        assert!((config.bellman.decay_rate - 0.01).abs() < f64::EPSILON);
        assert_eq!(config.storage.consolidation_threshold, 0.85);
        assert_eq!(config.storage.stale_age_days, 30);
    }
}
