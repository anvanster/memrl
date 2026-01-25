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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    #[serde(default = "default_limit")]
    pub default_limit: usize,
    #[serde(default = "default_similarity_weight")]
    pub similarity_weight: f32,
    #[serde(default = "default_utility_weight")]
    pub utility_weight: f32,
    #[serde(default = "default_min_similarity")]
    pub min_similarity: f32,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            default_limit: default_limit(),
            similarity_weight: default_similarity_weight(),
            utility_weight: default_utility_weight(),
            min_similarity: default_min_similarity(),
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
}

impl Default for BellmanConfig {
    fn default() -> Self {
        Self {
            gamma: default_gamma(),
            alpha: default_alpha(),
            propagate_interval: default_propagate_interval(),
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
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            max_age_days: default_max_age_days(),
            min_utility_threshold: default_min_utility_threshold(),
            min_retrievals: default_min_retrievals(),
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

impl Default for Config {
    fn default() -> Self {
        Self {
            capture: CaptureConfig::default(),
            embedding: EmbeddingConfig::default(),
            retrieval: RetrievalConfig::default(),
            bellman: BellmanConfig::default(),
            storage: StorageConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from ~/.memrl/config.toml
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

    /// Save configuration to ~/.memrl/config.toml
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
        Ok(home.join(".memrl").join("config.toml"))
    }

    /// Get the memrl data directory (~/.memrl)
    pub fn data_dir() -> Result<PathBuf> {
        let home = dirs::home_dir().context("Could not find home directory")?;
        Ok(home.join(".memrl"))
    }

    /// Get the episodes directory (~/.memrl/episodes)
    pub fn episodes_dir() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("episodes"))
    }

    /// Get the database path (~/.memrl/memrl.db)
    pub fn database_path() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("memrl.db"))
    }

    /// Get the feedback log path (~/.memrl/feedback.log)
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
    }
}
