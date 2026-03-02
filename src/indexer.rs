#![allow(dead_code)]

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::collections::HashSet;
use std::path::PathBuf;
use uuid::Uuid;
use vectrust::{CreateIndexConfig, DistanceMetric, LocalIndex, UpdateRequest, VectorItem};

use crate::config::Config;
use crate::episode::Episode;
use crate::store::EpisodeStore;

/// Embedding dimension for BGE-Small model
const EMBEDDING_DIM: usize = 384;

/// Episode indexer using vectrust for vector search.
///
/// Uses on-demand open/close pattern: the embedder is cached (expensive to load)
/// but the vector index is opened fresh per operation and released when done.
/// This allows multiple MCP server instances to share the same database.
pub struct EpisodeIndexer {
    embedder: TextEmbedding,
    index_path: PathBuf,
}

impl EpisodeIndexer {
    /// Create a new episode indexer
    pub async fn new() -> Result<Self> {
        let index_path = Self::db_path()?;
        std::fs::create_dir_all(&index_path)?;

        // Initialize the embedding model with global cache directory
        // MUST set FASTEMBED_CACHE_DIR before InitOptions::new() — its Default impl
        // calls get_cache_dir() which falls back to ".fastembed_cache" in CWD.
        // Note: the env var is FASTEMBED_CACHE_DIR (not _PATH).
        let cache_dir = Self::model_cache_path()?;
        std::fs::create_dir_all(&cache_dir)?;
        unsafe { std::env::set_var("FASTEMBED_CACHE_DIR", &cache_dir) };

        println!("Loading embedding model (this may download the model on first run)...");
        let embedder = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15)
                .with_cache_dir(cache_dir)
                .with_show_download_progress(true),
        )
        .context("Failed to initialize embedding model")?;
        println!("Embedding model loaded");

        Ok(Self {
            embedder,
            index_path,
        })
    }

    /// Open a fresh vectrust index for an operation.
    /// The index is dropped when it goes out of scope, releasing the RocksDB lock.
    async fn open_index(&self) -> Result<LocalIndex> {
        let index = LocalIndex::new(&self.index_path, Some("episodes".into()))
            .context("Failed to open vector index")?;

        if !index.is_index_created().await {
            index
                .create_index(Some(CreateIndexConfig {
                    distance_metric: DistanceMetric::Cosine,
                    ..Default::default()
                }))
                .await
                .context("Failed to create vector index")?;
        }

        Ok(index)
    }

    /// Get the vector database path
    fn db_path() -> Result<PathBuf> {
        let data_dir = Config::data_dir()?;
        Ok(data_dir.join("vectors"))
    }

    /// Get the global model cache path (~/.tempera/models/)
    fn model_cache_path() -> Result<PathBuf> {
        let data_dir = Config::data_dir()?;
        Ok(data_dir.join("models"))
    }

    /// Generate embedding for text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self
            .embedder
            .embed(vec![text.to_string()], None)
            .context("Failed to generate embedding")?;

        embeddings
            .into_iter()
            .next()
            .context("No embedding generated")
    }

    /// Create embedding text from an episode
    fn episode_to_embedding_text(episode: &Episode) -> String {
        let mut parts = Vec::new();

        // Intent information
        if !episode.intent.raw_prompt.is_empty() {
            parts.push(episode.intent.raw_prompt.clone());
        }
        if !episode.intent.extracted_intent.is_empty() {
            parts.push(episode.intent.extracted_intent.clone());
        }

        // Task type
        parts.push(format!("task type: {}", episode.intent.task_type));

        // Domain tags
        if !episode.intent.domain.is_empty() {
            parts.push(format!("tags: {}", episode.intent.domain.join(", ")));
        }

        // Files modified
        if !episode.context.files_modified.is_empty() {
            parts.push(format!(
                "files: {}",
                episode.context.files_modified.join(", ")
            ));
        }

        // Tools used
        if !episode.context.tools_invoked.is_empty() {
            parts.push(format!(
                "tools: {}",
                episode.context.tools_invoked.join(", ")
            ));
        }

        // Errors encountered
        if !episode.context.errors_encountered.is_empty() {
            let errors: Vec<_> = episode
                .context
                .errors_encountered
                .iter()
                .map(|e| e.message.clone())
                .collect();
            parts.push(format!("errors: {}", errors.join(", ")));

            // Include resolutions for better "how did I fix X?" queries
            let resolutions: Vec<_> = episode
                .context
                .errors_encountered
                .iter()
                .filter_map(|e| e.resolution.as_ref())
                .cloned()
                .collect();
            if !resolutions.is_empty() {
                parts.push(format!("resolutions: {}", resolutions.join(", ")));
            }
        }

        // Include outcome status for success/failure pattern matching
        parts.push(format!("outcome: {}", episode.outcome.status));

        parts.join(" | ")
    }

    /// Build a VectorItem from an episode
    fn episode_to_vector_item(&self, episode: &Episode) -> Result<VectorItem> {
        let embedding_text = Self::episode_to_embedding_text(episode);
        let embedding = self.embed(&embedding_text)?;

        let id = Uuid::parse_str(&episode.id).unwrap_or_else(|_| Uuid::new_v4());

        let metadata = serde_json::json!({
            "episode_id": episode.id,
            "project": episode.project,
            "task_type": episode.intent.task_type.to_string(),
            "intent_text": embedding_text,
            "timestamp": episode.timestamp_start.timestamp(),
            "utility_score": episode.utility.calculate_score(),
            "retrieval_count": episode.utility.retrieval_count,
            "helpful_count": episode.utility.helpful_count,
        });

        Ok(VectorItem {
            id,
            vector: embedding,
            metadata,
            ..Default::default()
        })
    }

    /// Index a single episode (upsert: delete existing then insert)
    pub async fn index_episode(&mut self, episode: &Episode) -> Result<()> {
        let item = self.episode_to_vector_item(episode)?;
        let index = self.open_index().await?;

        index.begin_update().await?;

        // Delete existing entry if present (upsert behavior)
        let _ = index.delete_item(&item.id).await;

        let _inserted: VectorItem = index
            .insert_item(item)
            .await
            .context("Failed to insert episode")?;

        index.end_update().await?;
        Ok(())
    }

    /// Index all episodes from the store
    pub async fn index_all(&mut self, reindex: bool) -> Result<usize> {
        let store = EpisodeStore::new()?;
        let episodes = store.list_all()?;

        if episodes.is_empty() {
            return Ok(0);
        }

        // If reindexing, delete and recreate the index
        if reindex {
            let index = self.open_index().await?;
            let _ = index.delete_index().await;
            drop(index);
        }

        // Open a fresh index for the batch operation
        let index = self.open_index().await?;

        // Get existing IDs if not reindexing
        let existing_ids: HashSet<String> = if !reindex {
            let items = index.list_items(None).await.unwrap_or_default();
            items
                .iter()
                .filter_map(|item| {
                    item.metadata
                        .get("episode_id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                })
                .collect()
        } else {
            HashSet::new()
        };

        let mut indexed = 0;
        let total = episodes.len();
        let mut batch = Vec::new();

        for episode in &episodes {
            if existing_ids.contains(&episode.id) {
                continue;
            }

            let item = self.episode_to_vector_item(episode)?;
            batch.push(item);
            indexed += 1;
            print!("\rIndexed {}/{} episodes", indexed, total);
        }

        if !batch.is_empty() {
            index.begin_update().await?;
            index
                .insert_items(batch)
                .await
                .context("Failed to insert episodes")?;
            index.end_update().await?;
            println!();
        }

        Ok(indexed)
    }

    /// Search for similar episodes using vector similarity
    pub async fn search(
        &self,
        query: &str,
        limit: usize,
        project_filter: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let index = self.open_index().await?;

        // Generate query embedding
        let query_embedding = self.embed(query)?;

        // Over-fetch to account for post-filtering by project
        let fetch_limit = if project_filter.is_some() {
            limit * 3
        } else {
            limit
        };

        let results = index
            .query_items(query_embedding, Some(fetch_limit as u32), None)
            .await
            .context("Failed to search vector index")?;

        let mut search_results = Vec::new();
        for result in results {
            let meta = &result.item.metadata;

            let episode_id = meta
                .get("episode_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            let project = meta
                .get("project")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            // Post-filter by project
            if let Some(filter_project) = project_filter {
                if project != filter_project {
                    continue;
                }
            }

            let intent_text = meta
                .get("intent_text")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            let utility_score = meta
                .get("utility_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;

            search_results.push(SearchResult {
                id: episode_id,
                project,
                intent_text,
                similarity_score: result.score,
                utility_score,
            });

            if search_results.len() >= limit {
                break;
            }
        }

        Ok(search_results)
    }

    /// Check if the index exists and has data
    pub async fn is_indexed(&self) -> bool {
        if let Ok(index) = self.open_index().await {
            if let Ok(stats) = index.get_stats().await {
                return stats.items > 0;
            }
        }
        false
    }

    /// Get index statistics
    pub async fn get_stats(&self) -> Result<IndexStats> {
        let index = self.open_index().await?;
        let stats = index
            .get_stats()
            .await
            .context("Failed to get index stats")?;

        Ok(IndexStats {
            total_indexed: stats.items,
            embedding_dim: EMBEDDING_DIM,
            model_name: "BGE-Small-EN-v1.5".to_string(),
        })
    }

    /// Update utility scores in the index
    pub async fn update_utility(&self, episode_id: &str, utility_score: f32) -> Result<()> {
        let index = self.open_index().await?;

        // Find the item by episode_id in metadata
        let items = index.list_items(None).await.unwrap_or_default();
        for item in items {
            if item.metadata.get("episode_id").and_then(|v| v.as_str()) == Some(episode_id) {
                let mut new_metadata = item.metadata.clone();
                new_metadata["utility_score"] = serde_json::json!(utility_score);

                index.begin_update().await?;
                index
                    .update_item(UpdateRequest {
                        id: item.id,
                        vector: None,
                        metadata: Some(new_metadata),
                    })
                    .await
                    .context("Failed to update utility score")?;
                index.end_update().await?;
                break;
            }
        }

        Ok(())
    }
}

/// Search result from vector search
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub project: String,
    pub intent_text: String,
    pub similarity_score: f32,
    pub utility_score: f32,
}

/// Index statistics
#[derive(Debug)]
pub struct IndexStats {
    pub total_indexed: usize,
    pub embedding_dim: usize,
    pub model_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_episode_to_embedding_text() {
        let episode = Episode::new("test-project".to_string(), "fix the login bug".to_string());
        let text = EpisodeIndexer::episode_to_embedding_text(&episode);
        assert!(text.contains("fix the login bug"));
    }
}
