#![allow(dead_code)]

use anyhow::{Context, Result};
use arrow_array::{
    Array, ArrayRef, Float32Array, Int32Array, Int64Array, RecordBatch, RecordBatchIterator,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use futures::TryStreamExt;
use lance_arrow::FixedSizeListArrayExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Table, connect};
use std::path::PathBuf;
use std::sync::Arc;

use crate::config::Config;
use crate::episode::Episode;
use crate::store::EpisodeStore;

/// Embedding dimension for BGE-Small model
const EMBEDDING_DIM: usize = 384;

/// Episode indexer using LanceDB for vector search
pub struct EpisodeIndexer {
    db: Connection,
    table: Option<Table>,
    embedder: TextEmbedding,
}

impl EpisodeIndexer {
    /// Create a new episode indexer
    pub async fn new() -> Result<Self> {
        let db_path = Self::db_path()?;
        std::fs::create_dir_all(db_path.parent().unwrap())?;

        let db = connect(db_path.to_str().unwrap())
            .execute()
            .await
            .context("Failed to connect to LanceDB")?;

        // Initialize the embedding model
        println!("🔄 Loading embedding model (this may download the model on first run)...");
        let embedder = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(true),
        )
        .context("Failed to initialize embedding model")?;
        println!("✅ Embedding model loaded");

        // Try to open existing table
        let table = match db.open_table("episodes").execute().await {
            Ok(t) => Some(t),
            Err(_) => None, // Table doesn't exist yet
        };

        Ok(Self {
            db,
            table,
            embedder,
        })
    }

    /// Get the LanceDB database path
    fn db_path() -> Result<PathBuf> {
        let data_dir = Config::data_dir()?;
        Ok(data_dir.join("vectors").join("episodes.lance"))
    }

    /// Create the episodes table schema
    fn create_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("project", DataType::Utf8, false),
            Field::new("task_type", DataType::Utf8, false),
            Field::new("intent_text", DataType::Utf8, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new("utility_score", DataType::Float32, false),
            Field::new("retrieval_count", DataType::Int32, false),
            Field::new("helpful_count", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    EMBEDDING_DIM as i32,
                ),
                false,
            ),
        ]))
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
        }

        parts.join(" | ")
    }

    /// Create a record batch from an episode
    fn create_record_batch(&self, episode: &Episode) -> Result<RecordBatch> {
        let embedding_text = Self::episode_to_embedding_text(episode);
        let embedding = self.embed(&embedding_text)?;
        let schema = Self::create_schema();

        let id_array = Arc::new(StringArray::from(vec![episode.id.clone()])) as ArrayRef;
        let project_array = Arc::new(StringArray::from(vec![episode.project.clone()])) as ArrayRef;
        let task_type_array = Arc::new(StringArray::from(vec![
            episode.intent.task_type.to_string(),
        ])) as ArrayRef;
        let intent_text_array = Arc::new(StringArray::from(vec![embedding_text])) as ArrayRef;
        let timestamp_array =
            Arc::new(Int64Array::from(vec![episode.timestamp_start.timestamp()])) as ArrayRef;
        let utility_score_array =
            Arc::new(Float32Array::from(vec![episode.utility.calculate_score()])) as ArrayRef;
        let retrieval_count_array = Arc::new(Int32Array::from(vec![
            episode.utility.retrieval_count as i32,
        ])) as ArrayRef;
        let helpful_count_array =
            Arc::new(Int32Array::from(vec![episode.utility.helpful_count as i32])) as ArrayRef;

        // Create fixed size list for vector
        let vector_values = Arc::new(Float32Array::from(embedding)) as ArrayRef;
        let vector_array = Arc::new(arrow_array::FixedSizeListArray::try_new_from_values(
            vector_values,
            EMBEDDING_DIM as i32,
        )?) as ArrayRef;

        let batch = RecordBatch::try_new(
            schema,
            vec![
                id_array,
                project_array,
                task_type_array,
                intent_text_array,
                timestamp_array,
                utility_score_array,
                retrieval_count_array,
                helpful_count_array,
                vector_array,
            ],
        )?;

        Ok(batch)
    }

    /// Index a single episode
    pub async fn index_episode(&mut self, episode: &Episode) -> Result<()> {
        let batch = self.create_record_batch(episode)?;
        let schema = Self::create_schema();

        // Create or add to table
        if self.table.is_none() {
            let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
            let table = self
                .db
                .create_table("episodes", Box::new(batches))
                .execute()
                .await
                .context("Failed to create episodes table")?;
            self.table = Some(table);
        } else {
            let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
            self.table
                .as_ref()
                .unwrap()
                .add(Box::new(batches))
                .execute()
                .await
                .context("Failed to add episode to index")?;
        }

        Ok(())
    }

    /// Index all episodes from the store
    pub async fn index_all(&mut self, reindex: bool) -> Result<usize> {
        let store = EpisodeStore::new()?;
        let episodes = store.list_all()?;

        if episodes.is_empty() {
            return Ok(0);
        }

        // If reindexing, drop existing table
        if reindex && self.table.is_some() {
            self.db
                .drop_table("episodes", &[])
                .await
                .context("Failed to drop existing table")?;
            self.table = None;
        }

        // Get existing IDs if not reindexing
        let existing_ids: std::collections::HashSet<String> = if !reindex {
            self.get_all_ids().await.unwrap_or_default()
        } else {
            std::collections::HashSet::new()
        };

        let mut indexed = 0;
        let total = episodes.len();
        for episode in &episodes {
            // Skip if already indexed
            if existing_ids.contains(&episode.id) {
                continue;
            }

            self.index_episode(episode).await?;
            indexed += 1;
            print!("\r🔄 Indexed {}/{} episodes", indexed, total);
        }
        if indexed > 0 {
            println!();
        }

        // Create vector index if we have enough records
        if indexed > 0 {
            self.create_vector_index().await?;
        }

        Ok(indexed)
    }

    /// Get all indexed episode IDs
    async fn get_all_ids(&self) -> Result<std::collections::HashSet<String>> {
        let mut ids = std::collections::HashSet::new();

        if let Some(table) = &self.table {
            let results = table
                .query()
                .select(lancedb::query::Select::Columns(vec!["id".to_string()]))
                .execute()
                .await?
                .try_collect::<Vec<_>>()
                .await?;

            for batch in results {
                if let Some(id_col) = batch.column_by_name("id") {
                    if let Some(string_array) = id_col.as_any().downcast_ref::<StringArray>() {
                        for i in 0..string_array.len() {
                            if let Some(id) = string_array.value(i).into() {
                                ids.insert(id.to_string());
                            }
                        }
                    }
                }
            }
        }

        Ok(ids)
    }

    /// Create vector index for faster search
    async fn create_vector_index(&self) -> Result<()> {
        if let Some(table) = &self.table {
            // Only create index if we have enough records
            let count = table
                .count_rows(None)
                .await
                .context("Failed to count rows")?;

            if count >= 256 {
                // IVF-PQ requires minimum vectors
                println!("🔧 Creating vector index...");
                table
                    .create_index(&["vector"], lancedb::index::Index::Auto)
                    .execute()
                    .await
                    .context("Failed to create vector index")?;
                println!("✅ Vector index created");
            }
        }
        Ok(())
    }

    /// Search for similar episodes using vector similarity
    pub async fn search(
        &self,
        query: &str,
        limit: usize,
        project_filter: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let table = self
            .table
            .as_ref()
            .context("Index not initialized. Run 'memrl index' first.")?;

        // Generate query embedding
        let query_embedding = self.embed(query)?;

        // Build search query
        let mut search_builder = table.query().nearest_to(query_embedding)?;

        // Apply project filter if specified
        if let Some(project) = project_filter {
            search_builder = search_builder.only_if(format!("project = '{}'", project));
        }

        // Execute search
        let results = search_builder
            .limit(limit)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        // Parse results
        let mut search_results = Vec::new();
        for batch in results {
            let id_col = batch
                .column_by_name("id")
                .context("Missing id column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid id column type")?;

            let project_col = batch
                .column_by_name("project")
                .context("Missing project column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid project column type")?;

            let intent_col = batch
                .column_by_name("intent_text")
                .context("Missing intent_text column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid intent_text column type")?;

            let utility_col = batch
                .column_by_name("utility_score")
                .context("Missing utility_score column")?
                .as_any()
                .downcast_ref::<Float32Array>()
                .context("Invalid utility_score column type")?;

            // Get distance column (automatically added by LanceDB)
            let distance_col = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            for i in 0..batch.num_rows() {
                let distance = distance_col.map(|d| d.value(i)).unwrap_or(0.0);
                // Convert distance to similarity (LanceDB uses L2 distance by default)
                let similarity = 1.0 / (1.0 + distance);

                search_results.push(SearchResult {
                    id: id_col.value(i).to_string(),
                    project: project_col.value(i).to_string(),
                    intent_text: intent_col.value(i).to_string(),
                    similarity_score: similarity,
                    utility_score: utility_col.value(i),
                });
            }
        }

        Ok(search_results)
    }

    /// Check if the index exists and has data
    pub async fn is_indexed(&self) -> bool {
        if let Some(table) = &self.table {
            table.count_rows(None).await.unwrap_or(0) > 0
        } else {
            false
        }
    }

    /// Get index statistics
    pub async fn get_stats(&self) -> Result<IndexStats> {
        let count = if let Some(table) = &self.table {
            table.count_rows(None).await.unwrap_or(0)
        } else {
            0
        };

        Ok(IndexStats {
            total_indexed: count,
            embedding_dim: EMBEDDING_DIM,
            model_name: "BGE-Small-EN-v1.5".to_string(),
        })
    }

    /// Update utility scores in the index
    pub async fn update_utility(&self, episode_id: &str, utility_score: f32) -> Result<()> {
        if let Some(table) = &self.table {
            table
                .update()
                .only_if(format!("id = '{}'", episode_id))
                .column("utility_score", utility_score.to_string())
                .execute()
                .await
                .context("Failed to update utility score")?;
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
