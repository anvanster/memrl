#![allow(dead_code)]
use anyhow::Result;
use chrono::Utc;
use colored::Colorize;
use std::io::Write;

use crate::config::Config;
use crate::episode::{Episode, RetrievalRecord};
use crate::indexer::EpisodeIndexer;
use crate::store::EpisodeStore;

/// Run the retrieve command
pub async fn run(
    query: &str,
    limit: usize,
    project: Option<String>,
    format: &str,
    config: &Config,
) -> Result<()> {
    let store = EpisodeStore::new()?;

    // Try vector search first if index exists
    let episodes = match try_vector_search(query, limit, project.as_deref(), config).await {
        Ok(results) if !results.is_empty() => {
            println!("🔍 Using semantic vector search...\n");
            results
        }
        _ => {
            println!("🔍 Using text-based search (run 'memrl index' for semantic search)...\n");
            retrieve_episodes_text(query, limit, project.as_deref(), config, &store)?
        }
    };

    if episodes.is_empty() {
        println!("No relevant episodes found.");
        return Ok(());
    }

    // Display results based on format
    match format {
        "json" => {
            let json = serde_json::to_string_pretty(&episodes)?;
            println!("{}", json);
        }
        _ => {
            // Default: markdown format
            print_markdown_results(&episodes, query);
        }
    }

    // Record retrieval for utility tracking
    record_retrievals(&episodes, query, &store)?;

    Ok(())
}

/// Try to retrieve episodes using vector search
async fn try_vector_search(
    query: &str,
    limit: usize,
    project_filter: Option<&str>,
    config: &Config,
) -> Result<Vec<ScoredEpisode>> {
    let indexer = EpisodeIndexer::new().await?;

    if !indexer.is_indexed().await {
        anyhow::bail!("Index not available");
    }

    let store = EpisodeStore::new()?;
    let search_results = indexer.search(query, limit * 2, project_filter).await?;

    // Convert search results to scored episodes
    let mut episodes = Vec::new();
    for result in search_results {
        if let Ok(episode) = store.load(&result.id) {
            let utility = episode.utility.calculate_score();

            // Combine similarity and utility scores
            let combined = (1.0 - config.retrieval.utility_weight) * result.similarity_score
                + config.retrieval.utility_weight * utility;

            episodes.push(ScoredEpisode {
                episode,
                similarity_score: result.similarity_score,
                utility_score: utility,
                combined_score: combined,
            });
        }
    }

    // Sort by combined score
    episodes.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Filter by minimum similarity and take top N
    episodes.retain(|e| e.similarity_score >= config.retrieval.min_similarity);
    episodes.truncate(limit);

    Ok(episodes)
}

/// Retrieve relevant episodes using text-based search (fallback)
pub fn retrieve_episodes_text(
    query: &str,
    limit: usize,
    project_filter: Option<&str>,
    config: &Config,
    store: &EpisodeStore,
) -> Result<Vec<ScoredEpisode>> {
    let all_episodes = store.list_all()?;

    // Score and rank episodes
    let mut scored: Vec<ScoredEpisode> = all_episodes
        .into_iter()
        .filter(|ep| {
            // Filter by project if specified
            if let Some(proj) = project_filter {
                ep.project.to_lowercase().contains(&proj.to_lowercase())
            } else {
                true
            }
        })
        .map(|ep| {
            let similarity = calculate_text_similarity(query, &ep);
            let utility = ep.utility.calculate_score();

            // Combine similarity and utility scores
            let combined = (1.0 - config.retrieval.utility_weight) * similarity
                + config.retrieval.utility_weight * utility;

            ScoredEpisode {
                episode: ep,
                similarity_score: similarity,
                utility_score: utility,
                combined_score: combined,
            }
        })
        .filter(|se| se.similarity_score >= config.retrieval.min_similarity)
        .collect();

    // Sort by combined score (descending)
    scored.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top N
    scored.truncate(limit);

    Ok(scored)
}

/// Calculate text-based similarity between query and episode
fn calculate_text_similarity(query: &str, episode: &Episode) -> f32 {
    let query_lower = query.to_lowercase();
    let query_words: Vec<&str> = query_lower.split_whitespace().collect();

    // Combine episode text for matching
    let episode_text = format!(
        "{} {} {} {}",
        episode.intent.raw_prompt.to_lowercase(),
        episode.intent.extracted_intent.to_lowercase(),
        episode.intent.domain.join(" ").to_lowercase(),
        episode.context.files_modified.join(" ").to_lowercase()
    );

    // Count matching words
    let matches = query_words
        .iter()
        .filter(|word| episode_text.contains(*word))
        .count();

    if query_words.is_empty() {
        return 0.0;
    }

    // Jaccard-like similarity
    let episode_words: Vec<&str> = episode_text.split_whitespace().collect();
    let total_unique = query_words.len() + episode_words.len() - matches;

    if total_unique == 0 {
        0.0
    } else {
        matches as f32 / total_unique as f32
    }
}

/// Print results in markdown format
fn print_markdown_results(episodes: &[ScoredEpisode], query: &str) {
    println!("{}", "## Relevant Past Experiences".bold());
    println!();
    println!("Query: {}", query.italic());
    println!();

    for (i, scored) in episodes.iter().enumerate() {
        let ep = &scored.episode;

        println!(
            "### {}. {}",
            i + 1,
            if ep.intent.extracted_intent.is_empty() {
                &ep.intent.raw_prompt
            } else {
                &ep.intent.extracted_intent
            }
        );

        println!(
            "**When**: {}",
            ep.timestamp_start.format("%Y-%m-%d %H:%M UTC")
        );
        println!("**Project**: {}", ep.project);
        println!("**Outcome**: {}", ep.outcome.status);
        println!(
            "**Relevance**: {:.0}% similarity, {:.0}% utility",
            scored.similarity_score * 100.0,
            scored.utility_score * 100.0
        );

        // Key insight from the episode
        if !ep.context.files_modified.is_empty() {
            println!(
                "**Files involved**: {}",
                ep.context.files_modified.join(", ")
            );
        }

        if !ep.intent.domain.is_empty() {
            println!("**Tags**: {}", ep.intent.domain.join(", "));
        }

        // Show errors if any were resolved
        let resolved_errors: Vec<_> = ep
            .context
            .errors_encountered
            .iter()
            .filter(|e| e.resolved)
            .collect();
        if !resolved_errors.is_empty() {
            println!("**Errors resolved**:");
            for err in resolved_errors.iter().take(3) {
                println!("  - {}", err.message);
            }
        }

        println!();
    }

    println!("{}", "---".dimmed());
    println!(
        "{}",
        "To provide feedback: memrl feedback helpful --episodes <id>,<id>".dimmed()
    );
}

/// Record retrievals for utility tracking
fn record_retrievals(episodes: &[ScoredEpisode], query: &str, store: &EpisodeStore) -> Result<()> {
    let project = std::env::current_dir()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
        .unwrap_or_else(|| "unknown".to_string());

    for scored in episodes {
        let mut episode = scored.episode.clone();

        // Add retrieval record
        episode.retrieval_history.push(RetrievalRecord {
            timestamp: Utc::now(),
            project: project.clone(),
            task_description: query.to_string(),
            was_helpful: None, // Will be updated via feedback
        });

        // Update retrieval count
        episode.utility.retrieval_count += 1;

        // Save updated episode
        store.update(&episode)?;
    }

    // Also save IDs to feedback log for easy reference
    let feedback_log = Config::feedback_log_path()?;
    let ids: Vec<String> = episodes
        .iter()
        .map(|e| e.episode.id[..8].to_string())
        .collect();
    let log_entry = format!(
        "{}\tquery:{}\tids:{}\n",
        Utc::now().to_rfc3339(),
        query.replace('\t', " "),
        ids.join(",")
    );
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(feedback_log)?
        .write_all(log_entry.as_bytes())?;

    Ok(())
}

/// A scored episode with similarity and utility scores
#[derive(Debug, Clone, serde::Serialize)]
pub struct ScoredEpisode {
    pub episode: Episode,
    pub similarity_score: f32,
    pub utility_score: f32,
    pub combined_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_text_similarity() {
        let episode = Episode::new("test".to_string(), "fix authentication bug".to_string());

        // Similar query
        let similarity = calculate_text_similarity("fix auth bug", &episode);
        assert!(similarity > 0.0);

        // Unrelated query
        let similarity = calculate_text_similarity("database migration", &episode);
        assert!(similarity < 0.3);
    }
}
