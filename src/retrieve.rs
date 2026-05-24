// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]
use anyhow::Result;
use chrono::Utc;
use colored::Colorize;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::Write;

use crate::config::{Config, RetrievalMode};
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

    // Dispatch to the configured retrieval mode (hybrid/vector/keyword); fall
    // back to text search only if all index-based paths fail.
    let episodes = match try_search(query, limit, project.as_deref(), config).await {
        Ok(results) if !results.is_empty() => {
            match config.retrieval.mode {
                RetrievalMode::Hybrid => println!("🔍 Hybrid retrieval (vector + BM25)...\n"),
                RetrievalMode::Vector => println!("🔍 Semantic vector search...\n"),
                RetrievalMode::Keyword => println!("🔍 BM25 keyword search...\n"),
            }
            results
        }
        _ => {
            println!("🔍 Using text-based search (run 'tempera index' for semantic search)...\n");
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

/// Dispatch retrieval to the configured mode. Falls back to vector-only if
/// keyword or hybrid pipelines aren't available.
pub async fn try_search(
    query: &str,
    limit: usize,
    project_filter: Option<&str>,
    config: &Config,
) -> Result<Vec<ScoredEpisode>> {
    match config.retrieval.mode {
        RetrievalMode::Vector => try_vector_search(query, limit, project_filter, config).await,
        RetrievalMode::Keyword => try_keyword_search(query, limit, project_filter, config).await,
        RetrievalMode::Hybrid => try_hybrid_search(query, limit, project_filter, config).await,
    }
}

/// Try to retrieve episodes using vector search
pub async fn try_vector_search(
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
            let recency = calculate_recency_score(&episode, config.retrieval.recency_halflife_days);
            let combined = combined_score(result.similarity_score, utility, recency, config);

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

    // Filter by minimum similarity
    episodes.retain(|e| e.similarity_score >= config.retrieval.min_similarity);

    // Apply MMR for diversity
    let episodes = apply_mmr(episodes, limit, config.retrieval.mmr_lambda);

    Ok(episodes)
}

/// Retrieve using only the BM25 keyword index. BM25 scores are normalized to
/// `[0, 1]` (divided by the max score in the result set) so they plug into
/// the same `combined_score` machinery as vector similarity.
pub async fn try_keyword_search(
    query: &str,
    limit: usize,
    project_filter: Option<&str>,
    config: &Config,
) -> Result<Vec<ScoredEpisode>> {
    let indexer = EpisodeIndexer::new().await?;
    if indexer.keyword_doc_count() == 0 {
        anyhow::bail!("Keyword index is empty");
    }
    let store = EpisodeStore::new()?;

    let hits = indexer.search_keyword(query, limit * 2, project_filter);
    if hits.is_empty() {
        return Ok(Vec::new());
    }
    let max_score = hits
        .iter()
        .map(|h| h.score)
        .fold(0.0_f32, f32::max)
        .max(1e-6);

    let mut episodes: Vec<ScoredEpisode> = hits
        .into_iter()
        .filter_map(|hit| {
            let episode = store.load(&hit.id).ok()?;
            let sim = hit.score / max_score;
            let utility = episode.utility.calculate_score();
            let recency = calculate_recency_score(&episode, config.retrieval.recency_halflife_days);
            let combined = combined_score(sim, utility, recency, config);
            Some(ScoredEpisode {
                episode,
                similarity_score: sim,
                utility_score: utility,
                combined_score: combined,
            })
        })
        .collect();

    episodes.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(Ordering::Equal)
    });

    // Skip min_similarity here — BM25 score thresholds aren't calibrated to
    // the cosine threshold the user picked for vector mode.

    Ok(apply_mmr(episodes, limit, config.retrieval.mmr_lambda))
}

/// Hybrid retrieval: run vector + keyword in parallel, fuse via RRF, then
/// apply the standard `combined_score` (utility + recency) to the fused
/// candidates. Falls back to whichever side has results if the other is
/// empty or unavailable.
pub async fn try_hybrid_search(
    query: &str,
    limit: usize,
    project_filter: Option<&str>,
    config: &Config,
) -> Result<Vec<ScoredEpisode>> {
    let indexer = EpisodeIndexer::new().await?;
    let store = EpisodeStore::new()?;

    let fetch = (limit * 4).max(20);

    // Keyword search is sync over the in-memory index; vector search is async.
    // Run them concurrently when both are available.
    let keyword_hits = indexer.search_keyword(query, fetch, project_filter);
    let vector_hits = if indexer.is_indexed().await {
        indexer
            .search(query, fetch, project_filter)
            .await
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    if keyword_hits.is_empty() && vector_hits.is_empty() {
        anyhow::bail!("No results from either retrieval path");
    }

    let keyword_ranked: Vec<String> = keyword_hits.iter().map(|h| h.id.clone()).collect();
    let vector_ranked: Vec<String> = vector_hits.iter().map(|r| r.id.clone()).collect();

    // Capture true vector similarity so we can preserve it on the returned
    // ScoredEpisode (used by callers that filter or display sim).
    let vector_sim: HashMap<String, f32> = vector_hits
        .iter()
        .map(|r| (r.id.clone(), r.similarity_score))
        .collect();

    let fused = reciprocal_rank_fusion(&[keyword_ranked, vector_ranked], config.retrieval.rrf_k);

    let max_rrf = fused.first().map(|(_, s)| *s).unwrap_or(1e-6).max(1e-6);

    let sim_w = config.retrieval.hybrid_similarity_weight;
    let util_w = config.retrieval.hybrid_utility_weight;
    let total_w = (sim_w + util_w).max(f32::EPSILON);

    let mut episodes: Vec<ScoredEpisode> = fused
        .into_iter()
        .take(fetch)
        .filter_map(|(id, rrf_score)| {
            let episode = store.load(&id).ok()?;
            // Normalize RRF to [0, 1] within this result set.
            let sim_normalized = rrf_score / max_rrf;
            let utility = episode.utility.calculate_score();
            // Hybrid-specific blend: RRF dominates (its rank carries both
            // lexical and semantic signal), utility is a smaller reweight.
            // Recency is intentionally omitted here — opt in via config if
            // you want it back in hybrid ranking.
            let combined = (sim_w * sim_normalized + util_w * utility) / total_w;
            // Surface true cosine sim when this doc came through the vector
            // path; fall back to RRF-normalized score otherwise.
            let surface_sim = vector_sim.get(&id).copied().unwrap_or(sim_normalized);
            Some(ScoredEpisode {
                episode,
                similarity_score: surface_sim,
                utility_score: utility,
                combined_score: combined,
            })
        })
        .collect();

    episodes.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(Ordering::Equal)
    });

    // min_similarity is calibrated for vector mode; hybrid mode skips it
    // because the score distribution is different.

    Ok(apply_mmr(episodes, limit, config.retrieval.mmr_lambda))
}

/// Reciprocal-rank fusion of N rankings. Each doc gets `Σ 1/(k + rank + 1)`
/// where `rank` is its 0-indexed position in each input ranking.
pub fn reciprocal_rank_fusion(rankings: &[Vec<String>], k: f32) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    for ranking in rankings {
        for (rank, id) in ranking.iter().enumerate() {
            *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    let mut v: Vec<(String, f32)> = scores.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    v
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
            let recency = calculate_recency_score(&ep, config.retrieval.recency_halflife_days);
            let combined = combined_score(similarity, utility, recency, config);

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

    // Apply MMR for diversity
    let scored = apply_mmr(scored, limit, config.retrieval.mmr_lambda);

    Ok(scored)
}

/// Calculate recency score using exponential decay with configurable half-life.
/// Returns 1.0 for episodes created now, 0.5 at halflife_days, approaches 0.0 for old episodes.
fn calculate_recency_score(episode: &Episode, halflife_days: f32) -> f32 {
    let age_days = (Utc::now() - episode.timestamp_end).num_hours() as f32 / 24.0;
    if halflife_days <= 0.0 {
        return 1.0;
    }
    (-age_days * 2.0_f32.ln() / halflife_days).exp()
}

/// Combine similarity, utility, and recency scores with weight normalization.
fn combined_score(similarity: f32, utility: f32, recency: f32, config: &Config) -> f32 {
    let sim_w = config.retrieval.similarity_weight;
    let util_w = config.retrieval.utility_weight;
    let rec_w = config.retrieval.recency_weight;
    let total = sim_w + util_w + rec_w;
    if total == 0.0 {
        return 0.0;
    }
    (sim_w * similarity + util_w * utility + rec_w * recency) / total
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

        // Show utility with confidence level based on retrieval count
        let confidence = match ep.utility.retrieval_count {
            0 => "untested",
            1..=2 => "low confidence",
            3..=5 => "moderate confidence",
            _ => "high confidence",
        };
        println!(
            "**Relevance**: {:.0}% similarity, {:.0}% utility ({}, {} retrievals)",
            scored.similarity_score * 100.0,
            scored.utility_score * 100.0,
            confidence,
            ep.utility.retrieval_count
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
        "To provide feedback: tempera feedback helpful --episodes <id>,<id>".dimmed()
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

/// Apply Maximal Marginal Relevance (MMR) for result diversity
/// lambda: 0.0 = pure diversity, 1.0 = pure relevance
pub fn apply_mmr(
    mut candidates: Vec<ScoredEpisode>,
    limit: usize,
    lambda: f32,
) -> Vec<ScoredEpisode> {
    if candidates.is_empty() || limit == 0 {
        return vec![];
    }

    let mut selected: Vec<ScoredEpisode> = Vec::with_capacity(limit);

    // First result is always the highest scoring
    selected.push(candidates.remove(0));

    while !candidates.is_empty() && selected.len() < limit {
        // Find candidate with best MMR score
        let best_idx = candidates
            .iter()
            .enumerate()
            .map(|(idx, candidate)| {
                // Max similarity to any already-selected episode
                let max_sim_to_selected = selected
                    .iter()
                    .map(|s| text_overlap_similarity(&candidate.episode, &s.episode))
                    .fold(0.0_f32, |a, b| a.max(b));

                // MMR score: λ * relevance - (1-λ) * redundancy
                let mmr_score =
                    lambda * candidate.combined_score - (1.0 - lambda) * max_sim_to_selected;

                (idx, mmr_score)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx);

        if let Some(idx) = best_idx {
            selected.push(candidates.remove(idx));
        } else {
            break;
        }
    }

    selected
}

/// Calculate text overlap between two episodes for MMR diversity
pub fn text_overlap_similarity(a: &Episode, b: &Episode) -> f32 {
    let a_text = format!(
        "{} {} {}",
        a.intent.raw_prompt.to_lowercase(),
        a.intent.domain.join(" ").to_lowercase(),
        a.context.files_modified.join(" ").to_lowercase()
    );
    let b_text = format!(
        "{} {} {}",
        b.intent.raw_prompt.to_lowercase(),
        b.intent.domain.join(" ").to_lowercase(),
        b.context.files_modified.join(" ").to_lowercase()
    );

    let a_words: std::collections::HashSet<&str> = a_text.split_whitespace().collect();
    let b_words: std::collections::HashSet<&str> = b_text.split_whitespace().collect();

    if a_words.is_empty() || b_words.is_empty() {
        return 0.0;
    }

    let intersection = a_words.intersection(&b_words).count();
    let union = a_words.union(&b_words).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
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

    #[test]
    fn test_recency_score() {
        // A brand-new episode should score close to 1.0
        let ep = Episode::new("test".to_string(), "prompt".to_string());
        let score = calculate_recency_score(&ep, 30.0);
        assert!(score > 0.99, "Fresh episode should be ~1.0, got {}", score);

        // halflife_days=0 should always return 1.0
        let score_zero = calculate_recency_score(&ep, 0.0);
        assert!((score_zero - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_combined_score_normalization() {
        let config = Config::default();

        // With default weights (sim=0.3, util=0.7, rec=0.0), should match old behavior
        let score = combined_score(0.8, 0.6, 1.0, &config);
        // (0.3*0.8 + 0.7*0.6 + 0.0*1.0) / (0.3+0.7+0.0) = (0.24+0.42)/1.0 = 0.66
        assert!(
            (score - 0.66).abs() < 0.01,
            "Default weights: expected ~0.66, got {}",
            score
        );

        // With recency enabled, scores should normalize
        let mut config2 = Config::default();
        config2.retrieval.recency_weight = 0.2;
        let score2 = combined_score(0.8, 0.6, 1.0, &config2);
        // (0.3*0.8 + 0.7*0.6 + 0.2*1.0) / (0.3+0.7+0.2) = (0.24+0.42+0.2)/1.2 = 0.7167
        assert!(
            (score2 - 0.7167).abs() < 0.01,
            "With recency: expected ~0.717, got {}",
            score2
        );
    }

    fn ids(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn rrf_basic_two_rankings() {
        let r1 = ids(&["a", "b", "c"]);
        let r2 = ids(&["b", "a", "d"]);
        let fused = reciprocal_rank_fusion(&[r1, r2], 60.0);
        // 'a' and 'b' tie at the top; HashMap iteration order is non-deterministic
        // so we assert the *set* of top two, not a specific ordering between ties.
        let top_two: std::collections::HashSet<&str> = [fused[0].0.as_str(), fused[1].0.as_str()]
            .into_iter()
            .collect();
        let expected: std::collections::HashSet<&str> = ["a", "b"].into_iter().collect();
        assert_eq!(top_two, expected);
        assert!((fused[0].1 - fused[1].1).abs() < 1e-9, "a and b should tie");
        let a_score = fused.iter().find(|(id, _)| id == "a").unwrap().1;
        let c_score = fused.iter().find(|(id, _)| id == "c").unwrap().1;
        assert!(a_score > c_score, "a should outrank c");
    }

    #[test]
    fn rrf_single_ranking_preserves_order() {
        let r = ids(&["a", "b", "c"]);
        let fused = reciprocal_rank_fusion(&[r], 60.0);
        assert_eq!(fused[0].0, "a");
        assert_eq!(fused[1].0, "b");
        assert_eq!(fused[2].0, "c");
    }

    #[test]
    fn rrf_empty_rankings_returns_empty() {
        let fused = reciprocal_rank_fusion(&[], 60.0);
        assert!(fused.is_empty());
    }

    #[test]
    fn rrf_higher_k_compresses_score_spread() {
        let r = ids(&["a", "b"]);
        let low_k = reciprocal_rank_fusion(std::slice::from_ref(&r), 10.0);
        let high_k = reciprocal_rank_fusion(std::slice::from_ref(&r), 100.0);
        let low_diff = low_k[0].1 - low_k[1].1;
        let high_diff = high_k[0].1 - high_k[1].1;
        assert!(low_diff > high_diff, "lower k should produce wider spread");
    }

    #[test]
    fn rrf_score_decreases_with_rank() {
        let r = ids(&["a", "b", "c", "d", "e"]);
        let fused = reciprocal_rank_fusion(&[r], 60.0);
        for w in fused.windows(2) {
            assert!(
                w[0].1 > w[1].1,
                "scores should strictly decrease for a single ranking"
            );
        }
    }
}
