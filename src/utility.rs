#![allow(dead_code)]

use anyhow::Result;
use chrono::{Duration, Utc};
use std::collections::HashMap;

use crate::episode::Episode;
use crate::indexer::EpisodeIndexer;
use crate::store::EpisodeStore;

/// Utility learning parameters
#[derive(Debug, Clone)]
pub struct UtilityParams {
    /// Decay rate per day for unused episodes (0.0 - 1.0)
    pub decay_rate: f64,
    /// Discount factor for Bellman updates (gamma, 0.0 - 1.0)
    pub discount_factor: f64,
    /// Learning rate for utility updates (alpha, 0.0 - 1.0)
    pub learning_rate: f64,
    /// Minimum similarity threshold for propagation
    pub propagation_threshold: f32,
    /// Maximum propagation depth (hops)
    pub max_propagation_depth: u32,
}

impl Default for UtilityParams {
    fn default() -> Self {
        Self {
            decay_rate: 0.01,           // 1% decay per day
            discount_factor: 0.9,       // Standard RL discount
            learning_rate: 0.1,         // Conservative updates
            propagation_threshold: 0.5, // 50% similarity minimum
            max_propagation_depth: 2,   // 2-hop propagation
        }
    }
}

/// Results from a utility propagation run
#[derive(Debug)]
pub struct PropagationResult {
    pub episodes_processed: usize,
    pub episodes_updated: usize,
    pub total_utility_change: f64,
    pub decayed_episodes: usize,
    pub propagated_episodes: usize,
}

/// Run the full utility learning pipeline
pub async fn run_propagation() -> Result<PropagationResult> {
    let store = EpisodeStore::new()?;
    let params = UtilityParams::default();

    let mut result = PropagationResult {
        episodes_processed: 0,
        episodes_updated: 0,
        total_utility_change: 0.0,
        decayed_episodes: 0,
        propagated_episodes: 0,
    };

    // Load all episodes
    let episodes = store.list_all()?;
    result.episodes_processed = episodes.len();

    if episodes.is_empty() {
        return Ok(result);
    }

    println!("  Processing {} episodes...", episodes.len());

    // Step 1: Apply time-based decay
    println!("  📉 Applying utility decay...");
    let decay_result = apply_utility_decay(&store, &episodes, &params)?;
    result.decayed_episodes = decay_result.0;
    result.total_utility_change += decay_result.1;

    // Step 2: Bellman propagation (if we have the vector index)
    println!("  🔄 Running Bellman propagation...");
    match run_bellman_propagation(&store, &params).await {
        Ok((propagated, change)) => {
            result.propagated_episodes = propagated;
            result.total_utility_change += change;
        }
        Err(e) => {
            println!("    ⚠️  Skipping vector propagation: {}", e);
            // Fall back to tag-based propagation
            let (propagated, change) = run_tag_propagation(&store, &episodes, &params)?;
            result.propagated_episodes = propagated;
            result.total_utility_change += change;
        }
    }

    // Step 3: Update stored utility scores
    println!("  💾 Saving updated utilities...");
    let updated = save_utility_updates(&store)?;
    result.episodes_updated = updated;

    // Step 4: Sync utility scores to vector index
    println!("  🔍 Syncing to vector index...");
    if let Err(e) = sync_utility_to_index().await {
        println!("    ⚠️  Index sync skipped: {}", e);
    }

    Ok(result)
}

/// Apply time-based utility decay to episodes
fn apply_utility_decay(
    store: &EpisodeStore,
    episodes: &[Episode],
    params: &UtilityParams,
) -> Result<(usize, f64)> {
    let now = Utc::now();
    let mut decayed = 0;
    let mut total_change = 0.0;

    for episode in episodes {
        // Calculate days since last activity (retrieval or creation)
        let last_activity = episode
            .retrieval_history
            .last()
            .map(|r| r.timestamp)
            .unwrap_or(episode.timestamp_end);

        let days_inactive = (now - last_activity).num_days().max(0) as f64;

        // Apply exponential decay: utility *= (1 - decay_rate)^days
        let decay_factor = (1.0 - params.decay_rate).powf(days_inactive);

        // Only apply decay if significant
        if decay_factor < 0.99 {
            let mut ep = episode.clone();
            let old_score = ep.utility.calculate_score();

            // Decay is applied by reducing the effective helpful ratio
            // We don't change counts, but store the decayed score
            let new_score = old_score * decay_factor as f32;
            ep.utility.score = Some(new_score);

            total_change += (new_score - old_score) as f64;

            store.update(&ep)?;
            decayed += 1;
        }
    }

    Ok((decayed, total_change))
}

/// Run Bellman-style utility propagation using vector similarity
async fn run_bellman_propagation(
    store: &EpisodeStore,
    params: &UtilityParams,
) -> Result<(usize, f64)> {
    let indexer = EpisodeIndexer::new().await?;

    if !indexer.is_indexed().await {
        anyhow::bail!("Vector index not available");
    }

    let episodes = store.list_all()?;
    let mut propagated = 0;
    let mut total_change = 0.0;

    // Find episodes with high helpfulness to propagate from
    let helpful_episodes: Vec<_> = episodes
        .iter()
        .filter(|ep| {
            let ratio = if ep.utility.retrieval_count > 0 {
                ep.utility.helpful_count as f32 / ep.utility.retrieval_count as f32
            } else {
                0.0
            };
            ratio > 0.5 && ep.utility.retrieval_count >= 2
        })
        .collect();

    if helpful_episodes.is_empty() {
        return Ok((0, 0.0));
    }

    println!(
        "    Found {} high-utility episodes to propagate from",
        helpful_episodes.len()
    );

    // For each helpful episode, find similar episodes and propagate utility
    for source in &helpful_episodes {
        // Create search query from episode content
        let query = format!(
            "{} {} {}",
            source.intent.raw_prompt,
            source.intent.domain.join(" "),
            source.intent.task_type
        );

        // Find similar episodes
        let similar = indexer.search(&query, 10, None).await?;

        for result in similar {
            // Skip self
            if result.id == source.id {
                continue;
            }

            // Skip if similarity is too low
            if result.similarity_score < params.propagation_threshold {
                continue;
            }

            // Load the target episode
            if let Ok(mut target) = store.load(&result.id) {
                let old_score = target.utility.score.unwrap_or(0.5);
                let source_score = source.utility.calculate_score();

                // Bellman update: Q(s) = Q(s) + α * (γ * Q(s') - Q(s))
                // Where s' is the similar helpful episode
                let td_error =
                    params.discount_factor * source_score as f64 * result.similarity_score as f64
                        - old_score as f64;
                let new_score = old_score + (params.learning_rate * td_error) as f32;
                let new_score = new_score.clamp(0.0, 1.0);

                if (new_score - old_score).abs() > 0.01 {
                    target.utility.score = Some(new_score);
                    store.update(&target)?;

                    total_change += (new_score - old_score) as f64;
                    propagated += 1;
                }
            }
        }
    }

    Ok((propagated, total_change))
}

/// Fallback tag-based propagation when vector index is unavailable
fn run_tag_propagation(
    store: &EpisodeStore,
    episodes: &[Episode],
    params: &UtilityParams,
) -> Result<(usize, f64)> {
    // Build tag -> episode mapping
    let mut tag_episodes: HashMap<String, Vec<&Episode>> = HashMap::new();

    for ep in episodes {
        for tag in &ep.intent.domain {
            tag_episodes.entry(tag.to_lowercase()).or_default().push(ep);
        }
        // Also use task type as implicit tag
        tag_episodes
            .entry(ep.intent.task_type.to_string().to_lowercase())
            .or_default()
            .push(ep);
    }

    let mut propagated = 0;
    let mut total_change = 0.0;

    // For each tag group, propagate from high-utility to low-utility episodes
    for (_tag, group) in &tag_episodes {
        if group.len() < 2 {
            continue;
        }

        // Find the average utility in this group
        let avg_utility: f32 = group
            .iter()
            .map(|ep| ep.utility.calculate_score())
            .sum::<f32>()
            / group.len() as f32;

        // Propagate from above-average to below-average
        for ep in group {
            let current = ep.utility.calculate_score();

            if current < avg_utility - 0.1 {
                // This episode could benefit from propagation
                let mut updated = (*ep).clone();
                let new_score = current + params.learning_rate as f32 * (avg_utility - current);
                let new_score = new_score.clamp(0.0, 1.0);

                if (new_score - current).abs() > 0.01 {
                    updated.utility.score = Some(new_score);
                    store.update(&updated)?;

                    total_change += (new_score - current) as f64;
                    propagated += 1;
                }
            }
        }
    }

    Ok((propagated, total_change))
}

/// Save any pending utility updates
fn save_utility_updates(store: &EpisodeStore) -> Result<usize> {
    // Updates are saved incrementally, so just return count
    let episodes = store.list_all()?;
    Ok(episodes
        .iter()
        .filter(|ep| ep.utility.score.is_some())
        .count())
}

/// Sync utility scores to the vector index
async fn sync_utility_to_index() -> Result<()> {
    let store = EpisodeStore::new()?;
    let indexer = EpisodeIndexer::new().await?;

    if !indexer.is_indexed().await {
        anyhow::bail!("Index not available");
    }

    let episodes = store.list_all()?;

    for ep in episodes {
        let score = ep
            .utility
            .score
            .unwrap_or_else(|| ep.utility.calculate_score());
        indexer.update_utility(&ep.id, score).await?;
    }

    Ok(())
}

/// Prune episodes based on age and utility
pub fn prune_episodes(
    store: &EpisodeStore,
    max_age_days: Option<u32>,
    min_utility: Option<f32>,
    dry_run: bool,
) -> Result<PruneResult> {
    let episodes = store.list_all()?;
    let now = Utc::now();

    let mut result = PruneResult {
        candidates: Vec::new(),
        pruned: 0,
        retained: 0,
    };

    for ep in episodes {
        let mut should_prune = false;
        let mut reasons = Vec::new();

        // Check age
        if let Some(max_days) = max_age_days {
            let age_days = (now - ep.timestamp_start).num_days();
            if age_days > max_days as i64 {
                should_prune = true;
                reasons.push(format!("age: {} days", age_days));
            }
        }

        // Check utility
        if let Some(min_util) = min_utility {
            let utility = ep
                .utility
                .score
                .unwrap_or_else(|| ep.utility.calculate_score());
            if utility < min_util {
                should_prune = true;
                reasons.push(format!("utility: {:.0}%", utility * 100.0));
            }
        }

        // Don't prune episodes that have been helpful
        if ep.utility.helpful_count > 0 {
            should_prune = false;
            reasons.clear();
            reasons.push("retained: has helpful feedback".to_string());
        }

        if should_prune {
            result.candidates.push(PruneCandidate {
                id: ep.id.clone(),
                short_id: ep.id[..8].to_string(),
                intent: ep.intent.raw_prompt.chars().take(50).collect(),
                reasons,
            });

            if !dry_run {
                store.delete(&ep.id)?;
                result.pruned += 1;
            }
        } else {
            result.retained += 1;
        }
    }

    Ok(result)
}

/// Results from a prune operation
#[derive(Debug)]
pub struct PruneResult {
    pub candidates: Vec<PruneCandidate>,
    pub pruned: usize,
    pub retained: usize,
}

/// A candidate for pruning
#[derive(Debug)]
pub struct PruneCandidate {
    pub id: String,
    pub short_id: String,
    pub intent: String,
    pub reasons: Vec<String>,
}

/// Calculate temporal credit assignment for a sequence of episodes
/// Episodes that led to successful outcomes get credit
pub fn temporal_credit_assignment(
    store: &EpisodeStore,
    project: Option<&str>,
    params: &UtilityParams,
) -> Result<usize> {
    let mut episodes = store.list_all()?;

    // Filter by project if specified
    if let Some(proj) = project {
        episodes.retain(|ep| ep.project.to_lowercase() == proj.to_lowercase());
    }

    // Sort by timestamp
    episodes.sort_by_key(|ep| ep.timestamp_start);

    if episodes.len() < 2 {
        return Ok(0);
    }

    let mut updated = 0;

    // Look for success patterns: sequences where a success followed other episodes
    for i in 1..episodes.len() {
        let current = &episodes[i];

        // If this episode was successful, credit preceding related episodes
        if current.outcome.status == crate::episode::OutcomeStatus::Success {
            // Look back at recent episodes (within 1 hour)
            let lookback = Duration::hours(1);

            for j in (0..i).rev() {
                let prev = &episodes[j];

                // Stop if too old
                if current.timestamp_start - prev.timestamp_end > lookback {
                    break;
                }

                // Check if related (same project, similar tags)
                let related = prev.project == current.project
                    || prev
                        .intent
                        .domain
                        .iter()
                        .any(|t| current.intent.domain.contains(t));

                if related {
                    let mut prev_updated = prev.clone();
                    let old_score = prev_updated.utility.score.unwrap_or(0.5);

                    // Give credit based on temporal distance
                    let time_factor = 1.0 - (i - j) as f64 * 0.2; // Decreases by 20% per step
                    let credit = params.discount_factor * time_factor * 0.1; // Small credit boost

                    let new_score = (old_score as f64 + credit).min(1.0) as f32;

                    if new_score > old_score + 0.01 {
                        prev_updated.utility.score = Some(new_score);
                        store.update(&prev_updated)?;
                        updated += 1;
                    }
                }
            }
        }
    }

    Ok(updated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utility_params_default() {
        let params = UtilityParams::default();
        assert!(params.decay_rate > 0.0 && params.decay_rate < 1.0);
        assert!(params.discount_factor > 0.0 && params.discount_factor <= 1.0);
        assert!(params.learning_rate > 0.0 && params.learning_rate <= 1.0);
    }

    #[test]
    fn test_decay_calculation() {
        let params = UtilityParams::default();

        // After 30 days with 1% decay rate
        let decay_factor = (1.0 - params.decay_rate).powf(30.0);
        assert!(decay_factor < 1.0);
        assert!(decay_factor > 0.5); // Should still retain most value

        // After 100 days
        let decay_factor_100 = (1.0 - params.decay_rate).powf(100.0);
        assert!(decay_factor_100 < decay_factor);
    }
}
