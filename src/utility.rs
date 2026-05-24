// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

use anyhow::Result;
use chrono::{Duration, Utc};
use std::collections::HashMap;

use crate::config::Config;
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
    /// v0.6.2: only episodes whose Claim has `falsifiability >=` this are
    /// allowed to seed propagation.
    pub min_falsifiability: f32,
    /// If true, episodes without any Claim (older / non-LLM captures) are
    /// also excluded from propagation. Default false → lenient.
    pub require_claim_for_propagation: bool,
}

impl Default for UtilityParams {
    fn default() -> Self {
        Self {
            decay_rate: 0.01,           // 1% decay per day
            discount_factor: 0.9,       // Standard RL discount
            learning_rate: 0.1,         // Conservative updates
            propagation_threshold: 0.5, // 50% similarity minimum
            max_propagation_depth: 2,   // 2-hop propagation
            min_falsifiability: 0.7,
            require_claim_for_propagation: false,
        }
    }
}

impl UtilityParams {
    /// Create UtilityParams from the loaded config, so config.toml actually controls RL behavior.
    pub fn from_config(config: &Config) -> Self {
        Self {
            decay_rate: config.bellman.decay_rate,
            discount_factor: config.bellman.gamma as f64,
            learning_rate: config.bellman.alpha as f64,
            propagation_threshold: config.bellman.propagation_threshold,
            max_propagation_depth: config.bellman.max_propagation_depth,
            min_falsifiability: config.bellman.min_falsifiability,
            require_claim_for_propagation: config.bellman.require_claim_for_propagation,
        }
    }
}

/// True when `episode` is allowed to seed Bellman propagation per v0.6.2
/// gating rules. Encapsulated so the same predicate can be applied across
/// future propagation paths (temporal credit, multi-hop, dream cycle).
pub fn episode_can_propagate(episode: &Episode, params: &UtilityParams) -> bool {
    match &episode.intent.claim {
        Some(c) => c.falsifiability >= params.min_falsifiability,
        None => !params.require_claim_for_propagation,
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
    /// Number of propagation hops executed
    pub hops_executed: u32,
    /// Whether propagation converged early (no updates in last hop)
    pub converged: bool,
}

/// Run the full utility learning pipeline
pub async fn run_propagation(config: &Config) -> Result<PropagationResult> {
    let store = EpisodeStore::new()?;
    let params = UtilityParams::from_config(config);

    let mut result = PropagationResult {
        episodes_processed: 0,
        episodes_updated: 0,
        total_utility_change: 0.0,
        decayed_episodes: 0,
        propagated_episodes: 0,
        hops_executed: 0,
        converged: false,
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
    match run_bellman_propagation(&store, &params, None).await {
        Ok(bellman_result) => {
            result.propagated_episodes = bellman_result.propagated;
            result.total_utility_change += bellman_result.total_change;
            result.hops_executed = bellman_result.hops_executed;
            result.converged = bellman_result.converged;
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

/// Decay-only entrypoint for `tempera dream`'s decay phase.
///
/// Runs the same scope-aware decay as `run_propagation`'s step 1, then
/// persists updated utility scores. Skips Bellman propagation, temporal
/// credit, and the post-step vector-index sync so a dream cycle can apply
/// decay cheaply without re-doing expensive work.
pub async fn run_decay_only(config: &Config) -> Result<PropagationResult> {
    let store = EpisodeStore::new()?;
    let params = UtilityParams::from_config(config);
    let mut result = PropagationResult {
        episodes_processed: 0,
        episodes_updated: 0,
        total_utility_change: 0.0,
        decayed_episodes: 0,
        propagated_episodes: 0,
        hops_executed: 0,
        converged: false,
    };

    let episodes = store.list_all()?;
    result.episodes_processed = episodes.len();
    if episodes.is_empty() {
        return Ok(result);
    }

    let (decayed, change) = apply_utility_decay(&store, &episodes, &params)?;
    result.decayed_episodes = decayed;
    result.total_utility_change = change;
    result.episodes_updated = save_utility_updates(&store)?;
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

        // v0.6.4: scope-aware decay rate. Forever-scoped truths don't
        // decay; workaround-scoped notes decay 5×. Episodes with no
        // validity_scope (pre-v0.6.4 captures, or modern captures the
        // agent didn't scope) fall back to params.decay_rate so behavior
        // matches the legacy uniform-decay default.
        let scope = episode
            .intent
            .claim
            .as_ref()
            .and_then(|c| c.validity_scope.as_ref());
        let decay_rate = match scope {
            Some(s) => crate::episode::decay_rate_per_day(Some(s)),
            None => params.decay_rate,
        };

        // Apply exponential decay: utility *= (1 - decay_rate)^days
        let decay_factor = (1.0 - decay_rate).powf(days_inactive);

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

/// Result of a multi-hop Bellman propagation run
pub struct BellmanResult {
    pub propagated: usize,
    pub total_change: f64,
    pub hops_executed: u32,
    pub converged: bool,
}

/// Run multi-hop Bellman-style utility propagation using vector similarity.
/// At each hop, episodes updated in the previous hop become new sources,
/// allowing value to flow through the similarity graph.
pub async fn run_bellman_propagation(
    store: &EpisodeStore,
    params: &UtilityParams,
    project_filter: Option<&str>,
) -> Result<BellmanResult> {
    let indexer = EpisodeIndexer::new().await?;

    if !indexer.is_indexed().await {
        anyhow::bail!("Vector index not available");
    }

    let mut total_propagated = 0;
    let mut total_change = 0.0;
    let mut hops_executed = 0;
    let mut converged = false;

    // Hop 0: seed sources are episodes with high helpfulness
    let all_episodes = store.list_all()?;
    let episodes: Vec<_> = if let Some(proj) = project_filter {
        all_episodes
            .into_iter()
            .filter(|e| e.project.to_lowercase().contains(&proj.to_lowercase()))
            .collect()
    } else {
        all_episodes
    };

    // v0.6.2: gate sources by Claim falsifiability so logistics-only
    // episodes don't shape the BKM lane. `episode_can_propagate` handles
    // the lenient-vs-strict default for episodes lacking a Claim.
    let mut filtered_for_claim = 0_usize;
    let mut source_ids: std::collections::HashSet<String> = episodes
        .iter()
        .filter(|ep| {
            let ratio = if ep.utility.retrieval_count > 0 {
                ep.utility.helpful_count as f32 / ep.utility.retrieval_count as f32
            } else {
                0.0
            };
            ratio > 0.5 && ep.utility.retrieval_count >= 2
        })
        .filter(|ep| {
            let ok = episode_can_propagate(ep, params);
            if !ok {
                filtered_for_claim += 1;
            }
            ok
        })
        .map(|ep| ep.id.clone())
        .collect();
    if filtered_for_claim > 0 {
        println!(
            "    Gated {filtered_for_claim} otherwise-high-utility episode(s) — \
             claim.falsifiability < {:.2} or absent",
            params.min_falsifiability
        );
    }

    if source_ids.is_empty() {
        return Ok(BellmanResult {
            propagated: 0,
            total_change: 0.0,
            hops_executed: 0,
            converged: true,
        });
    }

    println!(
        "    Found {} high-utility episodes to propagate from",
        source_ids.len()
    );

    for depth in 0..params.max_propagation_depth {
        // Discount decreases with depth: gamma^(depth+1)
        let depth_discount = params.discount_factor.powi(depth as i32 + 1);
        let mut updated_this_hop = std::collections::HashSet::new();
        let mut hop_change = 0.0;

        // Load source episodes for this hop
        let sources: Vec<Episode> = source_ids
            .iter()
            .filter_map(|id| store.load(id).ok())
            .collect();

        for source in &sources {
            let query = format!(
                "{} {} {}",
                source.intent.raw_prompt,
                source.intent.domain.join(" "),
                source.intent.task_type
            );

            let similar = indexer.search(&query, 10, project_filter).await?;

            for result in similar {
                if result.id == source.id || source_ids.contains(&result.id) {
                    continue; // Skip self and already-processed sources
                }
                if result.similarity_score < params.propagation_threshold {
                    continue;
                }

                if let Ok(mut target) = store.load(&result.id) {
                    let old_score = target.utility.score.unwrap_or(0.5);
                    let source_score = source.utility.calculate_score();

                    // Bellman update with depth-discounted gamma
                    let td_error =
                        depth_discount * source_score as f64 * result.similarity_score as f64
                            - old_score as f64;
                    let new_score = old_score + (params.learning_rate * td_error) as f32;
                    let new_score = new_score.clamp(0.0, 1.0);

                    if (new_score - old_score).abs() > 0.01 {
                        target.utility.score = Some(new_score);
                        store.update(&target)?;
                        hop_change += (new_score - old_score) as f64;
                        updated_this_hop.insert(target.id);
                    }
                }
            }
        }

        total_propagated += updated_this_hop.len();
        total_change += hop_change;
        hops_executed = depth + 1;

        if updated_this_hop.is_empty() {
            converged = true;
            break;
        }

        println!(
            "    Hop {}: updated {} episodes (Δ{:+.3})",
            depth + 1,
            updated_this_hop.len(),
            hop_change
        );

        // Next hop propagates from episodes updated in this hop
        source_ids = updated_this_hop;
    }

    // Session-link propagation (runs once, not per-hop)
    let helpful_episodes: Vec<Episode> = episodes
        .iter()
        .filter(|ep| {
            let ratio = if ep.utility.retrieval_count > 0 {
                ep.utility.helpful_count as f32 / ep.utility.retrieval_count as f32
            } else {
                0.0
            };
            ratio > 0.5 && ep.utility.retrieval_count >= 2
        })
        .cloned()
        .collect();

    for source in &helpful_episodes {
        if let Some(session_id) = &source.session_id {
            let session_episodes = store.list_by_session(session_id)?;
            for mut target in session_episodes {
                if target.id == source.id {
                    continue;
                }
                let old_score = target.utility.score.unwrap_or(0.5);
                let source_score = source.utility.calculate_score();

                let session_similarity = 0.9_f64;
                let td_error = params.discount_factor * source_score as f64 * session_similarity
                    - old_score as f64;
                let new_score = old_score + (params.learning_rate * td_error) as f32;
                let new_score = new_score.clamp(0.0, 1.0);

                if (new_score - old_score).abs() > 0.01 {
                    target.utility.score = Some(new_score);
                    store.update(&target)?;
                    total_change += (new_score - old_score) as f64;
                    total_propagated += 1;
                }
            }
        }
    }

    Ok(BellmanResult {
        propagated: total_propagated,
        total_change,
        hops_executed,
        converged,
    })
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
/// When CLI args are None, falls back to config values.
pub fn prune_episodes(
    store: &EpisodeStore,
    max_age_days: Option<u32>,
    min_utility: Option<f32>,
    dry_run: bool,
    config: &Config,
) -> Result<PruneResult> {
    let effective_max_age = max_age_days.unwrap_or(config.storage.max_age_days);
    let effective_min_utility = min_utility.unwrap_or(config.storage.min_utility_threshold);

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
        let age_days = (now - ep.timestamp_start).num_days();
        if age_days > effective_max_age as i64 {
            should_prune = true;
            reasons.push(format!("age: {} days", age_days));
        }

        // Check utility
        let utility = ep
            .utility
            .score
            .unwrap_or_else(|| ep.utility.calculate_score());
        if utility < effective_min_utility {
            should_prune = true;
            reasons.push(format!("utility: {:.0}%", utility * 100.0));
        }

        // Don't prune episodes that have been helpful
        if ep.utility.helpful_count > 0 {
            should_prune = false;
            reasons.clear();
            reasons.push("retained: has helpful feedback".to_string());
        }

        // Don't prune episodes with enough retrievals (config-driven protection)
        if should_prune && ep.utility.retrieval_count >= config.storage.min_retrievals {
            should_prune = false;
            reasons.clear();
            reasons.push(format!(
                "retained: {} retrievals (min: {})",
                ep.utility.retrieval_count, config.storage.min_retrievals
            ));
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
    config: &Config,
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
            // Look back at recent episodes (configurable window)
            let lookback = Duration::hours(config.bellman.temporal_credit_window_hours);

            for j in (0..i).rev() {
                let prev = &episodes[j];

                // Check if session-linked (bypasses time window)
                let same_session = current
                    .session_id
                    .as_ref()
                    .is_some_and(|s| prev.session_id.as_ref() == Some(s));

                // Stop if too old AND not session-linked
                if !same_session && current.timestamp_start - prev.timestamp_end > lookback {
                    break;
                }

                // Check if related (same session, same project, or similar tags)
                let related = same_session
                    || prev.project == current.project
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
    fn test_utility_params_from_config() {
        let mut config = Config::default();
        config.bellman.gamma = 0.95;
        config.bellman.alpha = 0.2;
        config.bellman.decay_rate = 0.05;
        config.bellman.propagation_threshold = 0.6;
        config.bellman.max_propagation_depth = 3;

        let params = UtilityParams::from_config(&config);
        assert!((params.discount_factor - 0.95_f32 as f64).abs() < 1e-6);
        assert!((params.learning_rate - 0.2_f32 as f64).abs() < 1e-6);
        assert!((params.decay_rate - 0.05).abs() < f64::EPSILON);
        assert_eq!(params.propagation_threshold, 0.6);
        assert_eq!(params.max_propagation_depth, 3);
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

    #[test]
    fn episode_can_propagate_lenient_default() {
        // No claim, require_claim=false (default) ⇒ allowed.
        let ep = Episode::new("p".into(), "test".into());
        let params = UtilityParams::default();
        assert!(episode_can_propagate(&ep, &params));
    }

    #[test]
    fn episode_can_propagate_blocks_no_claim_when_strict() {
        let ep = Episode::new("p".into(), "test".into());
        let params = UtilityParams {
            require_claim_for_propagation: true,
            ..UtilityParams::default()
        };
        assert!(!episode_can_propagate(&ep, &params));
    }

    #[test]
    fn episode_can_propagate_threshold_gates_low_falsifiability() {
        let mut ep = Episode::new("p".into(), "test".into());
        ep.intent.claim = Some(crate::episode::Claim {
            falsifiability: 0.5,
            category: crate::episode::ClaimCategory::Logistics,
            validity_scope: None,
        });
        let params = UtilityParams::default(); // threshold 0.7
        assert!(!episode_can_propagate(&ep, &params));
    }

    #[test]
    fn episode_can_propagate_threshold_admits_high_falsifiability() {
        let mut ep = Episode::new("p".into(), "test".into());
        ep.intent.claim = Some(crate::episode::Claim {
            falsifiability: 0.9,
            category: crate::episode::ClaimCategory::ApiContract,
            validity_scope: None,
        });
        let params = UtilityParams::default();
        assert!(episode_can_propagate(&ep, &params));
    }
}
