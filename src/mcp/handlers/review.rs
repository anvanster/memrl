use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::episode::Episode;
use crate::mcp::helpers::{extract_project, load_project_episodes};
use crate::{indexer, store};

/// Similarity threshold for clustering episodes as duplicates.
/// Must be high (0.85+) because project-scoped searches inflate similarity —
/// all episodes within the same project share semantic context.
const CLUSTER_THRESHOLD: f32 = 0.85;

/// Review and consolidate memories
pub(crate) async fn handle(args: &Value) -> Result<String, String> {
    let project = extract_project(args);
    let action = args
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("analyze");

    let store = store::EpisodeStore::new().map_err(|e| e.to_string())?;
    let project_episodes = load_project_episodes(&store, &project)?;

    if project_episodes.is_empty() {
        return Ok(format!("No memories found for project '{}'.", project));
    }

    let mut output = format!("🔍 Memory Review for '{}'\n", project);
    output.push_str(&"=".repeat(22 + project.len()));
    output.push_str("\n\n");

    // Find stale memories (old with low utility)
    let stale: Vec<_> = project_episodes
        .iter()
        .filter(|e| {
            let age_days = (chrono::Utc::now() - e.timestamp_start).num_days();
            let utility = e.utility.calculate_score();
            age_days > 30 && utility < 0.2
        })
        .collect();

    // Find duplicate clusters using vector similarity (or Jaccard fallback)
    let clusters = find_duplicate_clusters(&project_episodes, &project).await;

    // Calculate feedback rate
    let total_retrievals: u32 = project_episodes
        .iter()
        .map(|e| e.utility.retrieval_count)
        .sum();
    let total_helpful: u32 = project_episodes
        .iter()
        .map(|e| e.utility.helpful_count)
        .sum();
    let feedback_rate = if total_retrievals > 0 {
        total_helpful as f32 / total_retrievals as f32
    } else {
        0.0
    };

    // Count high-value episodes (threshold 0.4, matching status.rs)
    let high_value_count = project_episodes
        .iter()
        .filter(|e| e.utility.calculate_score() > 0.4)
        .count();

    output.push_str("📊 Analysis Results:\n");
    output.push_str(&format!("  - Total memories: {}\n", project_episodes.len()));
    output.push_str(&format!("  - Stale (>30d, low utility): {}\n", stale.len()));
    output.push_str(&format!(
        "  - Duplicate clusters: {} ({} episodes)\n",
        clusters.len(),
        clusters.iter().map(|c| c.len()).sum::<usize>()
    ));
    output.push_str(&format!("  - High-value memories: {}\n", high_value_count));
    output.push_str(&format!(
        "  - Feedback rate: {} of {} retrievals ({:.0}%)\n\n",
        total_helpful,
        total_retrievals,
        feedback_rate * 100.0
    ));

    if !stale.is_empty() {
        output.push_str("📅 Stale Memories:\n");
        for ep in stale.iter().take(5) {
            let summary: String = ep.intent.extracted_intent.chars().take(50).collect();
            output.push_str(&format!("  - {} ({}...)\n", &ep.id[..8], summary));
        }
        if stale.len() > 5 {
            output.push_str(&format!("  ... and {} more\n", stale.len() - 5));
        }
        output.push('\n');
    }

    if !clusters.is_empty() {
        output.push_str("🔄 Duplicate Clusters:\n");
        for (i, cluster) in clusters.iter().enumerate().take(5) {
            output.push_str(&format!(
                "  Cluster {} ({} episodes):\n",
                i + 1,
                cluster.len()
            ));
            for ep in cluster.iter().take(3) {
                let summary: String = ep.intent.extracted_intent.chars().take(45).collect();
                output.push_str(&format!("    - {} ({}...)\n", &ep.id[..8], summary));
            }
            if cluster.len() > 3 {
                output.push_str(&format!("    ... and {} more\n", cluster.len() - 3));
            }
        }
        if clusters.len() > 5 {
            output.push_str(&format!("  ... and {} more clusters\n", clusters.len() - 5));
        }
        output.push('\n');
    }

    match action {
        "consolidate" => {
            output.push_str("🔧 Consolidating:\n");
            let mut merged = 0;
            let mut removed = 0;

            for cluster in &clusters {
                if cluster.len() < 2 {
                    continue;
                }

                // Keep the most recent episode as the base BKM
                let mut sorted = cluster.clone();
                sorted.sort_by(|a, b| b.timestamp_end.cmp(&a.timestamp_end));

                let base = &sorted[0];
                let others = &sorted[1..];

                // Merge metadata from all others into base
                let mut updated = base.clone();
                for other in others {
                    // Union-merge tags
                    for tag in &other.intent.domain {
                        if !updated.intent.domain.contains(tag) {
                            updated.intent.domain.push(tag.clone());
                        }
                    }
                    // Union-merge files
                    for f in &other.context.files_modified {
                        if !updated.context.files_modified.contains(f) {
                            updated.context.files_modified.push(f.clone());
                        }
                    }
                    // Append unique errors
                    for err in &other.context.errors_encountered {
                        let already_has = updated
                            .context
                            .errors_encountered
                            .iter()
                            .any(|e| e.message == err.message);
                        if !already_has {
                            updated.context.errors_encountered.push(err.clone());
                        }
                    }
                    // Preserve highest helpful count
                    updated.utility.helpful_count = updated
                        .utility
                        .helpful_count
                        .max(other.utility.helpful_count);
                }

                // Save updated base
                if store.update(&updated).is_ok() {
                    let base_summary: String =
                        updated.intent.extracted_intent.chars().take(50).collect();
                    output.push_str(&format!(
                        "  ✓ Kept {} ({}...) — merged {} episodes\n",
                        &updated.id[..8],
                        base_summary,
                        others.len()
                    ));
                    merged += 1;

                    // Delete the non-base episodes
                    for other in others {
                        if store.delete(&other.id).is_ok() {
                            removed += 1;
                        }
                    }
                }
            }

            // Re-index consolidated episodes
            if merged > 0 {
                if let Ok(mut idx) = indexer::EpisodeIndexer::new().await {
                    // Re-index all remaining project episodes
                    if let Ok(remaining) = load_project_episodes(&store, &project) {
                        for ep in &remaining {
                            let _ = idx.index_episode(ep).await;
                        }
                    }
                }
            }

            if merged == 0 {
                output.push_str("  No clusters to consolidate.\n");
            } else {
                output.push_str(&format!(
                    "\n✅ Consolidated {} cluster(s), removed {} duplicate(s)\n",
                    merged, removed
                ));
            }
        }
        "cleanup" => {
            output.push_str("🧹 Cleanup Actions:\n");
            let mut removed = 0;

            // Remove stale memories with zero engagement
            for ep in &stale {
                if ep.utility.retrieval_count == 0 && ep.utility.helpful_count == 0 {
                    if store.delete(&ep.id).is_ok() {
                        removed += 1;
                        output.push_str(&format!(
                            "  ✓ Removed {} (stale, never retrieved)\n",
                            &ep.id[..8]
                        ));
                    }
                }
            }

            if removed == 0 {
                output.push_str("  No safe cleanup actions available.\n");
                output.push_str(
                    "  (Only removes stale episodes with zero engagement. Use 'consolidate' to merge duplicates.)\n",
                );
            } else {
                output.push_str(&format!("\n✅ Removed {} episode(s)\n", removed));
            }
        }
        _ => {
            // "analyze" — show recommendations
            output.push_str("💡 Recommendations:\n");

            if !clusters.is_empty() {
                output.push_str(&format!(
                    "  - {} duplicate cluster(s) found. Run with action: 'consolidate' to merge into refined BKMs.\n",
                    clusters.len()
                ));
            }
            if !stale.is_empty() {
                let stale_no_engagement: Vec<_> = stale
                    .iter()
                    .filter(|e| e.utility.retrieval_count == 0 && e.utility.helpful_count == 0)
                    .collect();
                if !stale_no_engagement.is_empty() {
                    output.push_str(&format!(
                        "  - {} stale memories with zero engagement. Run with action: 'cleanup' to remove.\n",
                        stale_no_engagement.len()
                    ));
                }
            }
            if feedback_rate < 0.2 && total_retrievals > 5 {
                output.push_str(&format!(
                    "  - Low feedback rate ({:.0}%). Use tempera_feedback after retrievals to improve BKM quality.\n",
                    feedback_rate * 100.0
                ));
            }

            let is_healthy = clusters.is_empty()
                && stale.is_empty()
                && (high_value_count > 0 || total_retrievals == 0);
            if is_healthy {
                output.push_str("  ✅ Your memory is healthy! No issues found.\n");
            }
        }
    }

    Ok(output)
}

/// Find clusters of similar episodes using vector similarity, with Jaccard fallback.
/// Returns groups of 2+ episodes that are semantically similar.
async fn find_duplicate_clusters(episodes: &[Episode], project: &str) -> Vec<Vec<Episode>> {
    // Try vector-based clustering first
    if let Some(clusters) = try_vector_clustering(episodes, project).await {
        return clusters;
    }
    // Fall back to Jaccard word similarity
    jaccard_clustering(episodes)
}

/// Vector-based duplicate detection using the embedding index
async fn try_vector_clustering(episodes: &[Episode], project: &str) -> Option<Vec<Vec<Episode>>> {
    let indexer = indexer::EpisodeIndexer::new().await.ok()?;

    if !indexer.is_indexed().await {
        return None;
    }

    // Build a similarity graph: for each episode, find similar ones
    let mut similar_pairs: Vec<(String, String)> = Vec::new();
    let episode_map: HashMap<String, &Episode> =
        episodes.iter().map(|e| (e.id.clone(), e)).collect();

    for ep in episodes {
        let query = &ep.intent.extracted_intent;
        if query.is_empty() {
            continue;
        }

        let results = indexer.search(query, 5, Some(project)).await.ok()?;

        for result in results {
            if result.id == ep.id {
                continue; // Skip self
            }
            if result.similarity_score >= CLUSTER_THRESHOLD && episode_map.contains_key(&result.id)
            {
                // Normalize pair order to avoid duplicates
                let (a, b) = if ep.id < result.id {
                    (ep.id.clone(), result.id.clone())
                } else {
                    (result.id.clone(), ep.id.clone())
                };
                if !similar_pairs.contains(&(a.clone(), b.clone())) {
                    similar_pairs.push((a, b));
                }
            }
        }
    }

    if similar_pairs.is_empty() {
        return Some(Vec::new());
    }

    // Transitive closure: group into clusters via union-find
    Some(group_into_clusters(&similar_pairs, episodes))
}

/// Jaccard word similarity fallback (original approach, improved)
fn jaccard_clustering(episodes: &[Episode]) -> Vec<Vec<Episode>> {
    let mut similar_pairs: Vec<(String, String)> = Vec::new();

    for i in 0..episodes.len() {
        for j in (i + 1)..episodes.len() {
            let e1 = &episodes[i];
            let e2 = &episodes[j];

            // Must share task type
            if e1.intent.task_type != e2.intent.task_type {
                continue;
            }

            let s1 = e1.intent.extracted_intent.to_lowercase();
            let s2 = e2.intent.extracted_intent.to_lowercase();
            let words1: HashSet<_> = s1.split_whitespace().collect();
            let words2: HashSet<_> = s2.split_whitespace().collect();
            let intersection = words1.intersection(&words2).count();
            let union = words1.union(&words2).count();

            if union > 0 && (intersection as f64 / union as f64) > 0.6 {
                similar_pairs.push((e1.id.clone(), e2.id.clone()));
            }
        }
    }

    group_into_clusters(&similar_pairs, episodes)
}

/// Group similar pairs into transitive clusters via union-find
fn group_into_clusters(pairs: &[(String, String)], episodes: &[Episode]) -> Vec<Vec<Episode>> {
    // Simple union-find via HashMap
    let mut parent: HashMap<String, String> = HashMap::new();

    let find = |parent: &mut HashMap<String, String>, id: &str| -> String {
        let mut root = id.to_string();
        while let Some(p) = parent.get(&root) {
            if p == &root {
                break;
            }
            root = p.clone();
        }
        // Path compression
        let mut current = id.to_string();
        while current != root {
            if let Some(p) = parent.get(&current).cloned() {
                parent.insert(current.clone(), root.clone());
                current = p;
            } else {
                break;
            }
        }
        root
    };

    for (a, b) in pairs {
        parent.entry(a.clone()).or_insert_with(|| a.clone());
        parent.entry(b.clone()).or_insert_with(|| b.clone());

        let root_a = find(&mut parent, a);
        let root_b = find(&mut parent, b);
        if root_a != root_b {
            parent.insert(root_b, root_a);
        }
    }

    // Group by root
    let episode_map: HashMap<String, &Episode> =
        episodes.iter().map(|e| (e.id.clone(), e)).collect();
    let mut groups: HashMap<String, Vec<Episode>> = HashMap::new();

    let all_ids: Vec<String> = parent.keys().cloned().collect();
    for id in &all_ids {
        let root = find(&mut parent, id);
        if let Some(ep) = episode_map.get(id) {
            groups.entry(root).or_default().push((*ep).clone());
        }
    }

    // Only return clusters with 2+ episodes
    groups.into_values().filter(|g| g.len() >= 2).collect()
}
