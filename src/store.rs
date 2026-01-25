#![allow(dead_code)]
use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::config::Config;
use crate::episode::{Episode, OutcomeStatus};

/// Episode store for file-based and database storage
pub struct EpisodeStore {
    episodes_dir: PathBuf,
}

impl EpisodeStore {
    /// Create a new episode store
    pub fn new() -> Result<Self> {
        let episodes_dir = Config::episodes_dir()?;
        std::fs::create_dir_all(&episodes_dir)?;
        Ok(Self { episodes_dir })
    }

    /// Save an episode to disk (both JSON and Markdown)
    pub fn save(&self, episode: &Episode) -> Result<PathBuf> {
        let date = episode.timestamp_start.format("%Y-%m-%d").to_string();
        let episode_dir = self.episodes_dir.join(&date);
        std::fs::create_dir_all(&episode_dir)?;

        // Generate filename from ID (first 8 chars)
        let id_short = &episode.id[..8.min(episode.id.len())];
        let json_path = episode_dir.join(format!("session-{}.json", id_short));
        let md_path = episode_dir.join(format!("session-{}.md", id_short));

        // Save JSON
        let json_content = serde_json::to_string_pretty(episode)?;
        std::fs::write(&json_path, json_content)?;

        // Save Markdown
        let md_content = episode.to_markdown();
        std::fs::write(&md_path, md_content)?;

        Ok(json_path)
    }

    /// Save git diff for an episode
    pub fn save_diff(&self, episode: &Episode, diff: &str) -> Result<PathBuf> {
        let date = episode.timestamp_start.format("%Y-%m-%d").to_string();
        let episode_dir = self.episodes_dir.join(&date);
        std::fs::create_dir_all(&episode_dir)?;

        let id_short = &episode.id[..8.min(episode.id.len())];
        let diff_path = episode_dir.join(format!("session-{}.diff", id_short));
        std::fs::write(&diff_path, diff)?;

        Ok(diff_path)
    }

    /// Load an episode by ID
    pub fn load(&self, id: &str) -> Result<Episode> {
        // Search through all date directories for the episode
        let entries = std::fs::read_dir(&self.episodes_dir)?;

        for entry in entries.flatten() {
            if entry.path().is_dir() {
                // Look for matching JSON file
                let pattern = format!("session-{}", &id[..8.min(id.len())]);
                let json_path = entry.path().join(format!("{}.json", pattern));

                if json_path.exists() {
                    let content = std::fs::read_to_string(&json_path)?;
                    let episode: Episode = serde_json::from_str(&content)?;
                    return Ok(episode);
                }
            }
        }

        anyhow::bail!("Episode not found: {}", id)
    }

    /// Load the latest episode
    pub fn load_latest(&self) -> Result<Episode> {
        let episodes = self.list_all()?;
        episodes
            .into_iter()
            .max_by_key(|e| e.timestamp_start)
            .context("No episodes found")
    }

    /// List all episodes
    pub fn list_all(&self) -> Result<Vec<Episode>> {
        let mut episodes = Vec::new();

        if !self.episodes_dir.exists() {
            return Ok(episodes);
        }

        let entries = std::fs::read_dir(&self.episodes_dir)?;

        for entry in entries.flatten() {
            if entry.path().is_dir() {
                // Read all JSON files in this date directory
                if let Ok(files) = std::fs::read_dir(entry.path()) {
                    for file in files.flatten() {
                        let path = file.path();
                        if path.extension().map_or(false, |e| e == "json") {
                            if let Ok(content) = std::fs::read_to_string(&path) {
                                if let Ok(episode) = serde_json::from_str::<Episode>(&content) {
                                    episodes.push(episode);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort by timestamp descending (newest first)
        episodes.sort_by(|a, b| b.timestamp_start.cmp(&a.timestamp_start));

        Ok(episodes)
    }

    /// List episodes with filters
    pub fn list_filtered(
        &self,
        limit: usize,
        project: Option<&str>,
        tag: Option<&str>,
        outcome: Option<&str>,
    ) -> Result<Vec<Episode>> {
        let all_episodes = self.list_all()?;

        let filtered: Vec<Episode> = all_episodes
            .into_iter()
            .filter(|ep| {
                // Filter by project
                if let Some(proj) = project {
                    if !ep.project.to_lowercase().contains(&proj.to_lowercase()) {
                        return false;
                    }
                }

                // Filter by tag
                if let Some(t) = tag {
                    let t_lower = t.to_lowercase();
                    if !ep
                        .intent
                        .domain
                        .iter()
                        .any(|d| d.to_lowercase().contains(&t_lower))
                    {
                        return false;
                    }
                }

                // Filter by outcome
                if let Some(o) = outcome {
                    let expected_status = match o.to_lowercase().as_str() {
                        "success" => OutcomeStatus::Success,
                        "partial" => OutcomeStatus::Partial,
                        "failure" => OutcomeStatus::Failure,
                        _ => return true, // Unknown outcome filter, allow all
                    };
                    if ep.outcome.status != expected_status {
                        return false;
                    }
                }

                true
            })
            .take(limit)
            .collect();

        Ok(filtered)
    }

    /// Update an episode
    pub fn update(&self, episode: &Episode) -> Result<()> {
        // Find and overwrite the episode file
        let entries = std::fs::read_dir(&self.episodes_dir)?;

        for entry in entries.flatten() {
            if entry.path().is_dir() {
                let pattern = format!("session-{}", &episode.id[..8.min(episode.id.len())]);
                let json_path = entry.path().join(format!("{}.json", pattern));

                if json_path.exists() {
                    let json_content = serde_json::to_string_pretty(episode)?;
                    std::fs::write(&json_path, json_content)?;

                    // Also update markdown
                    let md_path = entry.path().join(format!("{}.md", pattern));
                    let md_content = episode.to_markdown();
                    std::fs::write(&md_path, md_content)?;

                    return Ok(());
                }
            }
        }

        anyhow::bail!("Episode not found: {}", episode.id)
    }

    /// Delete an episode
    pub fn delete(&self, id: &str) -> Result<()> {
        let entries = std::fs::read_dir(&self.episodes_dir)?;

        for entry in entries.flatten() {
            if entry.path().is_dir() {
                let pattern = format!("session-{}", &id[..8.min(id.len())]);
                let json_path = entry.path().join(format!("{}.json", pattern));
                let md_path = entry.path().join(format!("{}.md", pattern));
                let diff_path = entry.path().join(format!("{}.diff", pattern));

                if json_path.exists() {
                    std::fs::remove_file(&json_path)?;
                    if md_path.exists() {
                        std::fs::remove_file(&md_path)?;
                    }
                    if diff_path.exists() {
                        std::fs::remove_file(&diff_path)?;
                    }
                    return Ok(());
                }
            }
        }

        anyhow::bail!("Episode not found: {}", id)
    }

    /// Get statistics about stored episodes
    pub fn get_stats(&self, project_filter: Option<&str>) -> Result<EpisodeStats> {
        let episodes = self.list_all()?;

        let filtered: Vec<&Episode> = episodes
            .iter()
            .filter(|ep| {
                if let Some(proj) = project_filter {
                    ep.project.to_lowercase().contains(&proj.to_lowercase())
                } else {
                    true
                }
            })
            .collect();

        let total = filtered.len();
        let success_count = filtered
            .iter()
            .filter(|e| e.outcome.status == OutcomeStatus::Success)
            .count();
        let partial_count = filtered
            .iter()
            .filter(|e| e.outcome.status == OutcomeStatus::Partial)
            .count();
        let failure_count = filtered
            .iter()
            .filter(|e| e.outcome.status == OutcomeStatus::Failure)
            .count();

        let total_retrievals: u32 = filtered.iter().map(|e| e.utility.retrieval_count).sum();
        let total_helpful: u32 = filtered.iter().map(|e| e.utility.helpful_count).sum();

        let avg_utility = if total > 0 {
            filtered
                .iter()
                .map(|e| e.utility.calculate_score())
                .sum::<f32>()
                / total as f32
        } else {
            0.0
        };

        // Collect unique projects
        let mut projects: Vec<String> = filtered.iter().map(|e| e.project.clone()).collect();
        projects.sort();
        projects.dedup();

        // Collect most common tags
        let mut tag_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for ep in &filtered {
            for tag in &ep.intent.domain {
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
            }
        }
        let mut top_tags: Vec<(String, usize)> = tag_counts.into_iter().collect();
        top_tags.sort_by(|a, b| b.1.cmp(&a.1));
        top_tags.truncate(10);

        Ok(EpisodeStats {
            total,
            success_count,
            partial_count,
            failure_count,
            total_retrievals,
            total_helpful,
            avg_utility,
            projects,
            top_tags,
        })
    }
}

/// Statistics about stored episodes
#[derive(Debug)]
pub struct EpisodeStats {
    pub total: usize,
    pub success_count: usize,
    pub partial_count: usize,
    pub failure_count: usize,
    pub total_retrievals: u32,
    pub total_helpful: u32,
    pub avg_utility: f32,
    pub projects: Vec<String>,
    pub top_tags: Vec<(String, usize)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_store() -> (EpisodeStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let store = EpisodeStore {
            episodes_dir: temp_dir.path().to_path_buf(),
        };
        (store, temp_dir)
    }

    #[test]
    fn test_save_and_load() {
        let (store, _temp) = create_test_store();
        let episode = Episode::new("test-project".to_string(), "test prompt".to_string());

        store.save(&episode).unwrap();
        let loaded = store.load(&episode.id).unwrap();

        assert_eq!(episode.id, loaded.id);
        assert_eq!(episode.project, loaded.project);
    }

    #[test]
    fn test_list_all() {
        let (store, _temp) = create_test_store();

        let ep1 = Episode::new("project1".to_string(), "prompt1".to_string());
        let ep2 = Episode::new("project2".to_string(), "prompt2".to_string());

        store.save(&ep1).unwrap();
        store.save(&ep2).unwrap();

        let all = store.list_all().unwrap();
        assert_eq!(all.len(), 2);
    }
}
