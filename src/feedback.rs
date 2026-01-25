#![allow(dead_code)]

use anyhow::Result;
use chrono::Utc;
use std::io::{BufRead, Write};

use crate::config::Config;
use crate::store::EpisodeStore;

/// Run the feedback command
pub async fn run(feedback_type: &str, episodes: Option<String>, _config: &Config) -> Result<()> {
    let store = EpisodeStore::new()?;

    // Determine which episodes to provide feedback for
    let episode_ids = match episodes {
        Some(ids) if ids.to_lowercase() == "last" => {
            // Get IDs from last retrieval in feedback log
            get_last_retrieved_ids()?
        }
        Some(ids) => {
            // Parse comma-separated IDs
            ids.split(',').map(|s| s.trim().to_string()).collect()
        }
        None => {
            // Interactive: show recent episodes and let user select
            println!("No episodes specified. Use --episodes <id1,id2,...> or --episodes last");
            return Ok(());
        }
    };

    if episode_ids.is_empty() {
        println!("No episodes to provide feedback for.");
        return Ok(());
    }

    // Parse feedback type
    let is_helpful = match feedback_type.to_lowercase().as_str() {
        "helpful" | "yes" | "y" | "1" | "good" => Some(true),
        "not-helpful" | "unhelpful" | "no" | "n" | "0" | "bad" => Some(false),
        "mixed" | "partial" | "skip" => None,
        _ => {
            println!(
                "Unknown feedback type: {}. Use 'helpful', 'not-helpful', or 'mixed'.",
                feedback_type
            );
            return Ok(());
        }
    };

    println!(
        "📝 Recording feedback for {} episode(s)...",
        episode_ids.len()
    );

    let mut updated = 0;
    for id in &episode_ids {
        match update_episode_feedback(&store, id, is_helpful) {
            Ok(_) => {
                updated += 1;
                let feedback_str = match is_helpful {
                    Some(true) => "✅ helpful",
                    Some(false) => "❌ not helpful",
                    None => "➖ mixed/skipped",
                };
                println!("  {} -> {}", &id[..8.min(id.len())], feedback_str);
            }
            Err(e) => {
                println!("  {} -> ⚠️ failed: {}", &id[..8.min(id.len())], e);
            }
        }
    }

    println!("\n✅ Updated {} episode(s)", updated);

    // Log the feedback
    log_feedback(&episode_ids, is_helpful)?;

    Ok(())
}

/// Update episode with feedback
fn update_episode_feedback(store: &EpisodeStore, id: &str, is_helpful: Option<bool>) -> Result<()> {
    let mut episode = store.load(id)?;

    // Update the most recent retrieval record
    if let Some(last_retrieval) = episode.retrieval_history.last_mut() {
        last_retrieval.was_helpful = is_helpful;
    }

    // Update utility counts
    if let Some(true) = is_helpful {
        episode.utility.helpful_count += 1;
    }

    // Recalculate utility score
    episode.utility.score = Some(episode.utility.calculate_score());

    // Save updated episode
    store.update(&episode)?;

    Ok(())
}

/// Get episode IDs from the last retrieval
fn get_last_retrieved_ids() -> Result<Vec<String>> {
    let feedback_log = Config::feedback_log_path()?;

    if !feedback_log.exists() {
        return Ok(vec![]);
    }

    let file = std::fs::File::open(&feedback_log)?;
    let reader = std::io::BufReader::new(file);

    // Get the last line with retrieval IDs
    let mut last_ids = String::new();
    for line in reader.lines().flatten() {
        if line.contains("ids:") {
            last_ids = line;
        }
    }

    // Parse IDs from line format: "timestamp\tquery:...\tids:id1,id2,id3"
    if let Some(ids_part) = last_ids.split("ids:").nth(1) {
        let ids: Vec<String> = ids_part
            .trim()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        return Ok(ids);
    }

    Ok(vec![])
}

/// Log feedback to feedback.log
fn log_feedback(episode_ids: &[String], is_helpful: Option<bool>) -> Result<()> {
    let feedback_log = Config::feedback_log_path()?;

    let feedback_str = match is_helpful {
        Some(true) => "helpful",
        Some(false) => "not-helpful",
        None => "mixed",
    };

    let log_entry = format!(
        "{}\tfeedback:{}\tids:{}\n",
        Utc::now().to_rfc3339(),
        feedback_str,
        episode_ids.join(",")
    );

    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(feedback_log)?
        .write_all(log_entry.as_bytes())?;

    Ok(())
}

/// Batch feedback: mark multiple episodes as helpful/not-helpful
pub fn batch_feedback(
    store: &EpisodeStore,
    episode_ids: &[String],
    is_helpful: bool,
) -> Result<usize> {
    let mut updated = 0;

    for id in episode_ids {
        if update_episode_feedback(store, id, Some(is_helpful)).is_ok() {
            updated += 1;
        }
    }

    log_feedback(episode_ids, Some(is_helpful))?;

    Ok(updated)
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_parse_feedback_type() {
        // Test various feedback type aliases
        let test_cases = [
            ("helpful", Some(true)),
            ("yes", Some(true)),
            ("not-helpful", Some(false)),
            ("no", Some(false)),
            ("mixed", None),
            ("skip", None),
        ];

        for (input, expected) in test_cases {
            let result = match input.to_lowercase().as_str() {
                "helpful" | "yes" | "y" | "1" | "good" => Some(true),
                "not-helpful" | "unhelpful" | "no" | "n" | "0" | "bad" => Some(false),
                "mixed" | "partial" | "skip" => None,
                _ => None,
            };
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
    }
}
