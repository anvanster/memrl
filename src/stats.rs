#![allow(dead_code)]
use anyhow::Result;
use colored::Colorize;
use tabled::{Table, Tabled};

use crate::config::Config;
use crate::episode::OutcomeStatus;
use crate::store::EpisodeStore;

/// List episodes
pub async fn list(
    limit: usize,
    project: Option<String>,
    tag: Option<String>,
    outcome: Option<String>,
    _config: &Config,
) -> Result<()> {
    let store = EpisodeStore::new()?;

    let episodes = store.list_filtered(
        limit,
        project.as_deref(),
        tag.as_deref(),
        outcome.as_deref(),
    )?;

    if episodes.is_empty() {
        println!("No episodes found.");
        return Ok(());
    }

    println!("{}", "📚 Episodes".bold());
    println!();

    // Convert to table rows
    let rows: Vec<EpisodeRow> = episodes
        .iter()
        .map(|ep| EpisodeRow {
            id: ep.id[..8].to_string(),
            date: ep.timestamp_start.format("%Y-%m-%d").to_string(),
            project: truncate(&ep.project, 15),
            intent: truncate(
                if ep.intent.extracted_intent.is_empty() {
                    &ep.intent.raw_prompt
                } else {
                    &ep.intent.extracted_intent
                },
                40,
            ),
            outcome: format_outcome(&ep.outcome.status),
            utility: format!("{:.0}%", ep.utility.calculate_score() * 100.0),
            retrievals: ep.utility.retrieval_count.to_string(),
        })
        .collect();

    let table = Table::new(rows).to_string();
    println!("{}", table);

    Ok(())
}

/// Show a single episode in detail
pub async fn show(id: &str, _config: &Config) -> Result<()> {
    let store = EpisodeStore::new()?;

    let episode = if id.to_lowercase() == "latest" || id.to_lowercase() == "last" {
        store.load_latest()?
    } else {
        store.load(id)?
    };

    // Print episode details
    println!("{}", "📄 Episode Details".bold());
    println!();
    println!("{}", episode.to_markdown());

    // Additional details not in markdown
    println!("{}", "## Utility Metrics".bold());
    println!("Retrieval count: {}", episode.utility.retrieval_count);
    println!("Helpful count: {}", episode.utility.helpful_count);
    println!(
        "Utility score: {:.2}%",
        episode.utility.calculate_score() * 100.0
    );

    Ok(())
}

/// Show statistics
pub async fn run(project: Option<String>, _config: &Config) -> Result<()> {
    let store = EpisodeStore::new()?;
    let stats = store.get_stats(project.as_deref())?;

    println!("{}", "📊 MemRL Statistics".bold());
    println!();

    if let Some(ref proj) = project {
        println!("Filter: project = {}", proj);
        println!();
    }

    // Overview
    println!("{}", "Overview".underline());
    println!("Total episodes: {}", stats.total);
    println!(
        "Success rate: {:.1}%",
        if stats.total > 0 {
            (stats.success_count as f32 / stats.total as f32) * 100.0
        } else {
            0.0
        }
    );
    println!();

    // Outcome breakdown
    println!("{}", "Outcomes".underline());
    println!(
        "  ✅ Success: {} ({:.1}%)",
        stats.success_count,
        percentage(stats.success_count, stats.total)
    );
    println!(
        "  ⚠️  Partial: {} ({:.1}%)",
        stats.partial_count,
        percentage(stats.partial_count, stats.total)
    );
    println!(
        "  ❌ Failure: {} ({:.1}%)",
        stats.failure_count,
        percentage(stats.failure_count, stats.total)
    );
    println!();

    // Utility metrics
    println!("{}", "Utility Metrics".underline());
    println!("Total retrievals: {}", stats.total_retrievals);
    println!("Total helpful: {}", stats.total_helpful);
    println!(
        "Helpful rate: {:.1}%",
        if stats.total_retrievals > 0 {
            (stats.total_helpful as f32 / stats.total_retrievals as f32) * 100.0
        } else {
            0.0
        }
    );
    println!("Average utility score: {:.1}%", stats.avg_utility * 100.0);
    println!();

    // Projects
    if !stats.projects.is_empty() {
        println!("{}", "Projects".underline());
        for proj in &stats.projects {
            println!("  - {}", proj);
        }
        println!();
    }

    // Top tags
    if !stats.top_tags.is_empty() {
        println!("{}", "Top Tags".underline());
        for (tag, count) in &stats.top_tags {
            println!("  {} ({})", tag, count);
        }
    }

    Ok(())
}

/// Table row for episode list
#[derive(Tabled)]
struct EpisodeRow {
    #[tabled(rename = "ID")]
    id: String,
    #[tabled(rename = "Date")]
    date: String,
    #[tabled(rename = "Project")]
    project: String,
    #[tabled(rename = "Intent")]
    intent: String,
    #[tabled(rename = "Outcome")]
    outcome: String,
    #[tabled(rename = "Utility")]
    utility: String,
    #[tabled(rename = "Retriev")]
    retrievals: String,
}

/// Truncate string to max length
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Format outcome status
fn format_outcome(status: &OutcomeStatus) -> String {
    match status {
        OutcomeStatus::Success => "✅".to_string(),
        OutcomeStatus::Partial => "⚠️".to_string(),
        OutcomeStatus::Failure => "❌".to_string(),
    }
}

/// Calculate percentage
fn percentage(part: usize, total: usize) -> f32 {
    if total > 0 {
        (part as f32 / total as f32) * 100.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("this is a very long string", 10), "this is...");
    }

    #[test]
    fn test_percentage() {
        assert_eq!(percentage(50, 100), 50.0);
        assert_eq!(percentage(0, 0), 0.0);
    }
}
