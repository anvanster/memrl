use serde_json::Value;

use crate::mcp::helpers::{extract_project, load_project_episodes};
use crate::store;

/// Check memory status for current project
pub(crate) async fn handle(args: &Value) -> Result<String, String> {
    let project = extract_project(args);
    let store = store::EpisodeStore::new().map_err(|e| e.to_string())?;
    let project_episodes = load_project_episodes(&store, &project)?;

    let total_count = project_episodes.len();

    if total_count == 0 {
        return Ok(format!(
            "📊 Memory Status for '{}'\n\
             ========================\n\n\
             No memories found for this project.\n\n\
             💡 Tip: After completing a task, use tempera_capture to save it.",
            project
        ));
    }

    // Find last capture date
    let last_capture = project_episodes
        .iter()
        .map(|e| e.timestamp_start)
        .max()
        .expect("non-empty project_episodes guaranteed by early return above");
    let days_since_capture = (chrono::Utc::now() - last_capture).num_days();

    // Find unused memories (never retrieved or not retrieved in 30+ days)
    let unused: Vec<_> = project_episodes
        .iter()
        .filter(|e| {
            if e.utility.retrieval_count == 0 {
                return true;
            }
            if let Some(last) = e.retrieval_history.last() {
                (chrono::Utc::now() - last.timestamp).num_days() > 30
            } else {
                true
            }
        })
        .collect();

    // Calculate average utility
    let avg_utility: f32 = project_episodes
        .iter()
        .map(|e| e.utility.calculate_score())
        .sum::<f32>()
        / total_count as f32;

    // Find high-value memories (lowered from 0.6 to 0.4 — Wilson at 3/3 helpful = 0.44)
    let high_value: Vec<_> = project_episodes
        .iter()
        .filter(|e| e.utility.calculate_score() > 0.4)
        .collect();

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

    let mut output = format!("📊 Memory Status for '{}'\n", project);
    output.push_str(&"=".repeat(24 + project.len()));
    output.push_str("\n\n");

    output.push_str(&format!("📁 Total memories: {}\n", total_count));
    output.push_str(&format!(
        "📅 Last capture: {} ({} days ago)\n",
        last_capture.format("%Y-%m-%d"),
        days_since_capture
    ));
    output.push_str(&format!("⭐ High-value memories: {}\n", high_value.len()));
    output.push_str(&format!("💤 Unused memories: {}\n", unused.len()));
    output.push_str(&format!(
        "📈 Average utility: {:.0}%\n",
        avg_utility * 100.0
    ));
    output.push_str(&format!(
        "🔄 Feedback rate: {} of {} retrievals ({:.0}%)\n\n",
        total_helpful,
        total_retrievals,
        feedback_rate * 100.0
    ));

    // Suggestions
    output.push_str("💡 Suggestions:\n");

    if days_since_capture > 7 {
        output.push_str("  - You haven't captured memories recently. Remember to capture after completing tasks!\n");
    }

    if unused.len() > total_count / 2 {
        output.push_str(
            "  - Many memories are unused. Consider running tempera_review to consolidate.\n",
        );
    }

    if feedback_rate < 0.2 && total_retrievals > 5 {
        output.push_str(&format!(
            "  - Low feedback rate ({:.0}%). Use tempera_feedback after retrievals to improve BKM quality.\n",
            feedback_rate * 100.0
        ));
    }

    if avg_utility < 0.3 && total_retrievals > 0 {
        output
            .push_str("  - Low average utility. Use tempera_feedback to mark helpful memories.\n");
    }

    if high_value.is_empty() {
        output.push_str(
            "  - No high-value memories yet. Keep using feedback to build utility scores.\n",
        );
    } else {
        output.push_str(&format!(
            "  - {} high-value memories ready to help with similar tasks.\n",
            high_value.len()
        ));
    }

    Ok(output)
}
