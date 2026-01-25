// Allow common clippy warnings for prototype code
#![allow(clippy::collapsible_if)]
#![allow(clippy::single_char_add_str)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::lines_filter_map_ok)]
#![allow(clippy::manual_ok_err)]
#![allow(clippy::for_kv_map)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::ptr_arg)]

use anyhow::Result;
use clap::{Parser, Subcommand};

mod capture;
mod config;
mod episode;
mod feedback;
mod indexer;
mod llm;
mod retrieve;
mod stats;
mod store;
mod utility;

#[derive(Parser)]
#[command(name = "memrl")]
#[command(about = "MemRL-inspired memory system for Claude Code")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Capture a coding session as an episode
    Capture {
        /// Path to session transcript
        #[arg(long)]
        session: Option<std::path::PathBuf>,

        /// Project directory (defaults to current)
        #[arg(long)]
        project: Option<std::path::PathBuf>,

        /// Use LLM to extract intent
        #[arg(long, default_value = "true")]
        extract_intent: bool,

        /// Capture git diff
        #[arg(long, default_value = "true")]
        capture_diff: bool,
    },

    /// Retrieve relevant episodes for a task
    Retrieve {
        /// Task description to find relevant episodes for
        query: String,

        /// Number of episodes to retrieve
        #[arg(long, short, default_value = "3")]
        limit: usize,

        /// Filter by project
        #[arg(long)]
        project: Option<String>,

        /// Output format (markdown, json)
        #[arg(long, default_value = "markdown")]
        format: String,
    },

    /// Record feedback on retrieved episodes
    Feedback {
        /// Feedback type: helpful, not-helpful, mixed
        feedback_type: String,

        /// Episode IDs (comma-separated, or "last" for last retrieved)
        #[arg(long)]
        episodes: Option<String>,
    },

    /// List episodes
    List {
        /// Number of episodes to show
        #[arg(default_value = "10")]
        limit: usize,

        /// Filter by project
        #[arg(long)]
        project: Option<String>,

        /// Filter by tag
        #[arg(long)]
        tag: Option<String>,

        /// Filter by outcome (success, partial, failure)
        #[arg(long)]
        outcome: Option<String>,
    },

    /// Show episode details
    Show {
        /// Episode ID or "latest"
        id: String,
    },

    /// Show statistics
    Stats {
        /// Filter by project
        #[arg(long)]
        project: Option<String>,
    },

    /// Index episodes for vector search (Phase 2)
    Index {
        /// Reindex all episodes
        #[arg(long)]
        reindex: bool,
    },

    /// Run Bellman utility propagation (Phase 3)
    Propagate {
        /// Also run temporal credit assignment
        #[arg(long)]
        temporal: bool,

        /// Project filter for propagation
        #[arg(long)]
        project: Option<String>,
    },

    /// Prune old/low-utility episodes
    Prune {
        /// Prune episodes older than N days
        #[arg(long)]
        older_than: Option<u32>,

        /// Prune episodes with utility below threshold
        #[arg(long)]
        min_utility: Option<f32>,

        /// Actually delete (default is dry-run)
        #[arg(long)]
        execute: bool,
    },

    /// Initialize memrl in current project
    Init,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = config::Config::load()?;

    match cli.command {
        Commands::Capture {
            session,
            project,
            extract_intent,
            capture_diff,
        } => {
            capture::run(session, project, extract_intent, capture_diff, &config).await?;
        }

        Commands::Retrieve {
            query,
            limit,
            project,
            format,
        } => {
            retrieve::run(&query, limit, project, &format, &config).await?;
        }

        Commands::Feedback {
            feedback_type,
            episodes,
        } => {
            feedback::run(&feedback_type, episodes, &config).await?;
        }

        Commands::List {
            limit,
            project,
            tag,
            outcome,
        } => {
            stats::list(limit, project, tag, outcome, &config).await?;
        }

        Commands::Show { id } => {
            stats::show(&id, &config).await?;
        }

        Commands::Stats { project } => {
            stats::run(project, &config).await?;
        }

        Commands::Index { reindex } => {
            run_index(reindex).await?;
        }

        Commands::Propagate { temporal, project } => {
            run_propagate(temporal, project).await?;
        }

        Commands::Prune {
            older_than,
            min_utility,
            execute,
        } => {
            run_prune(older_than, min_utility, execute)?;
        }

        Commands::Init => {
            init_project()?;
        }
    }

    Ok(())
}

async fn run_index(reindex: bool) -> Result<()> {
    println!("🔍 Indexing episodes for vector search...");

    if reindex {
        println!("Reindexing all episodes (this will rebuild the entire index)...");
    }

    let mut indexer = indexer::EpisodeIndexer::new().await?;
    let indexed = indexer.index_all(reindex).await?;

    // Get stats
    let stats = indexer.get_stats().await?;

    println!("\n✅ Indexing complete!");
    println!("   Episodes indexed: {}", indexed);
    println!("   Total in index: {}", stats.total_indexed);
    println!("   Embedding model: {}", stats.model_name);
    println!("   Embedding dimensions: {}", stats.embedding_dim);

    Ok(())
}

async fn run_propagate(temporal: bool, project: Option<String>) -> Result<()> {
    println!("📈 Running utility propagation...\n");

    // Run the main propagation pipeline
    let result = utility::run_propagation().await?;

    println!("\n📊 Propagation Results:");
    println!("   Episodes processed: {}", result.episodes_processed);
    println!("   Decayed: {}", result.decayed_episodes);
    println!("   Propagated: {}", result.propagated_episodes);
    println!(
        "   Total utility change: {:+.3}",
        result.total_utility_change
    );

    // Run temporal credit assignment if requested
    if temporal {
        println!("\n⏱️  Running temporal credit assignment...");
        let store = store::EpisodeStore::new()?;
        let params = utility::UtilityParams::default();
        let updated = utility::temporal_credit_assignment(&store, project.as_deref(), &params)?;
        println!("   Episodes credited: {}", updated);
    }

    println!("\n✅ Propagation complete!");
    Ok(())
}

fn run_prune(older_than: Option<u32>, min_utility: Option<f32>, execute: bool) -> Result<()> {
    println!("🗑️  Analyzing episodes for pruning...\n");

    if !execute {
        println!("📋 DRY RUN - no episodes will be deleted");
        println!("   Use --execute to actually delete\n");
    }

    let store = store::EpisodeStore::new()?;
    let result = utility::prune_episodes(&store, older_than, min_utility, !execute)?;

    if result.candidates.is_empty() {
        println!("No episodes match pruning criteria.");
    } else {
        println!("Prune candidates ({}):", result.candidates.len());
        for candidate in &result.candidates {
            println!(
                "  {} - {}... ({})",
                candidate.short_id,
                candidate.intent,
                candidate.reasons.join(", ")
            );
        }
    }

    println!("\n📊 Summary:");
    println!("   Retained: {}", result.retained);
    if execute {
        println!("   Pruned: {}", result.pruned);
    } else {
        println!("   Would prune: {}", result.candidates.len());
    }

    println!("\n✅ Prune complete!");
    Ok(())
}

fn init_project() -> Result<()> {
    use std::fs;

    let memrl_dir = dirs::home_dir()
        .expect("Could not find home directory")
        .join(".memrl");

    // Create directories
    fs::create_dir_all(memrl_dir.join("episodes"))?;
    println!("✓ Created {}", memrl_dir.display());

    // Create today's directory
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
    fs::create_dir_all(memrl_dir.join("episodes").join(&today))?;
    println!("✓ Created episodes/{}", today);

    // Initialize feedback log
    let feedback_path = memrl_dir.join("feedback.log");
    if !feedback_path.exists() {
        fs::write(&feedback_path, "")?;
        println!("✓ Initialized feedback log");
    }

    // Create config if not exists
    let config_path = memrl_dir.join("config.toml");
    if !config_path.exists() {
        let default_config = include_str!("../default_config.toml");
        fs::write(&config_path, default_config)?;
        println!("✓ Created default config");
    }

    println!("\n🎉 MemRL initialized!");
    println!("\nNext steps:");
    println!("  memrl capture --session /path/to/transcript");
    println!("  memrl retrieve \"your task description\"");

    Ok(())
}
