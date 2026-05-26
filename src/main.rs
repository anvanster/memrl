// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

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
use clap::{Args, Parser, Subcommand};

mod ask_back_gen;
mod ask_backs;
mod asks;
mod backup;
mod brief;
mod calibration;
mod capture;
mod config;
mod contradict;
mod doctor;
mod dream;
mod episode;
mod eval;
mod feedback;
mod fingerprint;
mod indexer;
mod jobs;
mod keyword;
mod llm;
mod mistakes;
mod patterns;
mod reflect;
mod retrieve;
mod stats;
mod store;
mod templates;
mod templates_phase;
mod triage;
mod utility;

#[derive(Parser)]
#[command(name = "tempera")]
#[command(about = "Tempera - persistent memory system for Claude Code")]
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

    /// Show trend analytics (helpfulness over time, domain growth, learning curve)
    Trends {
        /// Filter by project
        #[arg(long)]
        project: Option<String>,

        /// Bucket size: "weekly" or "monthly"
        #[arg(long, default_value = "weekly")]
        bucket: String,
    },

    /// Initialize tempera in current project
    Init,

    /// Retrieval evaluation harness (P@K, R@K, MRR, nDCG@K)
    Eval(EvalArgs),

    /// Run the background job daemon (foreground, Ctrl+C to stop)
    Daemon,

    /// Manage background jobs (submit, list)
    Job(JobArgs),

    /// Snapshot or restore the tempera data directory
    Backup(BackupArgs),

    /// Health check + optional auto-remediation. By default just
    /// reports the current score with hints to lift it. v0.7.6 adds
    /// `--remediate` which walks a dependency-ordered plan to fix
    /// what's lifting-able (currently: reindex for low coverage).
    Doctor {
        /// Emit machine-readable JSON instead of the colored summary
        #[arg(long)]
        json: bool,

        /// Show the remediation plan but don't run anything.
        #[arg(long)]
        remediation_plan: bool,

        /// Actually run the remediation steps. Requires --yes.
        #[arg(long)]
        remediate: bool,

        /// Confirms --remediate. Without it the runner refuses to apply.
        #[arg(long)]
        yes: bool,

        /// Stop once the score reaches this value. Default 90.
        #[arg(long, default_value_t = 90)]
        target_score: u32,

        /// Dollar cap for remediation. Default $0.50.
        #[arg(long, default_value_t = 0.50)]
        max_usd: f32,
    },

    /// Author a reflection page for a single day. Runs Haiku triage
    /// first; if score < 0.5 the call is skipped. Otherwise Sonnet writes
    /// a short reflection page that goes into ~/.tempera/reflections/
    /// and an SQLite mirror.
    Reflect {
        /// Day to reflect on (YYYY-MM-DD). Defaults to yesterday (the
        /// natural target for a nightly cron — today is incomplete).
        #[arg(long)]
        date: Option<String>,

        /// Skip the cache and re-author even if a reflection exists.
        #[arg(long)]
        force: bool,

        /// Plan only: show triage signals without making the Sonnet call.
        #[arg(long)]
        dry_run: bool,

        /// Emit the reflection record as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Run the contradiction probe. Pairs frequently-retrieved episodes
    /// whose embeddings are related-but-not-duplicate and asks Haiku
    /// to judge whether they contradict on a factual claim. Surfaces
    /// active findings + a Wilson 95% CI on the rate.
    Contradict {
        /// Cap on judge calls per run. Overrides
        /// config.dream.contradict_max_pairs.
        #[arg(long)]
        max_pairs: Option<usize>,

        /// Plan only — show how many pairs would be sent without
        /// making any LLM calls.
        #[arg(long)]
        dry_run: bool,

        /// List active (unresolved) contradictions instead of probing.
        #[arg(long)]
        list: bool,

        /// Emit the report as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Cluster recent reflections and surface cross-day patterns. Reads
    /// the last N days of reflections (config.dream.patterns_lookback_days,
    /// default 30), embeds each, agglomerative-clusters at the configured
    /// cosine threshold, and asks Sonnet to name the shared theme for any
    /// cluster with `>= patterns_min_evidence` members.
    Patterns {
        /// Override patterns_lookback_days for this run.
        #[arg(long)]
        since_days: Option<u32>,

        /// Show what would run + planned clusters; no LLM call.
        #[arg(long)]
        dry_run: bool,

        /// Emit the report as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Triage a single day's captures (Haiku) to decide whether they're
    /// worth synthesizing. Caches results per (date, content-hash) so
    /// re-runs on unchanged days are free.
    Triage {
        /// Day to triage in YYYY-MM-DD form. Defaults to today.
        #[arg(long)]
        date: Option<String>,

        /// Skip the cache and re-call Haiku.
        #[arg(long)]
        force: bool,

        /// Emit the verdict as JSON instead of the colored summary.
        #[arg(long)]
        json: bool,
    },

    /// Run the dream cycle (v0.7+ orchestrator of background phases).
    /// v0.7.1 ships verify_advance + decay. More phases land in
    /// subsequent point releases (reflect, patterns, contradict, embed).
    Dream {
        /// Run a single phase instead of the full cycle. Try
        /// `tempera dream --list` to see available phases.
        #[arg(long)]
        phase: Option<String>,

        /// Plan only — print what would run + estimated cost, don't execute.
        #[arg(long)]
        dry_run: bool,

        /// Dollar cap for this run. Overrides config.dream.default_max_usd.
        #[arg(long)]
        max_usd: Option<f32>,

        /// List available phases and exit.
        #[arg(long)]
        list: bool,

        /// Emit a JSON report instead of the human summary.
        #[arg(long)]
        json: bool,
    },

    /// Log a correction the user made to an assumption / decision /
    /// piece of code. Mirrors the `tempera_log_correction` MCP tool
    /// so humans can record offline corrections too.
    LogCorrection {
        /// Topic label (will be normalized to lowercase snake_case).
        #[arg(long)]
        category: String,

        /// One sentence: what was wrong.
        #[arg(long)]
        description: String,

        /// What the user said the right thing was.
        #[arg(long)]
        correction: String,

        /// Optional episode id (full or 8-char prefix).
        #[arg(long)]
        episode: Option<String>,

        /// Comma-separated list of files involved.
        #[arg(long)]
        files: Option<String>,

        /// Override project name (default: auto-detect from CWD).
        #[arg(long)]
        project: Option<String>,
    },

    /// View the anchored-mistakes index. Filter by project/category,
    /// or show `--top` to see the most-corrected categories per
    /// project.
    Mistakes {
        /// Filter by project (default: all projects).
        #[arg(long)]
        project: Option<String>,

        /// Filter by category.
        #[arg(long)]
        category: Option<String>,

        /// Show top-N categories by count instead of the raw list.
        #[arg(long)]
        top: Option<i64>,

        /// Result limit for the raw list.
        #[arg(long, default_value_t = 50)]
        limit: i64,

        /// Emit as JSON.
        #[arg(long)]
        json: bool,
    },

    /// View the per-bucket calibration profile — how often this agent's
    /// declared "success" claims actually survive into StableNoRevert.
    /// Buckets are keyed by `(task_type, project)`. v0.8.1 surfaces the
    /// data; v0.8.x will apply the overconfidence rate at retrieval time.
    Calibration {
        /// Filter by task type (e.g. bugfix, feature, refactor).
        #[arg(long)]
        task_type: Option<String>,

        /// Filter by project name.
        #[arg(long)]
        project: Option<String>,

        /// Emit buckets as JSON.
        #[arg(long)]
        json: bool,
    },

    /// One-call summary joining the working set against every v0.8
    /// surface — pending ask-backs, reasoning templates, top correction
    /// categories for these files, should-have-asked triggers,
    /// calibration warnings (v0.9).
    Brief {
        /// Comma-separated list of files about to be edited.
        #[arg(long)]
        files: String,

        /// Optional task type: bugfix | feature | refactor | test |
        /// docs | research | debug | setup. Required for the
        /// calibration warning + template sections.
        #[arg(long)]
        task_type: Option<String>,

        /// Optional domain tag, paired with task_type to look up a
        /// reasoning template.
        #[arg(long)]
        domain: Option<String>,

        /// Override project name (default: auto-detect from CWD).
        #[arg(long)]
        project: Option<String>,

        /// Emit the structured Brief as JSON instead of formatted text.
        #[arg(long)]
        json: bool,
    },

    /// Show the pending ask-back for a project, if any, and mark it
    /// served (v0.8.5). Mirrors the `tempera_session_start` MCP tool
    /// so humans can see what's queued.
    SessionStart {
        #[arg(long)]
        project: Option<String>,
    },

    /// View ask-back history (v0.8.5) — questions drafted by capture
    /// after vague-intent failures. Filter by project or status.
    AskBacks {
        #[arg(long)]
        project: Option<String>,

        /// Show only pending (default: all).
        #[arg(long)]
        pending: bool,

        #[arg(long, default_value_t = 50)]
        limit: i64,

        #[arg(long)]
        json: bool,
    },

    /// Log a question you should have asked up front but didn't, and
    /// the answer you eventually got (v0.8.4). Mirrors the
    /// `tempera_log_should_have_asked` MCP tool for humans who realize
    /// the gap offline.
    LogShouldHaveAsked {
        /// Observable context label (normalized to snake_case).
        #[arg(long)]
        trigger: String,

        /// The question you should have asked.
        #[arg(long)]
        question: String,

        /// The answer that turned out to be true.
        #[arg(long)]
        answer: String,

        /// Optional episode id (full or 8-char prefix).
        #[arg(long)]
        episode: Option<String>,

        /// Comma-separated list of files involved.
        #[arg(long)]
        files: Option<String>,

        /// Override project name (default: auto-detect from CWD).
        #[arg(long)]
        project: Option<String>,
    },

    /// View the should-have-asked log. Filter by project/trigger, or
    /// `--top` for most-frequent triggers.
    Asks {
        #[arg(long)]
        project: Option<String>,

        #[arg(long)]
        trigger: Option<String>,

        #[arg(long)]
        top: Option<i64>,

        #[arg(long, default_value_t = 50)]
        limit: i64,

        #[arg(long)]
        json: bool,
    },

    /// Inspect or extract reasoning templates (v0.8.3). Templates are
    /// stored per `(task_type, domain)` pair and surface via the
    /// `tempera_template` MCP tool; this CLI exposes them for humans
    /// and offers a manual extraction trigger.
    Templates {
        #[command(subcommand)]
        command: TemplatesCommand,
    },

    /// Move an episode's verification state forward (manual; future
    /// versions add git/test hooks that do this automatically)
    AdvanceVerification {
        /// Episode id (full or 8-char short prefix)
        #[arg(long)]
        episode: String,

        /// Target state: untested | tests_pass | merged | stable_no_revert
        #[arg(long)]
        to: String,

        /// Commit SHA (required when --to merged)
        #[arg(long)]
        commit: Option<String>,

        /// Test run id (required when --to tests_pass)
        #[arg(long)]
        run_id: Option<String>,

        /// Days stable (used when --to stable_no_revert; default 1)
        #[arg(long)]
        days: Option<u32>,
    },
}

#[derive(Subcommand)]
enum TemplatesCommand {
    /// List all stored templates (most recent first).
    List {
        /// Filter by task type.
        #[arg(long)]
        task_type: Option<String>,

        /// Filter by domain.
        #[arg(long)]
        domain: Option<String>,

        /// Emit as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Fetch a single template by (task_type, domain).
    Get {
        #[arg(long)]
        task_type: String,
        #[arg(long)]
        domain: String,

        #[arg(long)]
        json: bool,
    },

    /// Run the templates dream phase right now (Sonnet-backed).
    /// Bounded by `--max-usd`.
    Extract {
        /// Dollar cap (default: dream.default_max_usd from config).
        #[arg(long)]
        max_usd: Option<f32>,

        /// Print what would be done without spending budget.
        #[arg(long)]
        dry_run: bool,
    },
}

#[derive(Args)]
struct BackupArgs {
    /// Restore a previous snapshot by timestamp (e.g. 20260524T123456Z)
    #[arg(long)]
    restore: Option<String>,

    /// List available snapshots
    #[arg(long)]
    list: bool,

    /// With --restore: skip safety checks (empty snapshot, etc.)
    #[arg(long)]
    force: bool,
}

#[derive(Args)]
struct JobArgs {
    #[command(subcommand)]
    command: JobCommand,
}

#[derive(Subcommand)]
enum JobCommand {
    /// Submit a new job to the queue
    Submit {
        /// Job kind (currently: index | propagate)
        kind: String,

        /// JSON payload for the handler (default: {})
        #[arg(long, default_value = "{}")]
        payload: String,
    },

    /// List jobs in the queue
    List {
        /// Filter by status: pending | running | completed | dead
        #[arg(long)]
        status: Option<String>,

        /// Maximum rows to show
        #[arg(long, default_value_t = 20)]
        limit: i64,
    },
}

#[derive(Args)]
struct EvalArgs {
    #[command(subcommand)]
    command: EvalCommand,
}

#[derive(Subcommand)]
enum EvalCommand {
    /// Run the fixture and save the result as a new baseline.
    Baseline {
        /// Path to a JSONL fixture file
        #[arg(long)]
        fixture: std::path::PathBuf,

        /// Top-K cutoff for metrics
        #[arg(long, default_value_t = eval::DEFAULT_K)]
        k: usize,

        /// Override retrieval mode for this run (vector|keyword|hybrid).
        /// If omitted, uses config.retrieval.mode.
        #[arg(long)]
        mode: Option<String>,
    },

    /// Run the fixture and diff against the most recent baseline.
    Run {
        /// Path to a JSONL fixture file
        #[arg(long)]
        fixture: std::path::PathBuf,

        /// Top-K cutoff for metrics
        #[arg(long, default_value_t = eval::DEFAULT_K)]
        k: usize,

        /// Also persist this run as a baseline alongside the diff
        #[arg(long)]
        save: bool,

        /// Override retrieval mode for this run (vector|keyword|hybrid).
        /// If omitted, uses config.retrieval.mode.
        #[arg(long)]
        mode: Option<String>,
    },
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
            run_propagate(temporal, project, &config).await?;
        }

        Commands::Prune {
            older_than,
            min_utility,
            execute,
        } => {
            run_prune(older_than, min_utility, execute, &config)?;
        }

        Commands::Trends { project, bucket } => {
            stats::trends(project, &bucket, &config).await?;
        }

        Commands::Init => {
            init_project()?;
        }

        Commands::Eval(args) => match args.command {
            EvalCommand::Baseline { fixture, k, mode } => {
                let cfg = override_mode(&config, mode.as_deref())?;
                eval::run_baseline(&fixture, k, &cfg).await?;
            }
            EvalCommand::Run {
                fixture,
                k,
                save,
                mode,
            } => {
                let cfg = override_mode(&config, mode.as_deref())?;
                eval::run_against_baseline(&fixture, k, save, &cfg).await?;
            }
        },

        Commands::Daemon => {
            let queue = jobs::JobQueue::open_default().await?;
            jobs::run_daemon(&queue, &config).await?;
        }

        Commands::Contradict {
            max_pairs,
            dry_run,
            list,
            json,
        } => {
            if list {
                let store = contradict::ContradictionStore::open_default().await?;
                let active = store.list_active(100).await?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&active)?);
                } else if active.is_empty() {
                    println!("No active contradictions.");
                } else {
                    use colored::Colorize;
                    println!();
                    println!(
                        "{} active contradiction(s):",
                        active.len().to_string().bold()
                    );
                    for c in active.iter().take(20) {
                        let sev_colored = match c.severity {
                            contradict::Severity::High => c.severity.as_str().red(),
                            contradict::Severity::Medium => c.severity.as_str().yellow(),
                            contradict::Severity::Low => c.severity.as_str().normal(),
                        };
                        let a8 = &c.episode_a[..8.min(c.episode_a.len())];
                        let b8 = &c.episode_b[..8.min(c.episode_b.len())];
                        println!(
                            "  [{}] {} ↔ {} (sim {:.2}, conf {:.2}): {}",
                            sev_colored, a8, b8, c.similarity, c.confidence, c.explanation
                        );
                        if let Some(hint) = &c.resolution_hint {
                            println!("    hint: {}", hint);
                        }
                    }
                    println!();
                }
                return Ok(());
            }
            let mut cfg = config.clone();
            if let Some(n) = max_pairs {
                cfg.dream.contradict_max_pairs = n;
            }
            if dry_run {
                let plan = serde_json::json!({
                    "top_n_candidates": cfg.dream.contradict_top_n,
                    "min_similarity": cfg.dream.contradict_min_similarity,
                    "max_similarity": cfg.dream.contradict_max_similarity,
                    "max_pairs": cfg.dream.contradict_max_pairs,
                    "estimated_cost_usd_worst_case":
                        contradict::JUDGE_ESTIMATED_COST_USD
                            * cfg.dream.contradict_max_pairs as f32,
                    "judge_model": &cfg.dream.triage_model,
                });
                println!("{}", serde_json::to_string_pretty(&plan)?);
                return Ok(());
            }
            let budget = dream::CostBudget::new(cfg.dream.default_max_usd);
            let report = contradict::run_probe(&cfg, Some(&budget)).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!();
                println!("Contradiction probe");
                println!("  pairs evaluated:       {}", report.pairs_evaluated);
                println!("  contradictions found:  {}", report.contradictions_found);
                if report.pairs_evaluated > 0 {
                    let rate = report.contradictions_found as f32 / report.pairs_evaluated as f32;
                    println!(
                        "  rate:                  {:.1}%  (Wilson 95% CI {:.0}% – {:.0}%)",
                        rate * 100.0,
                        report.rate_ci_lower * 100.0,
                        report.rate_ci_upper * 100.0,
                    );
                }
                println!(
                    "  by severity:           high={} medium={} low={}",
                    report.by_severity.high, report.by_severity.medium, report.by_severity.low
                );
                if report.small_sample {
                    println!("  note: small sample — CI is wide; treat as directional.");
                }
                println!();
            }
        }

        Commands::Patterns {
            since_days,
            dry_run,
            json,
        } => {
            let mut cfg = config.clone();
            if let Some(d) = since_days {
                cfg.dream.patterns_lookback_days = d;
            }
            if dry_run {
                let cutoff = chrono::Utc::now()
                    - chrono::Duration::days(cfg.dream.patterns_lookback_days as i64);
                let reflect_store = reflect::ReflectionStore::open_default().await?;
                let reflections = reflect_store.list_since(&cutoff.date_naive()).await?;
                let plan = serde_json::json!({
                    "lookback_days": cfg.dream.patterns_lookback_days,
                    "min_evidence": cfg.dream.patterns_min_evidence,
                    "cluster_threshold": cfg.dream.patterns_cluster_threshold,
                    "reflections_in_window": reflections.len(),
                    "would_cluster": reflections.len() >= cfg.dream.patterns_min_evidence,
                    "estimated_cost_usd_worst_case": patterns::PATTERN_ESTIMATED_COST_USD * 3.0,
                });
                println!("{}", serde_json::to_string_pretty(&plan)?);
                return Ok(());
            }
            let budget = dream::CostBudget::new(cfg.dream.default_max_usd);
            let report = patterns::run_patterns(&cfg, Some(&budget)).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!();
                println!("Patterns phase");
                println!("  reflections examined:  {}", report.reflections_examined);
                println!("  clusters found:        {}", report.clusters_found);
                println!(
                    "  clusters above min:    {} (min_evidence={})",
                    report.clusters_above_min, cfg.dream.patterns_min_evidence
                );
                println!("  patterns written:      {}", report.patterns_written);
                println!(
                    "  patterns already exist: {}",
                    report.patterns_skipped_existing
                );
                println!("  clusters w/o theme:    {}", report.clusters_with_no_theme);
                println!();
            }
        }

        Commands::Reflect {
            date,
            force,
            dry_run,
            json,
        } => {
            let target_date = match date.as_deref() {
                Some(s) => chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
                    .map_err(|e| anyhow::anyhow!("invalid --date '{s}': {e}"))?,
                None => (chrono::Utc::now() - chrono::Duration::days(1)).date_naive(),
            };
            let store = store::EpisodeStore::new()?;
            let day_eps: Vec<_> = store
                .list_all()?
                .into_iter()
                .filter(|e| e.timestamp_start.date_naive() == target_date)
                .collect();
            let reflect_store = reflect::ReflectionStore::open_default().await?;
            let id = reflect::Reflection::id_for(&target_date, None);
            if !force && let Some(existing) = reflect_store.get(&id).await? {
                if json {
                    println!("{}", serde_json::to_string_pretty(&existing)?);
                } else {
                    println!(
                        "Reflection {} already exists ({} citations). Pass --force to re-author.",
                        existing.id,
                        existing.citations.len()
                    );
                }
                return Ok(());
            }

            let triage_store = triage::TriageStore::open_default().await?;
            let budget = dream::CostBudget::new(config.dream.default_max_usd);
            let (verdict, from_cache) = triage::triage_day_with_model(
                &target_date,
                &day_eps,
                &triage_store,
                Some(&budget),
                false,
                &config.dream.triage_model,
            )
            .await?;

            if dry_run {
                let plan = serde_json::json!({
                    "date": target_date,
                    "episode_count": day_eps.len(),
                    "triage": &verdict,
                    "triage_from_cache": from_cache,
                    "would_author": verdict.worth_synthesizing(),
                    "estimated_cost_usd": if verdict.worth_synthesizing() {
                        reflect::REFLECT_ESTIMATED_COST_USD
                    } else {
                        0.0
                    },
                });
                println!("{}", serde_json::to_string_pretty(&plan)?);
                return Ok(());
            }

            if !verdict.worth_synthesizing() {
                println!(
                    "Triage score {:.2} below threshold — skipping authorship. \
                     Use --force to author anyway.",
                    verdict.score
                );
                return Ok(());
            }

            let reflection = reflect::author_reflection(
                &target_date,
                &day_eps,
                &verdict.signals,
                verdict.score,
                &config,
                Some(&budget),
                &reflect_store,
            )
            .await?;

            if json {
                println!("{}", serde_json::to_string_pretty(&reflection)?);
            } else {
                println!();
                println!("Authored reflection: {}", reflection.id);
                println!("  citations: {}", reflection.citations.join(", "));
                println!("  signals:   {}", reflection.signals.join(", "));
                println!("  model:     {}", reflection.model);
                println!("  sidecar:   ~/.tempera/reflections/{}.md", reflection.id);
                println!();
                println!("--- body ---");
                println!("{}", reflection.body);
                println!();
            }
        }

        Commands::Triage { date, force, json } => {
            let target_date = match date.as_deref() {
                Some(s) => chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
                    .map_err(|e| anyhow::anyhow!("invalid --date '{s}': {e}"))?,
                None => chrono::Utc::now().date_naive(),
            };
            let store = store::EpisodeStore::new()?;
            let all = store.list_all()?;
            let for_day: Vec<_> = all
                .into_iter()
                .filter(|e| e.timestamp_start.date_naive() == target_date)
                .collect();
            let triage_store = triage::TriageStore::open_default().await?;
            let budget = dream::CostBudget::new(config.dream.default_max_usd);
            let (verdict, from_cache) =
                triage::triage_day(&target_date, &for_day, &triage_store, Some(&budget), force)
                    .await?;
            if json {
                let body = serde_json::json!({
                    "date": target_date,
                    "episode_count": for_day.len(),
                    "from_cache": from_cache,
                    "verdict": verdict,
                });
                println!("{}", serde_json::to_string_pretty(&body)?);
            } else {
                use colored::Colorize;
                let header = format!(
                    "{} ({} episode(s), {})",
                    target_date,
                    for_day.len(),
                    if from_cache { "cached" } else { "fresh" }
                );
                println!();
                println!("{}", "Triage verdict".bold());
                println!("  {header}");
                let score_str = format!("{:.2}", verdict.score);
                let score_colored = if verdict.score >= 0.7 {
                    score_str.green()
                } else if verdict.score >= 0.5 {
                    score_str.yellow()
                } else {
                    score_str.red()
                };
                println!(
                    "  score:    {} ({})",
                    score_colored,
                    if verdict.worth_synthesizing() {
                        "worth synthesizing".green()
                    } else {
                        "skip".dimmed()
                    }
                );
                if !verdict.signals.is_empty() {
                    println!("  signals:  {}", verdict.signals.join(", "));
                }
                println!("  reason:   {}", verdict.reasoning);
                println!();
            }
        }

        Commands::Dream {
            phase,
            dry_run,
            max_usd,
            list,
            json,
        } => {
            if list {
                println!("Available phases:");
                for p in dream::all_phases() {
                    println!("  {}", p.as_str());
                }
                return Ok(());
            }
            let max_usd = max_usd.unwrap_or(config.dream.default_max_usd);
            let report = if let Some(name) = phase {
                let p: dream::PhaseName = name.parse()?;
                if dry_run {
                    dream::plan(&[p], max_usd)
                } else {
                    dream::run_one(p, &config, max_usd).await?
                }
            } else if dry_run {
                dream::plan(dream::all_phases(), max_usd)
            } else {
                dream::run_cycle(&config, max_usd).await?
            };
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                dream::print_cycle(&report, dry_run);
            }
        }

        Commands::LogCorrection {
            category,
            description,
            correction,
            episode,
            files,
            project,
        } => {
            let project = project.unwrap_or_else(|| {
                std::env::current_dir()
                    .ok()
                    .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
                    .unwrap_or_else(|| "unknown".to_string())
            });
            let files_vec: Vec<String> = files
                .map(|s| s.split(',').map(|f| f.trim().to_string()).collect())
                .unwrap_or_default();
            let category_norm = mistakes::normalize_category(&category);
            if category_norm.is_empty() {
                anyhow::bail!("--category cannot be empty after normalization");
            }
            // Resolve short episode prefix → full id if provided.
            let episode_id = if let Some(id) = episode {
                let s = store::EpisodeStore::new()?;
                s.load(&id).ok().map(|ep| ep.id).or(Some(id))
            } else {
                None
            };
            let m = mistakes::Mistake {
                id: None,
                project: project.clone(),
                category: category_norm.clone(),
                episode_id,
                files: files_vec,
                description,
                correction,
                created_at: chrono::Utc::now(),
            };
            let store = mistakes::MistakeStore::open_default().await?;
            let id = store.record(&m).await?;
            println!("Logged correction #{id} in [{project}] / category {category_norm}");
        }

        Commands::Mistakes {
            project,
            category,
            top,
            limit,
            json,
        } => {
            let store = mistakes::MistakeStore::open_default().await?;
            if let Some(n) = top {
                let cats = store.top_categories(project.as_deref(), n).await?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&cats)?);
                } else if cats.is_empty() {
                    println!("No mistakes logged yet.");
                } else {
                    use colored::Colorize;
                    println!();
                    println!(
                        "{} (top {})",
                        "Top correction categories".bold(),
                        cats.len()
                    );
                    for c in &cats {
                        let last = c.last_seen.format("%Y-%m-%d").to_string();
                        let count_str = format!("{}×", c.count);
                        let count_colored = if c.count >= 5 {
                            count_str.red()
                        } else if c.count >= 3 {
                            count_str.yellow()
                        } else {
                            count_str.normal()
                        };
                        println!("  {:<6} {:<30} (last: {})", count_colored, c.category, last);
                    }
                    println!();
                }
            } else {
                let rows = store
                    .list(project.as_deref(), category.as_deref(), limit)
                    .await?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&rows)?);
                } else if rows.is_empty() {
                    println!("No mistakes match those filters.");
                } else {
                    use colored::Colorize;
                    println!();
                    println!(
                        "{} ({} row{})",
                        "Anchored mistakes".bold(),
                        rows.len(),
                        if rows.len() == 1 { "" } else { "s" }
                    );
                    for m in &rows {
                        let when = m.created_at.format("%Y-%m-%d %H:%M").to_string();
                        let cat_colored = m.category.cyan();
                        let id8 = m
                            .episode_id
                            .as_deref()
                            .map(|s| &s[..8.min(s.len())])
                            .unwrap_or("--------");
                        println!(
                            "  [{}] {} {} {}",
                            cat_colored,
                            m.project.dimmed(),
                            id8.dimmed(),
                            when.dimmed()
                        );
                        println!("    wrong:    {}", m.description);
                        println!("    right:    {}", m.correction);
                        if !m.files.is_empty() {
                            println!("    files:    {}", m.files.join(", ").dimmed());
                        }
                    }
                    println!();
                }
            }
        }

        Commands::Brief {
            files,
            task_type,
            domain,
            project,
            json,
        } => {
            let project = project.unwrap_or_else(|| {
                std::env::current_dir()
                    .ok()
                    .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
                    .unwrap_or_else(|| "unknown".to_string())
            });
            let files_vec: Vec<String> = files
                .split(',')
                .map(|f| f.trim().to_string())
                .filter(|f| !f.is_empty())
                .collect();
            let b = brief::build_brief(
                &project,
                &files_vec,
                task_type.as_deref(),
                domain.as_deref(),
            )
            .await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&b)?);
            } else {
                print!("{}", brief::render_text(&b));
            }
        }

        Commands::SessionStart { project } => {
            let project = project.unwrap_or_else(|| {
                std::env::current_dir()
                    .ok()
                    .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
                    .unwrap_or_else(|| "unknown".to_string())
            });
            let store = ask_backs::AskBackStore::open_default().await?;
            match store.get_pending_for_project(&project).await? {
                Some(ab) => {
                    use colored::Colorize;
                    if let Some(id) = ab.id {
                        let _ = store.mark_served(id).await;
                    }
                    println!();
                    println!("{} {}:", "Pending ask-back for".bold(), project.cyan());
                    println!("  📌 {}", ab.question);
                    println!();
                    println!(
                        "  drafted after episode {ep8} (model {model}, {when})",
                        ep8 = &ab.episode_id[..8.min(ab.episode_id.len())],
                        model = ab.model.dimmed(),
                        when = ab.created_at.format("%Y-%m-%d %H:%M").to_string().dimmed(),
                    );
                    println!();
                }
                None => {
                    println!("No pending ask-back for {project}.");
                }
            }
        }

        Commands::AskBacks {
            project,
            pending,
            limit,
            json,
        } => {
            let store = ask_backs::AskBackStore::open_default().await?;
            let mut rows = match project.as_deref() {
                Some(p) => store.list_by_project(p, limit).await?,
                None => store.list_all(limit).await?,
            };
            if pending {
                rows.retain(|r| r.status == ask_backs::AskBackStatus::Pending);
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&rows)?);
            } else if rows.is_empty() {
                println!("No ask-backs found.");
            } else {
                use colored::Colorize;
                println!();
                println!(
                    "{} ({} row{})",
                    "Ask-backs".bold(),
                    rows.len(),
                    if rows.len() == 1 { "" } else { "s" }
                );
                for ab in &rows {
                    let when = ab.created_at.format("%Y-%m-%d %H:%M").to_string();
                    let status_colored = match ab.status {
                        ask_backs::AskBackStatus::Pending => "pending".yellow(),
                        ask_backs::AskBackStatus::Served => "served".green(),
                        ask_backs::AskBackStatus::Dismissed => "dismissed".dimmed(),
                    };
                    let ep8 = &ab.episode_id[..8.min(ab.episode_id.len())];
                    println!(
                        "  [{}] {} {} {}",
                        status_colored,
                        ab.project.cyan(),
                        ep8.dimmed(),
                        when.dimmed()
                    );
                    println!("    {}", ab.question);
                }
                println!();
            }
        }

        Commands::LogShouldHaveAsked {
            trigger,
            question,
            answer,
            episode,
            files,
            project,
        } => {
            let project = project.unwrap_or_else(|| {
                std::env::current_dir()
                    .ok()
                    .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
                    .unwrap_or_else(|| "unknown".to_string())
            });
            let files_vec: Vec<String> = files
                .map(|s| s.split(',').map(|f| f.trim().to_string()).collect())
                .unwrap_or_default();
            let trigger_norm = asks::normalize_trigger(&trigger);
            if trigger_norm.is_empty() {
                anyhow::bail!("--trigger cannot be empty after normalization");
            }
            let episode_id = if let Some(id) = episode {
                let s = store::EpisodeStore::new()?;
                s.load(&id).ok().map(|ep| ep.id).or(Some(id))
            } else {
                None
            };
            let sha = asks::ShouldHaveAsked {
                id: None,
                project: project.clone(),
                trigger: trigger_norm.clone(),
                question,
                answer,
                episode_id,
                files: files_vec,
                created_at: chrono::Utc::now(),
            };
            let store = asks::AsksStore::open_default().await?;
            let id = store.record(&sha).await?;
            println!("Logged should-have-asked #{id} in [{project}] / trigger {trigger_norm}");
        }

        Commands::Asks {
            project,
            trigger,
            top,
            limit,
            json,
        } => {
            let store = asks::AsksStore::open_default().await?;
            if let Some(n) = top {
                let trigs = store.top_triggers(project.as_deref(), n).await?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&trigs)?);
                } else if trigs.is_empty() {
                    println!("No should-have-asked entries yet.");
                } else {
                    use colored::Colorize;
                    println!();
                    println!(
                        "{} (top {})",
                        "Top should-have-asked triggers".bold(),
                        trigs.len()
                    );
                    for t in &trigs {
                        let last = t.last_seen.format("%Y-%m-%d").to_string();
                        let count_str = format!("{}×", t.count);
                        let count_colored = if t.count >= 5 {
                            count_str.red()
                        } else if t.count >= 3 {
                            count_str.yellow()
                        } else {
                            count_str.normal()
                        };
                        println!("  {:<6} {:<30} (last: {})", count_colored, t.trigger, last);
                    }
                    println!();
                }
            } else {
                let rows = store
                    .list(project.as_deref(), trigger.as_deref(), limit)
                    .await?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&rows)?);
                } else if rows.is_empty() {
                    println!("No should-have-asked entries match those filters.");
                } else {
                    use colored::Colorize;
                    println!();
                    println!(
                        "{} ({} row{})",
                        "Should-have-asked log".bold(),
                        rows.len(),
                        if rows.len() == 1 { "" } else { "s" }
                    );
                    for sha in &rows {
                        let when = sha.created_at.format("%Y-%m-%d %H:%M").to_string();
                        let trig_colored = sha.trigger.cyan();
                        let id8 = sha
                            .episode_id
                            .as_deref()
                            .map(|s| &s[..8.min(s.len())])
                            .unwrap_or("--------");
                        println!(
                            "  [{}] {} {} {}",
                            trig_colored,
                            sha.project.dimmed(),
                            id8.dimmed(),
                            when.dimmed()
                        );
                        println!("    Q: {}", sha.question);
                        println!("    A: {}", sha.answer);
                        if !sha.files.is_empty() {
                            println!("    files: {}", sha.files.join(", ").dimmed());
                        }
                    }
                    println!();
                }
            }
        }

        Commands::Templates { command } => {
            handle_templates_command(command).await?;
        }

        Commands::Calibration {
            task_type,
            project,
            json,
        } => {
            let cal = calibration::CalibrationStore::open_default().await?;
            let mut buckets = if let Some(p) = &project {
                cal.list_by_project(p).await?
            } else {
                cal.list_all().await?
            };
            if let Some(tt) = &task_type {
                buckets.retain(|b| b.task_type.eq_ignore_ascii_case(tt));
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&buckets)?);
            } else if buckets.is_empty() {
                println!("No calibration data yet.");
                println!(
                    "  Buckets accumulate on every capture (declared) and verification advance"
                );
                println!("  to StableNoRevert+ (verified). Run `tempera capture` and");
                println!("  `tempera advance-verification` to populate.");
            } else {
                use colored::Colorize;
                println!();
                println!(
                    "{}  ({} bucket{})",
                    "Calibration profile".bold(),
                    buckets.len(),
                    if buckets.len() == 1 { "" } else { "s" }
                );
                println!(
                    "  {:<10} {:<22} {:>5} {:>5} {:>5} {:>7}",
                    "task", "project", "decl", "ver", "fail", "verif%"
                );
                for b in &buckets {
                    let ratio = b.verified_ratio() * 100.0;
                    let ratio_str = format!("{:.0}%", ratio);
                    let ratio_colored = if b.declared_success < 5 {
                        ratio_str.dimmed()
                    } else if ratio >= 70.0 {
                        ratio_str.green()
                    } else if ratio >= 40.0 {
                        ratio_str.yellow()
                    } else {
                        ratio_str.red()
                    };
                    let proj_truncated: String = b.project.chars().take(22).collect();
                    println!(
                        "  {:<10} {:<22} {:>5} {:>5} {:>5} {:>7}",
                        b.task_type,
                        proj_truncated,
                        b.declared_success,
                        b.verified_success,
                        b.declared_failure,
                        ratio_colored
                    );
                }
                println!();
                println!(
                    "  {}",
                    "note: buckets with <5 declared are dim — too small to read.".dimmed()
                );
                println!();
            }
        }

        Commands::AdvanceVerification {
            episode,
            to,
            commit,
            run_id,
            days,
        } => {
            let new_state = parse_verification_state(&to, commit, run_id, days)?;
            let store = store::EpisodeStore::new()?;
            let mut ep = store.load(&episode)?;
            let old_label = ep.outcome.verification.label();
            ep.set_verification(new_state.clone());
            store.update(&ep)?;
            // v0.8.1: bump calibration if we reached a verified state.
            // Best-effort; advance still succeeds if the store can't open.
            if let Ok(cal) = calibration::CalibrationStore::open_default().await {
                let _ = calibration::record_verification_advance(&cal, &ep).await;
            }
            println!(
                "{} {} → {}",
                &ep.id[..8.min(ep.id.len())],
                old_label,
                new_state.label()
            );
        }

        Commands::Doctor {
            json,
            remediation_plan,
            remediate,
            yes,
            target_score,
            max_usd,
        } => {
            let report = doctor::check().await?;

            // --remediation-plan: preview only, no execution.
            if remediation_plan {
                let plan = doctor::plan_remediation(&report, max_usd);
                if json {
                    println!("{}", serde_json::to_string_pretty(&plan)?);
                } else {
                    doctor::print_human(&report);
                    doctor::print_remediation_plan(&plan);
                }
                return Ok(());
            }

            // --remediate: build plan, gate on --yes, run, re-report.
            if remediate {
                if !yes {
                    eprintln!(
                        "refusing to run remediation without --yes (this would modify your data dir)"
                    );
                    std::process::exit(2);
                }
                let plan = doctor::plan_remediation(&report, max_usd);
                if !json {
                    doctor::print_human(&report);
                    doctor::print_remediation_plan(&plan);
                }
                if plan.steps.is_empty() {
                    if json {
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&serde_json::json!({
                                "note": "no remediation needed",
                                "initial_score": report.score
                            }))?
                        );
                    }
                    return Ok(());
                }
                let outcome = doctor::execute_remediation(plan, target_score, max_usd).await?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&outcome)?);
                } else {
                    doctor::print_remediation_outcome(&outcome);
                }
                if !outcome.target_reached {
                    // Exit non-zero so cron / CI can react.
                    std::process::exit(3);
                }
                return Ok(());
            }

            // Default: read-only report (same as v0.4.6 behavior).
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                doctor::print_human(&report);
            }
            if report.score < 50 {
                std::process::exit(1);
            }
        }

        Commands::Backup(args) => {
            if args.list {
                let snapshots = backup::list_snapshots()?;
                if snapshots.is_empty() {
                    println!("(no snapshots)");
                } else {
                    for ts in &snapshots {
                        println!("{ts}");
                    }
                }
            } else if let Some(ts) = args.restore {
                backup::restore(&ts, args.force)?;
                println!("restored snapshot {ts}");
            } else {
                let path = backup::snapshot()?;
                println!("snapshot saved: {}", path.display());
            }
        }

        Commands::Job(args) => match args.command {
            JobCommand::Submit { kind, payload } => {
                let queue = jobs::JobQueue::open_default().await?;
                let parsed: serde_json::Value = serde_json::from_str(&payload)
                    .map_err(|e| anyhow::anyhow!("--payload is not valid JSON: {e}"))?;
                let id = queue.submit(&kind, parsed).await?;
                println!("submitted job {id} ({kind})");
            }
            JobCommand::List { status, limit } => {
                let queue = jobs::JobQueue::open_default().await?;
                let status = match status.as_deref() {
                    Some(s) => Some(s.parse::<jobs::JobStatus>()?),
                    None => None,
                };
                let rows = queue.list(status, limit).await?;
                if rows.is_empty() {
                    println!("(no jobs)");
                } else {
                    for job in &rows {
                        println!("{}", jobs::format_job_summary(job));
                    }
                }
            }
        },
    }

    Ok(())
}

async fn handle_templates_command(command: TemplatesCommand) -> Result<()> {
    use colored::Colorize;
    let store = templates::TemplateStore::open_default().await?;
    match command {
        TemplatesCommand::List {
            task_type,
            domain,
            json,
        } => {
            let mut all = store.list_all().await?;
            if let Some(tt) = &task_type {
                all.retain(|t| t.task_type.eq_ignore_ascii_case(tt));
            }
            if let Some(d) = &domain {
                all.retain(|t| t.domain.eq_ignore_ascii_case(d));
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&all)?);
                return Ok(());
            }
            if all.is_empty() {
                println!("No reasoning templates yet.");
                println!("  Templates accrue during the dream cycle once at least");
                println!("  3 successful episodes share a (task_type, domain) bucket.");
                println!("  Run `tempera templates extract` to author them now.");
                return Ok(());
            }
            println!();
            println!(
                "{} ({} template{})",
                "Reasoning templates".bold(),
                all.len(),
                if all.len() == 1 { "" } else { "s" }
            );
            for t in &all {
                println!(
                    "  [{tt} / {dom}] {name}",
                    tt = t.task_type.cyan(),
                    dom = t.domain.cyan(),
                    name = t.name.bold()
                );
                println!(
                    "    success_rate {sr:.2}  evidence {ev}  used {used}×",
                    sr = t.success_rate,
                    ev = t.evidence_episodes.len(),
                    used = t.times_used
                );
                for (i, step) in t.steps.iter().enumerate() {
                    println!("    {}. {}", (i + 1).to_string().dimmed(), step);
                }
            }
            println!();
        }

        TemplatesCommand::Get {
            task_type,
            domain,
            json,
        } => {
            let Some(t) = store
                .get_by_pair(&task_type.to_lowercase(), &domain)
                .await?
            else {
                println!(
                    "No template for ({task_type}, {domain}). \
                     Try `tempera templates list` to see what's stored."
                );
                return Ok(());
            };
            if json {
                println!("{}", serde_json::to_string_pretty(&t)?);
                return Ok(());
            }
            println!();
            println!(
                "{} {} / {}",
                "Template:".bold(),
                t.task_type.cyan(),
                t.domain.cyan()
            );
            println!("  {}", t.name.bold());
            println!(
                "  success_rate {sr:.2}  evidence {ev}  used {used}×",
                sr = t.success_rate,
                ev = t.evidence_episodes.len(),
                used = t.times_used
            );
            println!();
            println!("Steps:");
            for (i, step) in t.steps.iter().enumerate() {
                println!("  {}. {}", i + 1, step);
            }
            if !t.evidence_episodes.is_empty() {
                println!();
                println!("Evidence episodes:");
                for id in &t.evidence_episodes {
                    println!("  - {id}");
                }
            }
            println!();
        }

        TemplatesCommand::Extract { max_usd, dry_run } => {
            let cfg = config::Config::load().unwrap_or_default();
            let cap = max_usd.unwrap_or(cfg.dream.default_max_usd);
            if dry_run {
                println!("{} would run with budget cap ${cap:.2}", "Dry-run:".bold());
                println!("  Templates phase scans all Success+verified episodes, groups by");
                println!(
                    "  (task_type, domain), authors templates for buckets ≥{} episodes.",
                    cfg.dream.templates_min_evidence
                );
                println!("  Estimated worst case: ~5 buckets × $0.05 = $0.25");
                return Ok(());
            }
            let report = dream::run_one(dream::PhaseName::Templates, &cfg, cap).await?;
            dream::print_cycle(&report, false);
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

async fn run_propagate(
    temporal: bool,
    project: Option<String>,
    config: &config::Config,
) -> Result<()> {
    println!("📈 Running utility propagation...\n");

    // Run the main propagation pipeline
    let result = utility::run_propagation(config).await?;

    println!("\n📊 Propagation Results:");
    println!("   Episodes processed: {}", result.episodes_processed);
    println!("   Decayed: {}", result.decayed_episodes);
    println!("   Propagated: {}", result.propagated_episodes);
    println!(
        "   Hops: {}{}",
        result.hops_executed,
        if result.converged { " (converged)" } else { "" }
    );
    println!(
        "   Total utility change: {:+.3}",
        result.total_utility_change
    );

    // Run temporal credit assignment if requested
    if temporal {
        println!("\n⏱️  Running temporal credit assignment...");
        let store = store::EpisodeStore::new()?;
        let params = utility::UtilityParams::from_config(config);
        let updated =
            utility::temporal_credit_assignment(&store, project.as_deref(), &params, config)?;
        println!("   Episodes credited: {}", updated);
    }

    println!("\n✅ Propagation complete!");
    Ok(())
}

fn run_prune(
    older_than: Option<u32>,
    min_utility: Option<f32>,
    execute: bool,
    config: &config::Config,
) -> Result<()> {
    println!("🗑️  Analyzing episodes for pruning...\n");

    if !execute {
        println!("📋 DRY RUN - no episodes will be deleted");
        println!("   Use --execute to actually delete\n");
    }

    let store = store::EpisodeStore::new()?;
    let result = utility::prune_episodes(&store, older_than, min_utility, !execute, config)?;

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

/// Parse `--to <state>` plus the per-state extras into a `VerificationState`.
fn parse_verification_state(
    to: &str,
    commit: Option<String>,
    run_id: Option<String>,
    days: Option<u32>,
) -> Result<episode::VerificationState> {
    use episode::VerificationState;
    let now = chrono::Utc::now();
    let normalized = to.to_lowercase().replace('-', "_");
    match normalized.as_str() {
        "untested" => Ok(VerificationState::Untested),
        "tests_pass" => {
            let run_id =
                run_id.ok_or_else(|| anyhow::anyhow!("--run-id required for tests_pass"))?;
            Ok(VerificationState::TestsPass { run_id, at: now })
        }
        "merged" => {
            let commit = commit.ok_or_else(|| anyhow::anyhow!("--commit required for merged"))?;
            Ok(VerificationState::Merged { commit, at: now })
        }
        "stable_no_revert" => {
            let days = days.unwrap_or(1);
            Ok(VerificationState::StableNoRevert { days, since: now })
        }
        "validated_cross_project" => anyhow::bail!(
            "validated_cross_project is set automatically by the dream cycle (future v0.7), not manually"
        ),
        other => anyhow::bail!(
            "unknown verification state: '{other}' (expected: untested | tests_pass | merged | stable_no_revert)"
        ),
    }
}

/// Clone `config` with `retrieval.mode` replaced if `mode_str` is set.
fn override_mode(config: &config::Config, mode_str: Option<&str>) -> Result<config::Config> {
    let Some(s) = mode_str else {
        return Ok(config.clone());
    };
    let mode = match s.to_lowercase().as_str() {
        "vector" => config::RetrievalMode::Vector,
        "keyword" => config::RetrievalMode::Keyword,
        "hybrid" => config::RetrievalMode::Hybrid,
        other => anyhow::bail!("invalid --mode '{other}': expected vector | keyword | hybrid"),
    };
    let mut cfg = config.clone();
    cfg.retrieval.mode = mode;
    Ok(cfg)
}

fn init_project() -> Result<()> {
    use std::fs;

    let tempera_dir = dirs::home_dir()
        .expect("Could not find home directory")
        .join(".tempera");

    // Create directories
    fs::create_dir_all(tempera_dir.join("episodes"))?;
    println!("✓ Created {}", tempera_dir.display());

    // Create today's directory
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
    fs::create_dir_all(tempera_dir.join("episodes").join(&today))?;
    println!("✓ Created episodes/{}", today);

    // Initialize feedback log
    let feedback_path = tempera_dir.join("feedback.log");
    if !feedback_path.exists() {
        fs::write(&feedback_path, "")?;
        println!("✓ Initialized feedback log");
    }

    // Create config if not exists
    let config_path = tempera_dir.join("config.toml");
    if !config_path.exists() {
        let default_config = include_str!("../default_config.toml");
        fs::write(&config_path, default_config)?;
        println!("✓ Created default config");
    }

    println!("\n🎉 Tempera initialized!");
    println!("\nNext steps:");
    println!("  tempera capture --session /path/to/transcript");
    println!("  tempera retrieve \"your task description\"");

    Ok(())
}
