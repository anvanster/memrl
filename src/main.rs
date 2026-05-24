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

mod backup;
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
mod patterns;
mod reflect;
mod retrieve;
mod stats;
mod store;
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

    /// Read-only health check (index, coverage, links, eval, queue)
    Doctor {
        /// Emit machine-readable JSON instead of the colored summary
        #[arg(long)]
        json: bool,
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
            println!(
                "{} {} → {}",
                &ep.id[..8.min(ep.id.len())],
                old_label,
                new_state.label()
            );
        }

        Commands::Doctor { json } => {
            let report = doctor::check().await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                doctor::print_human(&report);
            }
            // Non-zero exit when health is poor, so CI can gate on it later.
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
