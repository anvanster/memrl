// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Current on-disk schema version for `Episode`. Bump when adding fields that
/// cannot be filled by serde defaults — then add a `migrate_v{n}_to_v{n+1}`
/// arm in `Episode::migrate`. See the v0.5.4 roadmap entry.
///
/// History:
///   v1 — initial stable shape from v0.4.0.
///   v2 — added `Outcome.verification` (v0.6.1).
///   v3 — added `Intent.claim` (falsifiability + category) (v0.6.2).
///   v4 — added `Episode.alternatives_considered` (v0.6.3).
///   v5 — added `Claim.validity_scope` + scope-aware decay (v0.6.4).
pub const CURRENT_EPISODE_VERSION: u32 = 5;

fn default_schema_version() -> u32 {
    // Old episodes on disk pre-date the schema_version field. They are
    // implicitly v1 — the shape that has been stable since v0.4.0.
    1
}

/// A coding episode - a single session of work
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Schema version. Added in v0.4.5. Missing on older files → defaults to 1.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub id: String,
    pub timestamp_start: DateTime<Utc>,
    pub timestamp_end: DateTime<Utc>,
    pub project: String,
    pub intent: Intent,
    pub context: Context,
    pub outcome: Outcome,
    pub utility: Utility,
    #[serde(default)]
    pub retrieval_history: Vec<RetrievalRecord>,
    /// Groups episodes in the same logical session (multi-step task)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Explicit links to related episodes
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub related_episodes: Vec<RelatedEpisode>,
    /// Approaches considered but not taken, with the reason each was ruled
    /// out. Added in episode schema v4 (v0.6.3). The road not taken is
    /// often the single highest-value field for future-me debugging — it
    /// stops the next session from re-exploring rejected paths.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alternatives_considered: Vec<Alternative>,
}

impl Episode {
    /// Walk this episode forward through the migration chain to
    /// `CURRENT_EPISODE_VERSION`. Idempotent — already-current episodes
    /// pass through unchanged. Each `v_n → v_{n+1}` step is its own match arm.
    ///
    /// Episodes from a *future* version (downgrade scenario) bail; the safer
    /// failure mode is "refuse to load" rather than "drop fields silently".
    pub fn migrate(mut self) -> Result<Self> {
        while self.schema_version < CURRENT_EPISODE_VERSION {
            self = match self.schema_version {
                1 => migrate_v1_to_v2(self)?,
                2 => migrate_v2_to_v3(self)?,
                3 => migrate_v3_to_v4(self)?,
                4 => migrate_v4_to_v5(self)?,
                v => bail!(
                    "no migration from episode schema v{v} to v{}",
                    CURRENT_EPISODE_VERSION
                ),
            };
        }
        if self.schema_version > CURRENT_EPISODE_VERSION {
            bail!(
                "episode schema_version {} is newer than this binary supports (v{}); refusing to load",
                self.schema_version,
                CURRENT_EPISODE_VERSION
            );
        }
        Ok(self)
    }

    /// Manually transition the verification state. The dream cycle and CLI
    /// helpers both go through this so the bookkeeping (timestamp on
    /// `timestamp_end`) stays consistent.
    pub fn set_verification(&mut self, new: VerificationState) {
        self.outcome.verification = new;
    }
}

/// v1 → v2: added `Outcome.verification`. serde's `default = "default_verification"`
/// already populated the field on load, so the migration is just a version bump.
fn migrate_v1_to_v2(mut ep: Episode) -> Result<Episode> {
    ep.schema_version = 2;
    Ok(ep)
}

/// v2 → v3: added `Intent.claim`. Field is `Option<Claim>`; serde default
/// puts old episodes at `None` (unscored). The LLM extract path on capture
/// fills it for new episodes. Migration is a version bump.
fn migrate_v2_to_v3(mut ep: Episode) -> Result<Episode> {
    ep.schema_version = 3;
    Ok(ep)
}

/// v3 → v4: added `Episode.alternatives_considered`. Field is `Vec<Alternative>`
/// with serde default = empty. Old episodes simply have nothing recorded;
/// new captures populate via the MCP `alternatives_considered` parameter
/// when the agent has them to share.
fn migrate_v3_to_v4(mut ep: Episode) -> Result<Episode> {
    ep.schema_version = 4;
    Ok(ep)
}

/// v4 → v5: added `Claim.validity_scope`. Field is `Option<ValidityScope>`;
/// serde default = None means "behaves like a Project-scoped capture
/// (1%/day decay)". Migration is a version bump — when the field is None,
/// the new scope-aware `decay_rate_per_day` falls back to the legacy rate.
fn migrate_v4_to_v5(mut ep: Episode) -> Result<Episode> {
    ep.schema_version = 5;
    Ok(ep)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    /// The raw first prompt from the user
    pub raw_prompt: String,
    /// LLM-extracted intent summary
    pub extracted_intent: String,
    /// Task type classification
    pub task_type: TaskType,
    /// Domain tags
    pub domain: Vec<String>,
    /// Falsifiability assessment from the LLM-extract step.
    /// Added in episode schema v3 (v0.6.2). `None` for older episodes and
    /// for captures that skipped LLM extraction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim: Option<Claim>,
}

/// Captures the *kind* of statement an episode is making and how testable it
/// is. Episodes with `falsifiability >= 0.7` enter the BKM lane —
/// they propagate utility through the similarity graph and get promoted by
/// `tempera_review`. Logistics records and pure observations stay searchable
/// but don't shape the ranking surface.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Claim {
    /// 0.0 = logistics / unfalsifiable; 1.0 = specific, testable assertion.
    /// See the prompt in `llm.rs` for the calibration scale.
    pub falsifiability: f32,
    pub category: ClaimCategory,
    /// What world is this claim valid in? Drives time-decay rate
    /// (`decay_rate_per_day`) and future invalidation triggers (Cargo.lock
    /// changes, workaround expiry). `None` = legacy behavior (1%/day decay,
    /// no invalidation triggers). Added in episode schema v5 (v0.6.4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validity_scope: Option<ValidityScope>,
}

/// Where in the world a claim holds. Different scopes decay at different
/// rates: language-level truths are durable, version-pinned workarounds
/// expire quickly. The rates come straight from `decay_rate_per_day`.
///
/// Future versions wire scope-specific invalidation: `Crate` retires on
/// version mismatch in Cargo.lock, `Workaround` retires when the
/// referenced issue closes / `expires` passes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ValidityScope {
    /// Language semantics, math, "thiserror exists" — never decays.
    Forever,
    /// True about a language as a whole, e.g. "Rust doesn't have GC".
    Language { name: String },
    /// True for a specific crate at a specific version. Invalidates on
    /// version change in the active project's Cargo.lock (v0.6.4.1+).
    /// `version` is a free-form constraint string — `=1.43.0`, `>=1.0,<2.0`,
    /// `^1.0`, or empty for "any".
    Crate { name: String, version: String },
    /// True within a problem domain (e.g. "async-rust", "embedded-systems").
    Domain { tag: String },
    /// Tied to a specific issue / bug ID. Expires automatically when
    /// `expires` passes; future versions also retire when the referenced
    /// issue closes.
    Workaround {
        ref_: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        expires: Option<DateTime<Utc>>,
    },
    /// Tied to a single project's current state. Default decay (1%/day).
    Project { name: String },
}

impl ValidityScope {
    pub fn kind_label(&self) -> &'static str {
        match self {
            Self::Forever => "forever",
            Self::Language { .. } => "language",
            Self::Crate { .. } => "crate",
            Self::Domain { .. } => "domain",
            Self::Workaround { .. } => "workaround",
            Self::Project { .. } => "project",
        }
    }
}

/// Per-day utility decay rate for an episode's scope. Calibration is from
/// the v0.6 roadmap:
///
/// | Scope       | Rate    | Half-life (rough) |
/// |-------------|---------|-------------------|
/// | Forever     | 0.000   | ∞                 |
/// | Language    | 0.001   | ~3 years          |
/// | Domain      | 0.005   | ~140 days         |
/// | Project     | 0.010   | ~70 days          |
/// | Crate       | 0.020   | ~35 days          |
/// | Workaround  | 0.050   | ~14 days          |
///
/// `None` (no scope on the capture) falls back to the legacy 0.010 rate so
/// pre-v0.6.4 episodes behave exactly as before.
pub fn decay_rate_per_day(scope: Option<&ValidityScope>) -> f64 {
    match scope {
        None => 0.010,
        Some(ValidityScope::Forever) => 0.000,
        Some(ValidityScope::Language { .. }) => 0.001,
        Some(ValidityScope::Domain { .. }) => 0.005,
        Some(ValidityScope::Project { .. }) => 0.010,
        Some(ValidityScope::Crate { .. }) => 0.020,
        Some(ValidityScope::Workaround { .. }) => 0.050,
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClaimCategory {
    /// "function X returns Y under Z" — checkable against the codebase
    ApiContract,
    /// "approach A is faster than B" — measurable
    Performance,
    /// "module M is organized as X" — verifiable by inspection
    Structural,
    /// "team prefers X over Y in this project" — convention, partially verifiable
    Conventional,
    /// "this fix works around bug #N in crate@version" — expires with the dep
    Workaround,
    /// "ran the migration", "bumped a version" — actions, not claims
    Logistics,
    Other,
}

impl ClaimCategory {
    pub fn label(self) -> &'static str {
        match self {
            Self::ApiContract => "api_contract",
            Self::Performance => "performance",
            Self::Structural => "structural",
            Self::Conventional => "conventional",
            Self::Workaround => "workaround",
            Self::Logistics => "logistics",
            Self::Other => "other",
        }
    }
}

impl std::fmt::Display for ClaimCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// An approach the author considered but did not take, with the reason it
/// was ruled out. The most expensive thing a future session does is
/// re-explore a rejected path; this field tells them not to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Alternative {
    /// The approach itself, in the author's words.
    pub approach: String,
    /// Why this approach was rejected. The *reason* is what makes the
    /// record useful — without it, future-me can't judge whether the
    /// rejection still applies.
    pub why_not: String,
    /// Proximity to actually working. `NearMiss` is the most useful — it
    /// signals "if this constraint shifts, revisit me first."
    pub how_close: HowClose,
    /// Trigger condition for revisiting. Optional; only sometimes known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub would_revisit_if: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HowClose {
    /// Would have worked except for one specific reason.
    NearMiss,
    /// Plausibly correct, traded off against the chosen approach.
    Plausible,
    /// Considered briefly, dismissed.
    LongShot,
}

impl HowClose {
    pub fn label(self) -> &'static str {
        match self {
            Self::NearMiss => "near_miss",
            Self::Plausible => "plausible",
            Self::LongShot => "long_shot",
        }
    }
}

impl std::fmt::Display for HowClose {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TaskType {
    Bugfix,
    Feature,
    Refactor,
    Test,
    Docs,
    Research,
    Debug,
    Setup,
    Unknown,
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskType::Bugfix => write!(f, "bugfix"),
            TaskType::Feature => write!(f, "feature"),
            TaskType::Refactor => write!(f, "refactor"),
            TaskType::Test => write!(f, "test"),
            TaskType::Docs => write!(f, "docs"),
            TaskType::Research => write!(f, "research"),
            TaskType::Debug => write!(f, "debug"),
            TaskType::Setup => write!(f, "setup"),
            TaskType::Unknown => write!(f, "unknown"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context {
    pub files_read: Vec<String>,
    pub files_modified: Vec<String>,
    pub tools_invoked: Vec<String>,
    pub errors_encountered: Vec<ErrorRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRecord {
    pub error_type: String,
    pub message: String,
    pub resolved: bool,
    pub resolution: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    pub status: OutcomeStatus,
    pub tests_before: Option<TestResults>,
    pub tests_after: Option<TestResults>,
    pub commit_sha: Option<String>,
    pub pr_number: Option<u32>,
    /// How well verified is the outcome claim? Drives a multiplicative
    /// adjustment to learned utility in retrieval ranking — Untested
    /// captures should not weigh as much as ones that survived 30 days
    /// post-merge. Added in episode schema v2 (v0.6.1).
    #[serde(default = "default_verification")]
    pub verification: VerificationState,
}

fn default_verification() -> VerificationState {
    VerificationState::Untested
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OutcomeStatus {
    Success,
    Partial,
    Failure,
}

impl std::fmt::Display for OutcomeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutcomeStatus::Success => write!(f, "✅ success"),
            OutcomeStatus::Partial => write!(f, "⚠️ partial"),
            OutcomeStatus::Failure => write!(f, "❌ failure"),
        }
    }
}

/// How thoroughly was the outcome verified beyond the raw success/failure
/// declaration at capture time? Each step earns more weight in retrieval
/// ranking — see `VerificationState::weight()`.
///
/// Transitions are produced by:
///   - the capture path (Untested),
///   - test-runner hooks (TestsPass),
///   - git post-merge hooks (Merged),
///   - daemon time-based promotion (StableNoRevert),
///   - the future dream cycle (ValidatedCrossProject).
///
/// In v0.6.1 only the data model + ranking integration ship; transitions
/// happen via `tempera advance-verification` (manual) or
/// `Episode::set_verification()`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum VerificationState {
    /// Capture happened. No external signal that the claim holds.
    Untested,
    /// A test run was associated with this episode and passed.
    TestsPass { run_id: String, at: DateTime<Utc> },
    /// A git merge commit references this episode (commit message tag,
    /// PR body, or manual `--commit`).
    Merged { commit: String, at: DateTime<Utc> },
    /// `Merged` with no revert observed for `days` after `since`.
    StableNoRevert { days: u32, since: DateTime<Utc> },
    /// Multiple distinct projects cite this episode as a success-template.
    /// Surfaced by future dream-cycle synthesis.
    ValidatedCrossProject { evidence_episodes: Vec<String> },
}

impl VerificationState {
    /// Multiplier applied to learned `utility` during retrieval ranking.
    /// Lower for unverified outcomes, approaches 1.0 as evidence accrues.
    pub fn weight(&self) -> f32 {
        match self {
            Self::Untested => 0.30,
            Self::TestsPass { .. } => 0.60,
            Self::Merged { .. } => 0.80,
            Self::StableNoRevert { days, .. } if *days >= 30 => 0.95,
            Self::StableNoRevert { .. } => 0.85,
            Self::ValidatedCrossProject { .. } => 1.00,
        }
    }

    /// Short label for human display (also accepted by
    /// `tempera advance-verification --to <label>`).
    pub fn label(&self) -> &'static str {
        match self {
            Self::Untested => "untested",
            Self::TestsPass { .. } => "tests_pass",
            Self::Merged { .. } => "merged",
            Self::StableNoRevert { .. } => "stable_no_revert",
            Self::ValidatedCrossProject { .. } => "validated_cross_project",
        }
    }
}

impl Default for VerificationState {
    fn default() -> Self {
        Self::Untested
    }
}

impl std::fmt::Display for VerificationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResults {
    pub passed: u32,
    pub failed: u32,
    pub skipped: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Utility {
    /// Learned utility score (0.0 - 1.0)
    pub score: Option<f32>,
    /// Number of times this episode was retrieved
    pub retrieval_count: u32,
    /// Number of times marked as helpful
    pub helpful_count: u32,
}

impl Utility {
    /// Calculate utility score using Wilson score interval (lower bound)
    /// This handles uncertainty for low-sample episodes
    pub fn calculate_score(&self) -> f32 {
        let n = self.retrieval_count as f64;
        if n == 0.0 {
            return 0.5; // Default for unretreived episodes
        }

        let p = self.helpful_count as f64 / n;
        let z = 1.96; // 95% confidence

        // Wilson score lower bound
        let score = (p + z * z / (2.0 * n) - z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).sqrt())
            / (1.0 + z * z / n);

        score as f32
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalRecord {
    pub timestamp: DateTime<Utc>,
    pub project: String,
    pub task_description: String,
    pub was_helpful: Option<bool>,
}

/// A link to a related episode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedEpisode {
    pub id: String,
    pub relationship: EpisodeRelation,
}

/// The type of relationship between two episodes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum EpisodeRelation {
    /// Same task, next step
    Continuation,
    /// This episode was needed before the other
    Prerequisite,
    /// Different approach to the same problem
    Alternative,
    /// Loosely related
    Related,
}

impl Episode {
    pub fn new(project: String, raw_prompt: String) -> Self {
        Self {
            schema_version: CURRENT_EPISODE_VERSION,
            id: uuid::Uuid::new_v4().to_string(),
            timestamp_start: Utc::now(),
            timestamp_end: Utc::now(),
            project,
            intent: Intent {
                raw_prompt,
                extracted_intent: String::new(),
                task_type: TaskType::Unknown,
                domain: vec![],
                claim: None,
            },
            context: Context {
                files_read: vec![],
                files_modified: vec![],
                tools_invoked: vec![],
                errors_encountered: vec![],
            },
            outcome: Outcome {
                status: OutcomeStatus::Partial,
                tests_before: None,
                tests_after: None,
                commit_sha: None,
                pr_number: None,
                verification: VerificationState::Untested,
            },
            utility: Utility::default(),
            retrieval_history: vec![],
            session_id: None,
            related_episodes: vec![],
            alternatives_considered: vec![],
        }
    }

    /// Convert to markdown format for human-readable storage
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str(&format!(
            "# Episode: {}\n\n",
            if self.intent.extracted_intent.is_empty() {
                &self.intent.raw_prompt
            } else {
                &self.intent.extracted_intent
            }
        ));

        md.push_str(&format!("**ID**: {}\n", &self.id[..8]));
        md.push_str(&format!(
            "**Date**: {}\n",
            self.timestamp_start.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        md.push_str(&format!("**Project**: {}\n", self.project));
        md.push_str(&format!("**Outcome**: {}\n", self.outcome.status));
        if let Some(sid) = &self.session_id {
            md.push_str(&format!("**Session**: {}\n", &sid[..8.min(sid.len())]));
        }
        md.push('\n');

        md.push_str("## Intent\n\n");
        md.push_str(&format!("{}\n\n", self.intent.raw_prompt));

        md.push_str("## Context\n\n");
        md.push_str("### Files Read\n");
        if self.context.files_read.is_empty() {
            md.push_str("- None\n");
        } else {
            for f in &self.context.files_read {
                md.push_str(&format!("- {}\n", f));
            }
        }
        md.push_str("\n");

        md.push_str("### Files Modified\n");
        if self.context.files_modified.is_empty() {
            md.push_str("- None\n");
        } else {
            for f in &self.context.files_modified {
                md.push_str(&format!("- {}\n", f));
            }
        }
        md.push_str("\n");

        md.push_str("### Commands/Tools Used\n");
        if self.context.tools_invoked.is_empty() {
            md.push_str("- None\n");
        } else {
            for t in &self.context.tools_invoked {
                md.push_str(&format!("- {}\n", t));
            }
        }
        md.push_str("\n");

        if !self.context.errors_encountered.is_empty() {
            md.push_str("## Errors → Resolutions\n\n");
            md.push_str("| Error | Resolution |\n");
            md.push_str("|-------|------------|\n");
            for e in &self.context.errors_encountered {
                let resolution = e.resolution.as_deref().unwrap_or("unresolved");
                md.push_str(&format!("| {} | {} |\n", e.message, resolution));
            }
            md.push_str("\n");
        }

        md.push_str("## Tags\n\n");
        md.push_str(&format!("{}\n\n", self.intent.domain.join(", ")));

        if !self.alternatives_considered.is_empty() {
            md.push_str("## Alternatives Considered\n\n");
            for alt in &self.alternatives_considered {
                md.push_str(&format!(
                    "- **[{}] {}** — {}",
                    alt.how_close, alt.approach, alt.why_not
                ));
                if let Some(cond) = &alt.would_revisit_if {
                    md.push_str(&format!(" *(revisit if: {})*", cond));
                }
                md.push('\n');
            }
            md.push('\n');
        }

        if !self.related_episodes.is_empty() {
            md.push_str("## Related Episodes\n\n");
            for rel in &self.related_episodes {
                let rel_type = match rel.relationship {
                    EpisodeRelation::Continuation => "continuation",
                    EpisodeRelation::Prerequisite => "prerequisite",
                    EpisodeRelation::Alternative => "alternative",
                    EpisodeRelation::Related => "related",
                };
                md.push_str(&format!(
                    "- {} ({})\n",
                    &rel.id[..8.min(rel.id.len())],
                    rel_type
                ));
            }
            md.push('\n');
        }

        if !self.retrieval_history.is_empty() {
            md.push_str("## Retrieval History\n\n");
            md.push_str("| Date | Project | Task | Helpful |\n");
            md.push_str("|------|---------|------|--------|\n");
            for r in &self.retrieval_history {
                let helpful = match r.was_helpful {
                    Some(true) => "✅",
                    Some(false) => "❌",
                    None => "?",
                };
                md.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    r.timestamp.format("%Y-%m-%d"),
                    r.project,
                    r.task_description,
                    helpful
                ));
            }
        }

        md
    }

    /// Parse from markdown (basic implementation)
    pub fn from_markdown(content: &str, _file_path: &std::path::Path) -> anyhow::Result<Self> {
        // Basic parsing - extract key fields from markdown
        // This is a simplified implementation

        let mut episode = Episode::new(
            extract_field(content, "**Project**:").unwrap_or_default(),
            extract_section(content, "## Intent").unwrap_or_default(),
        );

        if let Some(id) = extract_field(content, "**ID**:") {
            episode.id = id;
        }

        if let Some(outcome) = extract_field(content, "**Outcome**:") {
            episode.outcome.status = match outcome.to_lowercase().as_str() {
                s if s.contains("success") => OutcomeStatus::Success,
                s if s.contains("partial") => OutcomeStatus::Partial,
                s if s.contains("failure") => OutcomeStatus::Failure,
                _ => OutcomeStatus::Partial,
            };
        }

        if let Some(tags) = extract_section(content, "## Tags") {
            episode.intent.domain = tags
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        Ok(episode)
    }
}

fn extract_field(content: &str, field: &str) -> Option<String> {
    for line in content.lines() {
        if line.starts_with(field) {
            return Some(line.trim_start_matches(field).trim().to_string());
        }
    }
    None
}

fn extract_section(content: &str, header: &str) -> Option<String> {
    let mut in_section = false;
    let mut section_content = String::new();

    for line in content.lines() {
        if line.starts_with(header) {
            in_section = true;
            continue;
        }
        if in_section {
            if line.starts_with("## ") {
                break;
            }
            section_content.push_str(line);
            section_content.push('\n');
        }
    }

    if section_content.is_empty() {
        None
    } else {
        Some(section_content.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utility_score_calculation() {
        // No retrievals = 0.5
        let utility = Utility::default();
        assert!((utility.calculate_score() - 0.5).abs() < 0.01);

        // 10 retrievals, 10 helpful = high score
        let utility = Utility {
            score: None,
            retrieval_count: 10,
            helpful_count: 10,
        };
        assert!(utility.calculate_score() > 0.7);

        // 10 retrievals, 0 helpful = low score
        let utility = Utility {
            score: None,
            retrieval_count: 10,
            helpful_count: 0,
        };
        assert!(utility.calculate_score() < 0.3);
    }

    #[test]
    fn test_backward_compat_deserialization() {
        // Old JSON without session_id and related_episodes should deserialize fine
        let json = r#"{
            "id": "test-1234-5678-abcd",
            "timestamp_start": "2026-03-01T10:00:00Z",
            "timestamp_end": "2026-03-01T11:00:00Z",
            "project": "test-proj",
            "intent": {
                "raw_prompt": "fix the bug",
                "extracted_intent": "fix auth bug",
                "task_type": "bugfix",
                "domain": ["rust", "auth"]
            },
            "context": {
                "files_read": [],
                "files_modified": ["src/auth.rs"],
                "tools_invoked": [],
                "errors_encountered": []
            },
            "outcome": {
                "status": "success",
                "tests_before": null,
                "tests_after": null,
                "commit_sha": null,
                "pr_number": null
            },
            "utility": {
                "score": null,
                "retrieval_count": 3,
                "helpful_count": 2
            }
        }"#;
        let ep: Episode = serde_json::from_str(json).unwrap();
        assert_eq!(ep.id, "test-1234-5678-abcd");
        assert!(ep.session_id.is_none());
        assert!(ep.related_episodes.is_empty());
        assert!(ep.retrieval_history.is_empty());
        // Older files lack `schema_version`; serde default puts them at v1.
        assert_eq!(ep.schema_version, 1);
        // serde default also populates verification → Untested.
        assert_eq!(ep.outcome.verification, VerificationState::Untested);
        // migrate() walks v1 forward to the current version without losing fields.
        let migrated = ep.migrate().unwrap();
        assert_eq!(migrated.schema_version, CURRENT_EPISODE_VERSION);
        assert_eq!(migrated.outcome.verification, VerificationState::Untested);
        // v3 added Intent.claim with default None — no LLM extract was applied.
        assert!(migrated.intent.claim.is_none());
    }

    #[test]
    fn migrate_passthroughs_current_version() {
        let ep = Episode::new("p".to_string(), "test".to_string());
        assert_eq!(ep.schema_version, CURRENT_EPISODE_VERSION);
        let migrated = ep.clone().migrate().unwrap();
        assert_eq!(migrated.schema_version, CURRENT_EPISODE_VERSION);
    }

    #[test]
    fn migrate_bails_on_future_version() {
        let mut ep = Episode::new("p".to_string(), "test".to_string());
        ep.schema_version = CURRENT_EPISODE_VERSION + 1;
        let err = ep.migrate().unwrap_err();
        assert!(
            err.to_string().contains("newer than this binary supports"),
            "got: {err}"
        );
    }

    #[test]
    fn new_episode_has_current_schema_version() {
        let ep = Episode::new("p".to_string(), "test".to_string());
        assert_eq!(ep.schema_version, CURRENT_EPISODE_VERSION);
    }

    #[test]
    fn schema_version_roundtrips_through_serde() {
        let mut ep = Episode::new("p".to_string(), "test".to_string());
        ep.schema_version = 1;
        let json = serde_json::to_string(&ep).unwrap();
        assert!(
            json.contains("\"schema_version\":1"),
            "schema_version not in output: {json}"
        );
        let parsed: Episode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.schema_version, 1);
    }

    #[test]
    fn verification_weights_match_roadmap() {
        // Calibration: spread between 0.30 and 1.00 matches the v0.6.1
        // table. If any weight changes, downstream eval results will move.
        assert!((VerificationState::Untested.weight() - 0.30).abs() < 1e-6);
        let tp = VerificationState::TestsPass {
            run_id: "x".into(),
            at: Utc::now(),
        };
        assert!((tp.weight() - 0.60).abs() < 1e-6);
        let m = VerificationState::Merged {
            commit: "abc".into(),
            at: Utc::now(),
        };
        assert!((m.weight() - 0.80).abs() < 1e-6);
        let snr_fresh = VerificationState::StableNoRevert {
            days: 7,
            since: Utc::now(),
        };
        assert!((snr_fresh.weight() - 0.85).abs() < 1e-6);
        let snr_mature = VerificationState::StableNoRevert {
            days: 30,
            since: Utc::now(),
        };
        assert!((snr_mature.weight() - 0.95).abs() < 1e-6);
        let vcp = VerificationState::ValidatedCrossProject {
            evidence_episodes: vec!["a".into()],
        };
        assert!((vcp.weight() - 1.00).abs() < 1e-6);
    }

    #[test]
    fn verification_state_serde_tag() {
        let tp = VerificationState::TestsPass {
            run_id: "r1".into(),
            at: Utc::now(),
        };
        let json = serde_json::to_string(&tp).unwrap();
        // Tag-based serialization → `"state": "tests_pass"`
        assert!(
            json.contains("\"state\":\"tests_pass\""),
            "expected tests_pass tag, got: {json}"
        );
        let parsed: VerificationState = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.label(), "tests_pass");
    }

    #[test]
    fn new_episode_starts_untested() {
        let ep = Episode::new("p".to_string(), "test".to_string());
        assert_eq!(ep.outcome.verification, VerificationState::Untested);
    }

    #[test]
    fn set_verification_updates_outcome() {
        let mut ep = Episode::new("p".to_string(), "test".to_string());
        ep.set_verification(VerificationState::Merged {
            commit: "deadbeef".to_string(),
            at: Utc::now(),
        });
        assert_eq!(ep.outcome.verification.label(), "merged");
    }

    #[test]
    fn migrate_v1_episode_advances_to_v2() {
        let mut ep = Episode::new("p".to_string(), "test".to_string());
        ep.schema_version = 1;
        let migrated = ep.migrate().unwrap();
        assert_eq!(migrated.schema_version, CURRENT_EPISODE_VERSION);
    }

    #[test]
    fn migrate_v2_episode_advances_to_current_keeping_verification() {
        // An episode at v2 should reach CURRENT_EPISODE_VERSION without
        // losing the v2 verification field.
        let mut ep = Episode::new("p".to_string(), "test".to_string());
        ep.schema_version = 2;
        ep.outcome.verification = VerificationState::Merged {
            commit: "abc".to_string(),
            at: Utc::now(),
        };
        let migrated = ep.migrate().unwrap();
        assert_eq!(migrated.schema_version, CURRENT_EPISODE_VERSION);
        assert!(matches!(
            migrated.outcome.verification,
            VerificationState::Merged { .. }
        ));
        // The new field defaults to None — no LLM-extracted claim yet.
        assert!(migrated.intent.claim.is_none());
    }

    #[test]
    fn claim_roundtrips_through_serde() {
        let mut ep = Episode::new("p".to_string(), "test".to_string());
        ep.intent.claim = Some(Claim {
            falsifiability: 0.85,
            category: ClaimCategory::ApiContract,
            validity_scope: None,
        });
        let json = serde_json::to_string(&ep).unwrap();
        // skip_serializing_if drops None claims; this one should serialize.
        assert!(json.contains("\"claim\""), "claim missing in JSON: {json}");
        assert!(json.contains("\"falsifiability\":0.85"));
        assert!(json.contains("\"category\":\"api_contract\""));
        let parsed: Episode = serde_json::from_str(&json).unwrap();
        let parsed_claim = parsed.intent.claim.expect("claim present");
        assert!((parsed_claim.falsifiability - 0.85).abs() < 1e-6);
        assert_eq!(parsed_claim.category, ClaimCategory::ApiContract);
    }

    #[test]
    fn claim_absent_when_none() {
        // Confirm skip_serializing_if removes the field for backwards compat.
        let ep = Episode::new("p".to_string(), "test".to_string());
        assert!(ep.intent.claim.is_none());
        let json = serde_json::to_string(&ep).unwrap();
        assert!(
            !json.contains("\"claim\""),
            "claim should be absent: {json}"
        );
    }

    #[test]
    fn decay_rate_calibration_matches_roadmap_table() {
        // These numbers are documented externally; a change here will move
        // every user's utility curve. Lock them down.
        assert_eq!(decay_rate_per_day(None), 0.010);
        assert_eq!(decay_rate_per_day(Some(&ValidityScope::Forever)), 0.000);
        assert_eq!(
            decay_rate_per_day(Some(&ValidityScope::Language {
                name: "rust".into()
            })),
            0.001
        );
        assert_eq!(
            decay_rate_per_day(Some(&ValidityScope::Domain {
                tag: "async".into()
            })),
            0.005
        );
        assert_eq!(
            decay_rate_per_day(Some(&ValidityScope::Project {
                name: "tempera".into()
            })),
            0.010
        );
        assert_eq!(
            decay_rate_per_day(Some(&ValidityScope::Crate {
                name: "tokio".into(),
                version: "=1.43.0".into(),
            })),
            0.020
        );
        assert_eq!(
            decay_rate_per_day(Some(&ValidityScope::Workaround {
                ref_: "issue/1234".into(),
                expires: None,
            })),
            0.050
        );
    }

    #[test]
    fn forever_scope_yields_no_decay_after_a_year() {
        // (1 - 0.000)^365 == 1.0 — Forever truths keep their utility.
        let r = decay_rate_per_day(Some(&ValidityScope::Forever));
        let factor = (1.0 - r).powf(365.0);
        assert!((factor - 1.0).abs() < 1e-9);
    }

    #[test]
    fn workaround_decays_five_times_faster_than_project() {
        // 100 days for both scopes:
        //   project:    (1 - 0.010)^100 ≈ 0.366
        //   workaround: (1 - 0.050)^100 ≈ 0.0059
        let p = (1.0 - decay_rate_per_day(Some(&ValidityScope::Project { name: "t".into() })))
            .powf(100.0);
        let w = (1.0
            - decay_rate_per_day(Some(&ValidityScope::Workaround {
                ref_: "x".into(),
                expires: None,
            })))
        .powf(100.0);
        assert!(p > 0.3, "project @ 100d should retain >0.3 (got {p})");
        assert!(
            w < 0.01,
            "workaround @ 100d should fall below 0.01 (got {w})"
        );
    }

    #[test]
    fn validity_scope_serde_tagged_per_variant() {
        for s in &[
            ValidityScope::Forever,
            ValidityScope::Language {
                name: "rust".into(),
            },
            ValidityScope::Crate {
                name: "tokio".into(),
                version: "=1.43.0".into(),
            },
            ValidityScope::Domain {
                tag: "async-rust".into(),
            },
            ValidityScope::Workaround {
                ref_: "rust-lang/rust#1234".into(),
                expires: None,
            },
            ValidityScope::Project {
                name: "tempera".into(),
            },
        ] {
            let json = serde_json::to_string(s).unwrap();
            assert!(
                json.contains(&format!("\"kind\":\"{}\"", s.kind_label())),
                "kind tag missing for {:?}: {json}",
                s
            );
            let parsed: ValidityScope = serde_json::from_str(&json).unwrap();
            assert_eq!(&parsed, s);
        }
    }

    #[test]
    fn validity_scope_absent_when_none() {
        let ep = Episode::new("p".into(), "test".into());
        // Add a claim without scope.
        let mut ep_with_claim = ep.clone();
        ep_with_claim.intent.claim = Some(Claim {
            falsifiability: 0.8,
            category: ClaimCategory::Performance,
            validity_scope: None,
        });
        let json = serde_json::to_string(&ep_with_claim).unwrap();
        assert!(
            !json.contains("\"validity_scope\""),
            "None scope should not serialize: {json}"
        );
    }

    #[test]
    fn migrate_v4_episode_advances_to_current() {
        let mut ep = Episode::new("p".into(), "test".into());
        ep.schema_version = 4;
        let migrated = ep.migrate().unwrap();
        assert_eq!(migrated.schema_version, CURRENT_EPISODE_VERSION);
        // claim is still None (no LLM ran in test)
        assert!(migrated.intent.claim.is_none());
    }

    #[test]
    fn alternatives_default_empty_and_skip_when_empty() {
        let ep = Episode::new("p".into(), "test".into());
        assert!(ep.alternatives_considered.is_empty());
        let json = serde_json::to_string(&ep).unwrap();
        assert!(
            !json.contains("alternatives_considered"),
            "empty Vec should be skipped: {json}"
        );
    }

    #[test]
    fn alternative_serde_roundtrip() {
        let mut ep = Episode::new("p".into(), "test".into());
        ep.alternatives_considered.push(Alternative {
            approach: "use Arc<Mutex<HashMap>>".to_string(),
            why_not: "lock contention on hot path".to_string(),
            how_close: HowClose::NearMiss,
            would_revisit_if: Some("single-writer becomes the case".to_string()),
        });
        let json = serde_json::to_string(&ep).unwrap();
        assert!(json.contains("\"how_close\":\"near_miss\""));
        assert!(json.contains("\"alternatives_considered\""));
        let parsed: Episode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.alternatives_considered.len(), 1);
        let alt = &parsed.alternatives_considered[0];
        assert_eq!(alt.how_close, HowClose::NearMiss);
        assert_eq!(
            alt.would_revisit_if.as_deref(),
            Some("single-writer becomes the case")
        );
    }

    #[test]
    fn how_close_serde_snake_case() {
        for hc in &[HowClose::NearMiss, HowClose::Plausible, HowClose::LongShot] {
            let json = serde_json::to_string(hc).unwrap();
            let parsed: HowClose = serde_json::from_str(&json).unwrap();
            assert_eq!(*hc, parsed);
        }
    }

    #[test]
    fn migrate_v3_episode_advances_to_current_keeps_claim() {
        let mut ep = Episode::new("p".into(), "test".into());
        ep.schema_version = 3;
        ep.intent.claim = Some(Claim {
            falsifiability: 0.75,
            category: ClaimCategory::Performance,
            validity_scope: None,
        });
        let migrated = ep.migrate().unwrap();
        assert_eq!(migrated.schema_version, CURRENT_EPISODE_VERSION);
        let claim = migrated.intent.claim.expect("claim retained");
        assert!((claim.falsifiability - 0.75).abs() < 1e-6);
        assert_eq!(claim.category, ClaimCategory::Performance);
        assert!(migrated.alternatives_considered.is_empty());
    }

    #[test]
    fn claim_category_serde_snake_case() {
        for cat in &[
            ClaimCategory::ApiContract,
            ClaimCategory::Performance,
            ClaimCategory::Structural,
            ClaimCategory::Conventional,
            ClaimCategory::Workaround,
            ClaimCategory::Logistics,
            ClaimCategory::Other,
        ] {
            let json = serde_json::to_string(cat).unwrap();
            let parsed: ClaimCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, *cat, "roundtrip failed for {}", cat.label());
        }
    }

    #[test]
    fn test_session_fields_serialization() {
        let mut ep = Episode::new("test".to_string(), "prompt".to_string());
        ep.session_id = Some("session-abc".to_string());
        ep.related_episodes.push(RelatedEpisode {
            id: "related-123".to_string(),
            relationship: EpisodeRelation::Continuation,
        });

        let json = serde_json::to_string(&ep).unwrap();
        let parsed: Episode = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.session_id, Some("session-abc".to_string()));
        assert_eq!(parsed.related_episodes.len(), 1);
        assert_eq!(
            parsed.related_episodes[0].relationship,
            EpisodeRelation::Continuation
        );
    }

    #[test]
    fn test_session_in_markdown() {
        let mut ep = Episode::new("test".to_string(), "prompt".to_string());
        ep.session_id = Some("abcdef12-3456-7890".to_string());
        ep.related_episodes.push(RelatedEpisode {
            id: "related-123456789".to_string(),
            relationship: EpisodeRelation::Prerequisite,
        });

        let md = ep.to_markdown();
        assert!(md.contains("**Session**: abcdef12"));
        assert!(md.contains("related- (prerequisite)"));
    }
}
