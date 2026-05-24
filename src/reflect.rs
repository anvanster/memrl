// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Reflection authorship phase.
//!
//! For each day that triage scored `>= MIN_SYNTHESIZE_SCORE`, run a
//! Sonnet pass that writes a short reflection page — what pattern emerges
//! across the day's captures, what surprised the author, what this day
//! teaches about the codebase. The point is to produce text that the
//! captures *don't already say* — a synthesis, not a summary.
//!
//! Quality bar:
//!   - Quote at least one capture verbatim (regex `` > `id:` "..." ``).
//!   - Cite at least one episode ID.
//!   - 80–1500 words.
//!   - No slop phrases ("we should consider", "perhaps", "in conclusion").
//!
//! On gate failure we attempt one regen with a stricter prompt; if that
//! also fails we fall back to a hand-written template stub. The Sonnet
//! call itself is the most expensive piece in the dream cycle
//! (~$0.05/reflection); the template fallback costs $0.
//!
//! Storage is dual: a .md sidecar at `~/.tempera/reflections/<id>.md` for
//! human readers, plus a SQLite row for the dream cycle to query
//! ("already reflected on this date?").

#![allow(dead_code)]

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool};
use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::dream::CostBudget;
use crate::episode::Episode;
use crate::llm::AnthropicClient;

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// USD budgeted per reflect call (Sonnet 4.6, ~1500 in + 1500 out tokens).
pub const REFLECT_ESTIMATED_COST_USD: f32 = 0.05;

/// Phrases that signal AI slop. If the body contains any of these the
/// gate fails and we either regenerate or fall back to template.
const SLOP_PHRASES: &[&str] = &[
    "we should consider",
    "in conclusion",
    "it is worth noting",
    "moving forward",
    "as we have seen",
    "going forward",
    "it should be noted",
];

/// Word-count bounds for accepted reflections. Below the floor reads
/// like a tweet; above the ceiling is rambling.
const WORD_FLOOR: usize = 80;
const WORD_CEILING: usize = 1500;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reflection {
    pub id: String,
    pub date: NaiveDate,
    pub project: Option<String>,
    pub body: String,
    pub citations: Vec<String>,
    pub signals: Vec<String>,
    pub triage_score: f32,
    pub model: String,
    pub created_at: DateTime<Utc>,
}

impl Reflection {
    pub fn id_for(date: &NaiveDate, project: Option<&str>) -> String {
        match project {
            Some(p) => format!("{date}-{p}"),
            None => format!("{date}-all"),
        }
    }
}

// ===== Store =====

#[derive(Clone)]
pub struct ReflectionStore {
    pool: SqlitePool,
}

impl ReflectionStore {
    pub async fn open_default() -> Result<Self> {
        let path = Config::data_dir()?.join("jobs.sqlite");
        Self::open(&path).await
    }

    pub async fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let opts = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true);
        let pool = SqlitePool::connect_with(opts)
            .await
            .with_context(|| format!("Failed to open reflections DB at {}", path.display()))?;
        MIGRATOR
            .run(&pool)
            .await
            .context("Failed to run reflection migrations")?;
        Ok(Self { pool })
    }

    pub async fn open_in_memory() -> Result<Self> {
        let pool = SqlitePool::connect("sqlite::memory:").await?;
        MIGRATOR.run(&pool).await?;
        Ok(Self { pool })
    }

    pub async fn get(&self, id: &str) -> Result<Option<Reflection>> {
        let row = sqlx::query(
            "SELECT id, date, project, body, citations, signals, triage_score, model, created_at
             FROM reflections WHERE id = ?1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        row.as_ref().map(Self::row_to_reflection).transpose()
    }

    pub async fn put(&self, r: &Reflection) -> Result<()> {
        sqlx::query(
            "INSERT INTO reflections
                (id, date, project, body, citations, signals, triage_score, model, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(id) DO UPDATE SET
                body = excluded.body,
                citations = excluded.citations,
                signals = excluded.signals,
                triage_score = excluded.triage_score,
                model = excluded.model,
                created_at = excluded.created_at",
        )
        .bind(&r.id)
        .bind(r.date.to_string())
        .bind(r.project.as_deref())
        .bind(&r.body)
        .bind(serde_json::to_string(&r.citations)?)
        .bind(serde_json::to_string(&r.signals)?)
        .bind(r.triage_score)
        .bind(&r.model)
        .bind(r.created_at.timestamp())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_by_date(&self, date: &NaiveDate) -> Result<Vec<Reflection>> {
        let rows = sqlx::query(
            "SELECT id, date, project, body, citations, signals, triage_score, model, created_at
             FROM reflections WHERE date = ?1 ORDER BY id ASC",
        )
        .bind(date.to_string())
        .fetch_all(&self.pool)
        .await?;
        rows.iter().map(Self::row_to_reflection).collect()
    }

    fn row_to_reflection(row: &sqlx::sqlite::SqliteRow) -> Result<Reflection> {
        let citations_json: String = row.get("citations");
        let signals_json: String = row.get("signals");
        let date_str: String = row.get("date");
        let created_ts: i64 = row.get("created_at");
        Ok(Reflection {
            id: row.get("id"),
            date: NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")?,
            project: row.get("project"),
            body: row.get("body"),
            citations: serde_json::from_str(&citations_json)?,
            signals: serde_json::from_str(&signals_json)?,
            triage_score: row.get("triage_score"),
            model: row.get("model"),
            created_at: DateTime::<Utc>::from_timestamp(created_ts, 0).unwrap_or_default(),
        })
    }
}

// ===== Prompts =====

const REFLECT_SYSTEM: &str = r#"You are reading a developer's coding-session captures from a single day. Your job is to write a short reflection (80–400 words) that says something the captures themselves don't say — a pattern, a surprise, what this day teaches about the codebase.

Rules:
- Quote captures verbatim when citing them. Don't paraphrase memorable phrasings. Format:
  > `<episode-id>:` "exact quote from intent or extracted_intent"
- Every substantive claim must reference at least one capture by its 8-char episode ID.
- Be concrete. Banned phrases: "we should consider", "perhaps", "in conclusion", "moving forward", "it is worth noting".
- If the day is truly logistics-only, write exactly: "no reflection — logistics day."
- Output the body as plain prose. Do not include frontmatter or a header — the system wraps your output.
"#;

const REFLECT_SYSTEM_STRICT: &str = r#"You are re-attempting a reflection that failed the quality gate. Stricter rules:
- Open with a one-sentence thesis that doesn't sound like an essay introduction.
- Every paragraph must quote a capture verbatim using format `> \`<id>:\` "quote"`.
- Cite at least 2 distinct episode IDs.
- No meta-commentary ("this day showed", "we can see that"). Just observations.
- 100–350 words.
"#;

/// Build the user message for one reflection — episode bodies, intent
/// summaries, and the triage signals that made this day worth processing.
pub fn build_reflect_user_message(
    date: &NaiveDate,
    episodes: &[Episode],
    triage_signals: &[String],
) -> String {
    let mut s = format!(
        "Date: {date}\nTriage signals: {}\nEpisodes: {} captured\n\n",
        triage_signals.join(", "),
        episodes.len()
    );
    for ep in episodes {
        let short = &ep.id[..8.min(ep.id.len())];
        s.push_str(&format!("--- {} ({}) ---\n", short, ep.project));
        s.push_str(&format!("task: {}\n", ep.intent.task_type));
        s.push_str(&format!("outcome: {}\n", ep.outcome.status));
        let intent = if !ep.intent.extracted_intent.is_empty() {
            &ep.intent.extracted_intent
        } else {
            &ep.intent.raw_prompt
        };
        s.push_str(&format!("intent: {intent}\n"));
        if let Some(claim) = &ep.intent.claim {
            s.push_str(&format!(
                "claim: falsifiability={:.2} category={}\n",
                claim.falsifiability, claim.category
            ));
        }
        if !ep.alternatives_considered.is_empty() {
            s.push_str("alternatives_considered:\n");
            for alt in &ep.alternatives_considered {
                s.push_str(&format!(
                    "  - [{}] {} — {}\n",
                    alt.how_close, alt.approach, alt.why_not
                ));
            }
        }
        if !ep.context.files_modified.is_empty() {
            s.push_str(&format!(
                "files_modified: {}\n",
                ep.context.files_modified.join(", ")
            ));
        }
        s.push('\n');
    }
    s
}

// ===== Quality gate =====

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateOutcome {
    Pass,
    Fail(GateFailure),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateFailure {
    NoQuote,
    NoCitation,
    TooShort,
    TooLong,
    SlopPhrase(String),
}

pub fn quality_gate(body: &str) -> GateOutcome {
    // "no reflection — logistics day." is an explicit short-circuit.
    if body
        .trim()
        .eq_ignore_ascii_case("no reflection — logistics day.")
        || body
            .trim()
            .eq_ignore_ascii_case("no reflection - logistics day.")
    {
        return GateOutcome::Pass;
    }

    let word_count = body.split_whitespace().count();
    if word_count < WORD_FLOOR {
        return GateOutcome::Fail(GateFailure::TooShort);
    }
    if word_count > WORD_CEILING {
        return GateOutcome::Fail(GateFailure::TooLong);
    }

    let lower = body.to_lowercase();
    if let Some(phrase) = SLOP_PHRASES.iter().find(|p| lower.contains(*p)) {
        return GateOutcome::Fail(GateFailure::SlopPhrase((*phrase).to_string()));
    }

    if !has_verbatim_quote(body) {
        return GateOutcome::Fail(GateFailure::NoQuote);
    }
    if extract_citations(body).is_empty() {
        return GateOutcome::Fail(GateFailure::NoCitation);
    }
    GateOutcome::Pass
}

fn has_verbatim_quote(body: &str) -> bool {
    // Accept either backtick-wrapped citations from our prompt or
    // straight quote characters as a courtesy.
    body.lines()
        .any(|l| l.trim_start().starts_with('>') && (l.contains('`') || l.contains('"')))
}

/// Pull 8-char hex-ish IDs out of the body. We're looking for "abc12345"
/// style episode-id prefixes. Conservatively requires alphanumeric and a
/// minimum length so we don't false-match arbitrary 4-letter words.
pub fn extract_citations(body: &str) -> Vec<String> {
    let mut out = std::collections::BTreeSet::new();
    for tok in body.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if tok.len() == 8 && tok.chars().all(|c| c.is_ascii_hexdigit()) {
            out.insert(tok.to_lowercase());
        }
    }
    out.into_iter().collect()
}

// ===== Authorship =====

/// Author a reflection for the given day. Calls Sonnet, runs the
/// quality gate, regenerates once on failure with a stricter prompt,
/// and falls back to a template if both attempts fail. Stores the
/// reflection in SQLite + a .md sidecar.
pub async fn author_reflection(
    date: &NaiveDate,
    episodes: &[Episode],
    triage_signals: &[String],
    triage_score: f32,
    config: &Config,
    budget: Option<&CostBudget>,
    store: &ReflectionStore,
) -> Result<Reflection> {
    if let Some(b) = budget {
        b.try_spend(REFLECT_ESTIMATED_COST_USD)?;
    }

    let user_msg = build_reflect_user_message(date, episodes, triage_signals);
    let client = AnthropicClient::with_model(&config.dream.reflect_model)?;

    let first = client
        .raw_completion(REFLECT_SYSTEM, &user_msg, config.dream.reflect_max_tokens)
        .await
        .context("reflect: initial Sonnet call failed")?;
    let body = match quality_gate(&first) {
        GateOutcome::Pass => first,
        GateOutcome::Fail(reason) => {
            eprintln!("reflect: gate failed ({reason:?}), regenerating");
            if let Some(b) = budget {
                b.try_spend(REFLECT_ESTIMATED_COST_USD)?;
            }
            let second = client
                .raw_completion(
                    REFLECT_SYSTEM_STRICT,
                    &user_msg,
                    config.dream.reflect_max_tokens,
                )
                .await
                .context("reflect: regen call failed")?;
            match quality_gate(&second) {
                GateOutcome::Pass => second,
                GateOutcome::Fail(reason2) => {
                    eprintln!("reflect: regen also failed ({reason2:?}), using template");
                    template_fallback(date, episodes)
                }
            }
        }
    };

    let citations = extract_citations(&body);
    let reflection = Reflection {
        id: Reflection::id_for(date, None),
        date: *date,
        project: None,
        body,
        citations,
        signals: triage_signals.to_vec(),
        triage_score,
        model: config.dream.reflect_model.clone(),
        created_at: Utc::now(),
    };

    store.put(&reflection).await?;
    write_sidecar(&reflection)?;
    Ok(reflection)
}

/// Hand-written fallback when LLM output fails the gate twice. Captures
/// who was active and on what projects without trying to synthesize.
pub fn template_fallback(date: &NaiveDate, episodes: &[Episode]) -> String {
    use std::collections::BTreeSet;
    let mut projects: BTreeSet<&str> = BTreeSet::new();
    for ep in episodes {
        projects.insert(&ep.project);
    }
    format!(
        "Reflection generation failed the quality gate for {date}. \
         {} episodes captured across project(s): {}. \
         Re-run `tempera reflect --date {} --force` to retry.",
        episodes.len(),
        projects.into_iter().collect::<Vec<_>>().join(", "),
        date
    )
}

// ===== Sidecar writer =====

fn reflections_dir() -> Result<PathBuf> {
    Ok(Config::data_dir()?.join("reflections"))
}

fn write_sidecar(r: &Reflection) -> Result<()> {
    let dir = reflections_dir()?;
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.md", r.id));
    let frontmatter = format!(
        "+++\nid = \"{id}\"\ndate = \"{date}\"\nproject = {project}\ncitations = {cites}\nsignals = {signals}\ntriage_score = {score}\nmodel = \"{model}\"\ncreated_at = \"{ts}\"\n+++\n\n",
        id = r.id,
        date = r.date,
        project = r
            .project
            .as_deref()
            .map(|p| format!("\"{p}\""))
            .unwrap_or_else(|| "null".to_string()),
        cites = toml_array(&r.citations),
        signals = toml_array(&r.signals),
        score = r.triage_score,
        model = r.model,
        ts = r.created_at.to_rfc3339(),
    );
    let header = format!(
        "# Reflection: {} ({} citation(s))\n\n",
        r.date,
        r.citations.len()
    );
    let content = format!("{frontmatter}{header}{}\n", r.body);
    std::fs::write(&path, content)
        .with_context(|| format!("Failed to write reflection sidecar at {}", path.display()))?;
    Ok(())
}

fn toml_array(items: &[String]) -> String {
    let quoted: Vec<String> = items
        .iter()
        .map(|s| format!("\"{}\"", s.replace('"', "\\\"")))
        .collect();
    format!("[{}]", quoted.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_body_with_citation(text: &str) -> String {
        format!(
            "{text}\n\n> `abc12345:` \"a verbatim quote from the capture\"\n\nThis observation about abc12345 holds.\n"
        )
    }

    #[test]
    fn extract_citations_finds_hex_ids() {
        let body = "References abc12345 and def67890 in passing. cafebabe also.";
        let ids = extract_citations(body);
        assert!(ids.contains(&"abc12345".to_string()));
        assert!(ids.contains(&"def67890".to_string()));
        assert!(ids.contains(&"cafebabe".to_string()));
    }

    #[test]
    fn extract_citations_ignores_short_or_non_hex_tokens() {
        let body = "this here is a sentence with abc 123abc12 hello-world";
        // "abc" too short, "123abc12" not 8 hex chars? actually "23abc12" — no, "123abc12" is 8 hex digits.
        let ids = extract_citations(body);
        // "123abc12" IS valid hex of length 8, so it gets picked up.
        // "abc" is not. "hello-world" is not.
        assert!(ids.contains(&"123abc12".to_string()));
        assert!(!ids.iter().any(|s| s == "abc"));
    }

    #[test]
    fn quality_gate_passes_logistics_short_circuit() {
        let g = quality_gate("no reflection — logistics day.");
        assert_eq!(g, GateOutcome::Pass);
    }

    #[test]
    fn quality_gate_fails_on_no_quote() {
        let body: String = std::iter::repeat_n("word ", 100).collect();
        let result = quality_gate(&format!("{body} abc12345"));
        assert!(matches!(result, GateOutcome::Fail(GateFailure::NoQuote)));
    }

    #[test]
    fn quality_gate_fails_on_no_citation() {
        let body_text: String = std::iter::repeat_n("word ", 100).collect();
        let body = format!("{body_text}\n> `nothere:` \"quote\"\n");
        // "nothere" is 7 chars; not extracted as a citation.
        let result = quality_gate(&body);
        // Actually quote check first passes, citation check fails.
        assert!(matches!(result, GateOutcome::Fail(GateFailure::NoCitation)));
    }

    #[test]
    fn quality_gate_fails_on_slop() {
        let body = make_body_with_citation(
            &"word "
                .repeat(100)
                .replace("word word word word", "word word we should consider word"),
        );
        let result = quality_gate(&body);
        assert!(matches!(
            result,
            GateOutcome::Fail(GateFailure::SlopPhrase(_))
        ));
    }

    #[test]
    fn quality_gate_fails_on_too_short() {
        let result = quality_gate("Short.");
        assert!(matches!(result, GateOutcome::Fail(GateFailure::TooShort)));
    }

    #[test]
    fn quality_gate_fails_on_too_long() {
        let body = "word ".repeat(WORD_CEILING + 100);
        let result = quality_gate(&body);
        assert!(matches!(result, GateOutcome::Fail(GateFailure::TooLong)));
    }

    #[test]
    fn quality_gate_passes_well_formed() {
        let prose: String = std::iter::repeat_n("word ", 150).collect();
        let body = make_body_with_citation(&prose);
        assert_eq!(quality_gate(&body), GateOutcome::Pass);
    }

    #[test]
    fn template_fallback_lists_projects() {
        let mut e1 = Episode::new("tempera".into(), "x".into());
        e1.id = "ep1".into();
        let mut e2 = Episode::new("smelt".into(), "y".into());
        e2.id = "ep2".into();
        let date = NaiveDate::from_ymd_opt(2026, 5, 24).unwrap();
        let t = template_fallback(&date, &[e1, e2]);
        assert!(t.contains("tempera"));
        assert!(t.contains("smelt"));
        assert!(t.contains("2026-05-24"));
    }

    #[test]
    fn reflection_id_for_with_and_without_project() {
        let d = NaiveDate::from_ymd_opt(2026, 5, 24).unwrap();
        assert_eq!(
            Reflection::id_for(&d, Some("tempera")),
            "2026-05-24-tempera"
        );
        assert_eq!(Reflection::id_for(&d, None), "2026-05-24-all");
    }

    #[tokio::test]
    async fn store_roundtrip() {
        let s = ReflectionStore::open_in_memory().await.unwrap();
        let r = Reflection {
            id: "2026-05-24-all".to_string(),
            date: NaiveDate::from_ymd_opt(2026, 5, 24).unwrap(),
            project: None,
            body: "body".to_string(),
            citations: vec!["abc12345".to_string()],
            signals: vec!["high-falsifiability".to_string()],
            triage_score: 0.8,
            model: "claude-sonnet-4-6".to_string(),
            created_at: Utc::now(),
        };
        s.put(&r).await.unwrap();
        let got = s.get(&r.id).await.unwrap().unwrap();
        assert_eq!(got.id, r.id);
        assert_eq!(got.citations, r.citations);
        assert_eq!(got.signals, r.signals);
        assert!((got.triage_score - 0.8).abs() < 1e-3);
    }

    #[tokio::test]
    async fn store_list_by_date() {
        let s = ReflectionStore::open_in_memory().await.unwrap();
        let d1 = NaiveDate::from_ymd_opt(2026, 5, 23).unwrap();
        let d2 = NaiveDate::from_ymd_opt(2026, 5, 24).unwrap();
        let mk = |id: &str, date: NaiveDate| Reflection {
            id: id.to_string(),
            date,
            project: None,
            body: "x".to_string(),
            citations: vec![],
            signals: vec![],
            triage_score: 0.5,
            model: "m".to_string(),
            created_at: Utc::now(),
        };
        s.put(&mk("a", d1)).await.unwrap();
        s.put(&mk("b", d2)).await.unwrap();
        s.put(&mk("c", d2)).await.unwrap();
        let on_d2 = s.list_by_date(&d2).await.unwrap();
        assert_eq!(on_d2.len(), 2);
        let on_d1 = s.list_by_date(&d1).await.unwrap();
        assert_eq!(on_d1.len(), 1);
    }

    #[test]
    fn build_reflect_user_message_includes_signals_and_eps() {
        use chrono::TimeZone;
        let ts = chrono::Utc.with_ymd_and_hms(2026, 5, 24, 10, 0, 0).unwrap();
        let mut ep = Episode::new("tempera".into(), "x".into());
        ep.id = "abc12345-1234-1234-1234-123456789012".into();
        ep.timestamp_start = ts;
        let msg = build_reflect_user_message(
            &ts.date_naive(),
            &[ep],
            &["high-falsifiability".to_string()],
        );
        assert!(msg.contains("abc12345"));
        assert!(msg.contains("high-falsifiability"));
        assert!(msg.contains("tempera"));
    }
}
