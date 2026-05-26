// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Ask-back generation pipeline (v0.8.5) — Haiku-backed.
//!
//! Called from the capture path on best-effort terms: any failure here
//! must NOT fail the capture. Two-stage gate before any LLM call:
//!
//! 1. **Outcome gate**: the episode's outcome must be Failure or
//!    Partial. Success episodes don't need a clarifying question.
//! 2. **Vague-intent gate**: the episode's intent has to look vague —
//!    short extracted_intent, or low claim falsifiability, or both.
//!    Specific intents that failed are bugs, not ambiguity gaps.
//!
//! And then **at-most-one-pending-per-project** debounce is enforced
//! by the DB partial unique index — the capture hook calls `record`
//! unconditionally and gets `AlreadyPending` back as the natural rate
//! limiter.

#![allow(dead_code)]

use anyhow::{Context, Result};
use chrono::Utc;
use serde::Deserialize;

use crate::ask_backs::{AskBack, AskBackStatus, AskBackStore, RecordOutcome};
use crate::config::Config;
use crate::episode::{Episode, OutcomeStatus};
use crate::llm::AnthropicClient;

/// Approximate USD per generation call. Haiku-class; lives here so the
/// capture path can decide whether to attempt generation without
/// reaching into [[triage]].
pub const ASK_BACK_ESTIMATED_COST_USD: f32 = 0.002;

/// Lower-bound intent length to consider the intent "specific enough"
/// that ambiguity is unlikely to be the cause of failure. Anything
/// shorter falls into the vague bucket.
const SPECIFIC_INTENT_MIN_CHARS: usize = 80;

/// Falsifiability below this threshold counts as vague even if the
/// intent string is long. Matches the falsifiability scale used in
/// `llm::extract_intent` — anything under 0.4 is essentially logistics
/// or a generic ask.
const SPECIFIC_FALSIFIABILITY_MIN: f32 = 0.4;

/// Returns true if the episode looks like a candidate for an ask-back.
/// Pure function — no IO, no LLM call. Cheap, safe to call from the
/// capture hot path before deciding to spend budget.
pub fn is_ask_back_candidate(ep: &Episode) -> bool {
    if !matches!(
        ep.outcome.status,
        OutcomeStatus::Failure | OutcomeStatus::Partial
    ) {
        return false;
    }
    let intent = if !ep.intent.extracted_intent.is_empty() {
        ep.intent.extracted_intent.as_str()
    } else {
        ep.intent.raw_prompt.as_str()
    };
    if intent.trim().is_empty() {
        // Empty intent = obviously vague.
        return true;
    }
    if intent.chars().count() < SPECIFIC_INTENT_MIN_CHARS {
        return true;
    }
    if let Some(claim) = &ep.intent.claim
        && claim.falsifiability < SPECIFIC_FALSIFIABILITY_MIN
    {
        return true;
    }
    false
}

const ASK_BACK_SYSTEM: &str = r#"You are a memory system that just observed a coding episode end in failure or partial success. The agent's stated intent looked vague — likely it tried to act before fully understanding what the user wanted.

Your job: draft ONE clarifying question the agent should ask the next time it starts a similar task in this project. The question gets queued for the next session, so phrase it so the agent will know to ask the user up front.

Rules:
- Exactly ONE question. No preamble, no commentary.
- The question must be answerable in one short sentence from the user.
- Focus on the SPECIFIC ambiguity that likely caused the failure. Do NOT ask generic things like "what is your goal?"
  Bad:  "What are you trying to accomplish?"
  Good: "Should the auth middleware run before or after request logging?"
- Phrase as a question the AGENT will ask the USER, not vice versa.
- Output JSON ONLY: {"question": "..."}
- If you genuinely cannot identify a specific ambiguity worth asking about, output: {"question": ""}
"#;

#[derive(Debug, Deserialize)]
struct AuthoredQuestion {
    #[serde(default)]
    question: String,
}

fn build_user_message(ep: &Episode) -> String {
    let intent = if !ep.intent.extracted_intent.is_empty() {
        ep.intent.extracted_intent.as_str()
    } else {
        ep.intent.raw_prompt.as_str()
    };
    let preview: String = intent.chars().take(400).collect();
    let files: Vec<String> = ep.context.files_modified.iter().take(8).cloned().collect();
    let files_line = if files.is_empty() {
        String::new()
    } else {
        format!("Files touched: {}\n", files.join(", "))
    };
    let errors_line = if ep.context.errors_encountered.is_empty() {
        String::new()
    } else {
        let first_err = &ep.context.errors_encountered[0];
        format!(
            "First error: {}: {}\n",
            first_err.error_type,
            first_err.message.chars().take(200).collect::<String>()
        )
    };
    format!(
        "Project: {project}\n\
         Task type: {tt}\n\
         Outcome: {outcome}\n\
         Stated intent: {intent}\n\
         {files}{errors}",
        project = ep.project,
        tt = ep.intent.task_type,
        outcome = ep.outcome.status,
        intent = preview,
        files = files_line,
        errors = errors_line,
    )
}

fn strip_json_fences(s: &str) -> &str {
    let s = s.trim();
    let s = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s);
    s.trim_end_matches("```").trim()
}

/// Draft a clarifying question via Haiku. Returns `Ok(None)` if the
/// model declined (empty question = "no specific ambiguity worth
/// asking about"). All errors are propagated so the caller can choose
/// whether to absorb them.
pub async fn draft_question(ep: &Episode, config: &Config) -> Result<Option<String>> {
    let model = &config.dream.triage_model;
    let client = AnthropicClient::with_model(model)?;
    let user = build_user_message(ep);
    let raw = client.raw_completion(ASK_BACK_SYSTEM, &user, 200).await?;
    let trimmed = strip_json_fences(&raw);
    let authored: AuthoredQuestion = serde_json::from_str(trimmed)
        .with_context(|| format!("failed to parse ask_back JSON: {raw}"))?;
    let q = authored.question.trim().to_string();
    if q.is_empty() {
        return Ok(None);
    }
    Ok(Some(q))
}

/// Best-effort: maybe generate + persist an ask-back for `ep`. Never
/// fails the caller — returns `Ok(false)` on any internal error.
/// Returns `Ok(true)` only if a pending row was actually written.
pub async fn maybe_generate_and_record(ep: &Episode, config: &Config) -> Result<bool> {
    if !is_ask_back_candidate(ep) {
        return Ok(false);
    }
    let store = match AskBackStore::open_default().await {
        Ok(s) => s,
        Err(_) => return Ok(false),
    };
    // Short-circuit: cheap DB check before spending the Haiku budget.
    if store
        .has_pending_for_project(&ep.project)
        .await
        .unwrap_or(false)
    {
        return Ok(false);
    }
    let question = match draft_question(ep, config).await {
        Ok(Some(q)) => q,
        Ok(None) => return Ok(false),
        Err(_) => return Ok(false),
    };
    let ab = AskBack {
        id: None,
        project: ep.project.clone(),
        episode_id: ep.id.clone(),
        question,
        status: AskBackStatus::Pending,
        model: config.dream.triage_model.clone(),
        created_at: Utc::now(),
        served_at: None,
    };
    Ok(matches!(
        store
            .record(&ab)
            .await
            .unwrap_or(RecordOutcome::AlreadyPending),
        RecordOutcome::Inserted(_)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episode::{
        Claim, ClaimCategory, Context, Intent, Outcome, OutcomeStatus, TaskType, Utility,
        VerificationState,
    };

    fn mk_episode(intent_text: &str, status: OutcomeStatus, claim: Option<Claim>) -> Episode {
        Episode {
            schema_version: 5,
            id: "ep-1".into(),
            timestamp_start: Utc::now(),
            timestamp_end: Utc::now(),
            project: "p".into(),
            intent: Intent {
                raw_prompt: intent_text.to_string(),
                extracted_intent: intent_text.to_string(),
                task_type: TaskType::Bugfix,
                domain: vec!["rust".into()],
                claim,
            },
            context: Context {
                files_read: vec![],
                files_modified: vec!["src/foo.rs".into()],
                tools_invoked: vec![],
                errors_encountered: vec![],
            },
            outcome: Outcome {
                status,
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

    #[test]
    fn success_outcome_is_never_a_candidate() {
        let ep = mk_episode("fix bug", OutcomeStatus::Success, None);
        assert!(!is_ask_back_candidate(&ep));
    }

    #[test]
    fn failure_with_short_intent_is_a_candidate() {
        let ep = mk_episode("fix bug", OutcomeStatus::Failure, None);
        assert!(is_ask_back_candidate(&ep));
    }

    #[test]
    fn partial_with_short_intent_is_a_candidate() {
        let ep = mk_episode("fix bug", OutcomeStatus::Partial, None);
        assert!(is_ask_back_candidate(&ep));
    }

    #[test]
    fn long_specific_intent_is_not_a_candidate_without_low_falsifiability() {
        let long_intent = "fix the borrow-checker error in the auth middleware where the request guard outlives the spawned future".to_string();
        assert!(long_intent.chars().count() >= SPECIFIC_INTENT_MIN_CHARS);
        let ep = mk_episode(&long_intent, OutcomeStatus::Failure, None);
        assert!(!is_ask_back_candidate(&ep));
    }

    #[test]
    fn long_intent_with_low_falsifiability_is_a_candidate() {
        let long_intent = "make the tests pass — I keep getting random failures and I think the timing logic is off somewhere".to_string();
        let claim = Claim {
            falsifiability: 0.2,
            category: ClaimCategory::Logistics,
            validity_scope: None,
        };
        let ep = mk_episode(&long_intent, OutcomeStatus::Failure, Some(claim));
        assert!(is_ask_back_candidate(&ep));
    }

    #[test]
    fn empty_intent_is_a_candidate() {
        let ep = mk_episode("", OutcomeStatus::Failure, None);
        assert!(is_ask_back_candidate(&ep));
    }

    #[test]
    fn build_user_message_includes_key_fields() {
        let ep = mk_episode("fix bug", OutcomeStatus::Failure, None);
        let msg = build_user_message(&ep);
        assert!(msg.contains("p")); // project
        assert!(msg.contains("bugfix"));
        assert!(msg.contains("fix bug"));
        assert!(msg.contains("src/foo.rs"));
    }

    #[test]
    fn strip_json_fences_handles_markdown() {
        assert_eq!(
            strip_json_fences("```json\n{\"question\":\"x\"}\n```"),
            "{\"question\":\"x\"}"
        );
        assert_eq!(strip_json_fences("  {\"q\":1}  "), "{\"q\":1}");
    }
}
