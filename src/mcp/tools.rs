// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;

use super::protocol::Tool;

/// Return all MCP tool definitions
pub(crate) fn tool_definitions() -> Vec<Tool> {
    vec![
        Tool {
            name: "tempera_retrieve".to_string(),
            description: "Search episodic memory for reusable insights from past sessions. Call at session start for non-trivial tasks. Look for: debugging strategies that worked, creative solutions to similar problems, mistakes to avoid, and patterns that transferred across contexts. Focus on retrieving *how* problems were solved, not *what* was changed.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Describe the challenge or pattern you're facing, not just the topic. Good: 'tree-sitter grammar producing ERROR nodes instead of expected AST'. Bad: 'fix codegraph-tcl'. An episode ID also works for full details."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of episodes to retrieve (default: 5)",
                        "default": 5
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter by project name (optional)"
                    },
                    "all": {
                        "type": "boolean",
                        "description": "If true, list all episodes instead of searching (ignores query)",
                        "default": false
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["project", "cross-project"],
                        "description": "v0.10: 'project' (default) stays inside the current project. 'cross-project' also pulls in transferable episodes from other projects — claims marked Forever, Language, Crate, Domain, or Workaround by validity scope. Use cross-project when the question is about a language/library/domain (e.g. async-rust deadlocks, sqlx migrations) and the current project has thin coverage. Project-scoped claims from other projects are always filtered out.",
                        "default": "project"
                    }
                },
                "required": []
            }),
        },
        Tool {
            name: "tempera_capture".to_string(),
            description: "Capture reusable insights as Best Known Methods (BKMs). Capture early and often during sessions — the system automatically consolidates with similar existing BKMs instead of creating duplicates. Focus on TRANSFERABLE KNOWLEDGE: debugging strategies, creative solutions, surprising behaviors, and patterns that would help solve a DIFFERENT problem in a FUTURE session. Litmus test: 'Would this help a model with no context about this project?' If yes, capture it. If it reads like a commit message, rewrite it.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Describe the INSIGHT, not the change. Bad: 'Fixed 24 failing tests in codegraph-tcl visitor.rs'. Good: 'tree-sitter grammars with ABI version patches can split first-position commands into ERROR(keyword)+command(args) sibling pairs. Fix: stitch siblings only when on same line (end_row==start_row) to avoid false joins across lines. Fragmented bodies require scanning scattered simple_word nodes for keywords instead of visiting structured body nodes.'"
                    },
                    "task_type": {
                        "type": "string",
                        "enum": ["bugfix", "feature", "refactor", "test", "docs", "research", "debug", "setup"],
                        "description": "Type of task completed"
                    },
                    "outcome": {
                        "type": "string",
                        "enum": ["success", "partial", "failure"],
                        "description": "Outcome of the task"
                    },
                    "project": {
                        "type": "string",
                        "description": "Override project name (default: auto-detect from working directory). Use for cross-project insights."
                    },
                    "files_modified": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "List of files that were modified"
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Tags for retrieval — use problem-domain terms (e.g., 'tree-sitter', 'error-recovery', 'sibling-stitching'), not project names"
                    },
                    "errors_resolved": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "error": { "type": "string" },
                                "resolution": { "type": "string" }
                            }
                        },
                        "description": "Errors encountered and the STRATEGY used to resolve them — focus on the approach, not the specific code change"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Link this episode to a session. Auto-detected if omitted: reuses session from recent same-project episodes (within 2 hours)."
                    },
                    "alternatives_considered": {
                        "type": "array",
                        "description": "Approaches you nearly took but ruled out, with the REASON each was rejected. The single highest-value field for future-you debugging — it stops the next session from re-exploring rejected paths. Populate when the capture's central claim is genuinely falsifiable (an actual BKM, not a logistics record).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "approach": { "type": "string", "description": "The approach itself, in your own words" },
                                "why_not": { "type": "string", "description": "WHY you rejected it. Without this the entry is useless." },
                                "how_close": {
                                    "type": "string",
                                    "enum": ["near_miss", "plausible", "long_shot"],
                                    "description": "near_miss = would have worked except one specific reason; plausible = correct, traded off against chosen approach; long_shot = dismissed briefly. Default plausible."
                                },
                                "would_revisit_if": {
                                    "type": "string",
                                    "description": "Optional trigger condition: 'revisit me if X changes'"
                                }
                            },
                            "required": ["approach", "why_not"]
                        }
                    },
                    "validity_scope": {
                        "type": "object",
                        "description": "Where does this claim hold? Drives time-decay: 'forever' for language semantics, 'crate' for version-pinned facts, 'workaround' for issue-bound notes. Omit if you're not sure — falls back to 1%/day project-level decay. Choose the narrowest scope that still applies.",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": ["forever", "language", "crate", "domain", "workaround", "project"]
                            },
                            "name": { "type": "string", "description": "For language/crate/project: identifier" },
                            "version": { "type": "string", "description": "For crate: version constraint (=1.43.0, >=1.0,<2.0, ^1.0). Empty = any." },
                            "tag": { "type": "string", "description": "For domain: tag (e.g. 'async-rust')" },
                            "ref_": { "type": "string", "description": "For workaround: issue/PR reference (e.g. 'tokio-rs/tokio#1234')" },
                            "expires": { "type": "string", "description": "For workaround: optional ISO8601 timestamp when this claim is presumed dead" }
                        },
                        "required": ["kind"]
                    }
                },
                "required": ["summary", "task_type", "outcome"]
            }),
        },
        Tool {
            name: "tempera_feedback".to_string(),
            description: "Record whether retrieved episodes actually influenced your approach. Call after using memories. 'Helpful' means the insight changed how you solved the problem — not just that it was topically related.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "episode_ids": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "IDs of episodes to provide feedback on"
                    },
                    "helpful": {
                        "type": "boolean",
                        "description": "Whether the episodes were helpful"
                    }
                },
                "required": ["episode_ids", "helpful"]
            }),
        },
        Tool {
            name: "tempera_stats".to_string(),
            description: "Get statistics or trend analytics about the episodic memory system. Use action 'trends' to see helpfulness over time, domain growth, and learning curve.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Filter stats by project (optional)"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["stats", "trends"],
                        "description": "stats: basic counts and rates. trends: time-series analytics, domain growth, learning curve.",
                        "default": "stats"
                    },
                    "bucket": {
                        "type": "string",
                        "enum": ["weekly", "monthly"],
                        "description": "Bucket size for trends (default: weekly)",
                        "default": "weekly"
                    }
                }
            }),
        },
        Tool {
            name: "tempera_status".to_string(),
            description: "Check memory health for current project. Shows last capture date, episode count, and unused memories. Use this to understand your memory state.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project to check (default: auto-detect from working directory)"
                    }
                }
            }),
        },
        Tool {
            name: "tempera_propagate".to_string(),
            description: "Run utility propagation to spread value from helpful episodes to similar ones. Use periodically to improve memory quality.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "temporal": {
                        "type": "boolean",
                        "description": "Also run temporal credit assignment (credits episodes that preceded successful outcomes)",
                        "default": false
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter propagation to a specific project (optional)"
                    }
                }
            }),
        },
        Tool {
            name: "tempera_review".to_string(),
            description: "Review and consolidate BKMs. Actions: 'analyze' (default) shows duplicate clusters, stale memories, and feedback rate. 'consolidate' merges duplicate clusters into refined BKMs (keeps most recent, union-merges tags/errors/files, deletes duplicates). 'cleanup' removes stale zero-engagement memories. Use consolidate after a series of related tasks to keep memory lean.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project to review (default: auto-detect from working directory)"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["analyze", "consolidate", "cleanup"],
                        "description": "analyze: show duplicate clusters, stale memories, feedback rate. consolidate: merge duplicate clusters into refined BKMs. cleanup: remove stale zero-engagement memories.",
                        "default": "analyze"
                    }
                }
            }),
        },
        Tool {
            name: "tempera_log_correction".to_string(),
            description: "Log a correction the user just made to your assumption, decision, or code. Call this when you notice a user turn is a correction — \"actually X\", \"that's wrong, Y\", \"you missed Z\", \"don't do A, do B\" — not when they're affirming or continuing. Categorize so similar corrections cluster: prefer kebab-case or snake_case topic labels like 'lifetime_annotations', 'test_setup', 'auth_middleware_order'. The future `tempera_brief` surfaces top correction categories for files you're about to edit, so logging consistently makes future-you smarter. DON'T log routine refinements (typos, formatting). DO log: factual errors you made, assumptions you held that turned out wrong, patterns the user had to correct twice.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Short topic label, e.g. 'lifetime_annotations', 'test_setup', 'auth_middleware_order'. Normalized to lowercase snake_case on save so 'Test Setup' and 'test-setup' merge."
                    },
                    "description": {
                        "type": "string",
                        "description": "One sentence: what you got wrong. Plain English, not a commit message."
                    },
                    "correction": {
                        "type": "string",
                        "description": "What the user said the right thing was."
                    },
                    "episode_id": {
                        "type": "string",
                        "description": "Optional: the episode this correction happened during (full or 8-char prefix)."
                    },
                    "files": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Files involved — populates future per-file warning surfaces."
                    },
                    "project": {
                        "type": "string",
                        "description": "Override project name (default: auto-detect from working directory)."
                    }
                },
                "required": ["category", "description", "correction"]
            }),
        },
        Tool {
            name: "tempera_log_should_have_asked".to_string(),
            description: "Log a question you SHOULD have asked the user up front but didn't — and the answer you eventually figured out (often by being told). Call this when, mid-task or after, you realize you assumed something the user could have clarified in one turn. Examples: 'should have asked which auth provider this repo uses before writing the middleware', 'should have asked whether feature flags should default-on'. The `trigger` is the observable context that should fire the question NEXT time — kebab/snake_case like 'edit-auth-middleware', 'new-rust-crate'. The future `tempera_brief` surfaces top triggers when the agent is about to touch a matching file/topic, so logging consistently teaches future-you what to ask before guessing. DO log: ambiguity that wasted a turn, decisions you made and had to undo, assumptions that turned out wrong. DON'T log: questions whose answers are documented in the repo, or one-off project quirks unlikely to recur.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "trigger": {
                        "type": "string",
                        "description": "Observable context label that should make future-you ask the question. E.g. 'edit-auth-middleware', 'new-rust-crate', 'sqlx-migration'. Normalized to lowercase snake_case on save."
                    },
                    "question": {
                        "type": "string",
                        "description": "What you should have asked up front. Phrase as the question itself, not 'I should have asked X' — e.g. 'Which auth provider is wired up?'"
                    },
                    "answer": {
                        "type": "string",
                        "description": "What turned out to be true (often what the user told you when you finally asked). Doubles as fallback knowledge: if asking would be disruptive next time, the prior answer is still a better default than guessing."
                    },
                    "episode_id": {
                        "type": "string",
                        "description": "Optional: the episode this realization happened during (full or 8-char prefix)."
                    },
                    "files": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Files involved — populates the future per-file brief surface."
                    },
                    "project": {
                        "type": "string",
                        "description": "Override project name (default: auto-detect from working directory)."
                    }
                },
                "required": ["trigger", "question", "answer"]
            }),
        },
        Tool {
            name: "tempera_brief".to_string(),
            description: "One-call summary of everything tempera has learned about the exact patch of code you're about to touch. Joins your working set against five v0.8 surfaces: pending ask-backs (questions to ask the user first), reasoning templates (the step sequence past wins followed), top correction categories for these files (where you've been wrong here before), should-have-asked triggers (questions to ask up front in this kind of context), and calibration warnings (your verified vs declared success rate for this task_type/project). Call this AT TASK START once you know the files; pass task_type + domain too for the template section to fire. Sections are omitted from the response when empty, so a short response means tempera has nothing specific to surface — fall back to tempera_retrieve in that case. Pairs with tempera_session_start as the standard warmup duo.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Files the agent is about to edit / read / consider. Used to join against mistakes + should-have-asked. Empty list means no file-scoped sections fire."
                    },
                    "task_type": {
                        "type": "string",
                        "description": "Optional: bugfix | feature | refactor | test | docs | research | debug | setup. Enables the calibration warning + template lookup."
                    },
                    "domain": {
                        "type": "string",
                        "description": "Optional: domain tag, e.g. 'rust', 'async-rust', 'sqlx'. Combined with task_type to pull a reasoning template if one exists."
                    },
                    "project": {
                        "type": "string",
                        "description": "Project name (default: auto-detect from working directory)."
                    },
                    "cross_project": {
                        "type": "boolean",
                        "description": "v0.10.2: when true, the mistakes + should-have-asked sections also surface rows from OTHER projects whose files overlap. Foreign-project rows are tagged `[from <project>]` so you can see the boundary. Use when current project has thin history but the working set is in a well-understood language/framework — corrections that recur across projects (lifetime errors, sqlx patterns) are often more useful than the noise of one project. Pending ask-back, calibration, and template sections stay project-scoped regardless — those signals don't generalize.",
                        "default": false
                    }
                },
                "required": ["files"]
            }),
        },
        Tool {
            name: "tempera_session_start".to_string(),
            description: "Call this ONCE at the very start of a session in a project — before you start exploring, reading files, or planning. If tempera previously observed an episode in this project end in failure/partial with vague intent, it drafted ONE clarifying question via Haiku. This tool returns that question so you can ask the user before guessing the same way again. If there's no pending question the response says so and you can proceed normally. Calling this is cheap (one DB lookup) and the question gets marked 'served' so it won't surface again. Best-paired with tempera_template and tempera_retrieve as the standard session-warmup trio.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project name (default: auto-detect from working directory)."
                    }
                }
            }),
        },
        Tool {
            name: "tempera_template".to_string(),
            description: "Pull the reasoning template that worked for a (task_type, domain) pair the last time tempera observed several successful episodes there. Call this at task start, BEFORE you start exploring or editing, when the kind of task is clear — e.g. (bugfix, async-rust), (refactor, sqlx-migrations), (feature, mcp-handlers). The response is an ordered list of steps the agent has historically followed when this kind of task succeeded; treat them as a starting checklist, not a rigid script. If no template exists yet for the pair, fall back to tempera_retrieve. Templates accrue during the dream cycle — they get better with use, so missing template = early days, not an error.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "description": "One of: bugfix | feature | refactor | test | docs | research | debug | setup. Case-insensitive."
                    },
                    "domain": {
                        "type": "string",
                        "description": "Domain tag the task touches, e.g. 'rust', 'async-rust', 'sqlx', 'mcp-handlers'. Must match one of the domains episodes were tagged with during capture."
                    }
                },
                "required": ["task_type", "domain"]
            }),
        },
    ]
}
