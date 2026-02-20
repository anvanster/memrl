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
                    }
                },
                "required": []
            }),
        },
        Tool {
            name: "tempera_capture".to_string(),
            description: "Capture reusable insights after completing a task. Focus on TRANSFERABLE KNOWLEDGE — not what changed (commits track that), but what you LEARNED. Capture: debugging strategies, creative solutions, surprising behaviors, mistakes avoided, and patterns that would help solve a DIFFERENT problem in a FUTURE session. Litmus test: 'Would this help a model with no context about this project?' If yes, capture it. If it reads like a commit message, rewrite it.".to_string(),
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
            description: "Get statistics about the episodic memory system.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Filter stats by project (optional)"
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
            description: "Review and consolidate memories after completing a series of related tasks. Identifies duplicate/similar episodes, stale memories, and optimization opportunities.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project to review (default: auto-detect from working directory)"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["analyze", "cleanup"],
                        "description": "analyze: show recommendations only. cleanup: apply safe optimizations (removes zero-utility duplicates)",
                        "default": "analyze"
                    }
                }
            }),
        },
    ]
}
