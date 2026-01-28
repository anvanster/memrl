//! MemRL MCP Server
//!
//! This binary implements the Model Context Protocol (MCP) server for MemRL,
//! allowing Claude Code to access episodic memory functionality.

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
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

mod config;
mod episode;
mod feedback;
mod indexer;
mod retrieve;
mod stats;
mod store;
mod utility;

/// MCP Server implementation
struct McpServer {
    initialized: bool,
}

/// JSON-RPC 2.0 Request
#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

/// JSON-RPC 2.0 Response
#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 Error
#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

/// MCP Tool definition
#[derive(Debug, Serialize)]
struct Tool {
    name: String,
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

impl McpServer {
    fn new() -> Self {
        Self { initialized: false }
    }

    /// Handle incoming JSON-RPC request
    async fn handle_request(&mut self, request: JsonRpcRequest) -> JsonRpcResponse {
        let id = request.id.clone().unwrap_or(Value::Null);

        let result = match request.method.as_str() {
            "initialize" => self.handle_initialize(&request.params),
            "initialized" => Ok(json!({})),
            "tools/list" => self.handle_tools_list(),
            "tools/call" => self.handle_tools_call(&request.params).await,
            "shutdown" => {
                self.initialized = false;
                Ok(json!({}))
            }
            _ => Err(JsonRpcError {
                code: -32601,
                message: format!("Method not found: {}", request.method),
                data: None,
            }),
        };

        match result {
            Ok(value) => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: Some(value),
                error: None,
            },
            Err(error) => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: None,
                error: Some(error),
            },
        }
    }

    /// Handle initialize request
    fn handle_initialize(&mut self, _params: &Value) -> Result<Value, JsonRpcError> {
        self.initialized = true;
        Ok(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "memrl",
                "version": env!("CARGO_PKG_VERSION")
            }
        }))
    }

    /// Handle tools/list request
    fn handle_tools_list(&self) -> Result<Value, JsonRpcError> {
        let tools = vec![
            Tool {
                name: "memrl_retrieve".to_string(),
                description: "MANDATORY at session start for non-trivial tasks. Search episodic memory for similar problems you've solved before. If this is your first action in a session, always check for relevant memories first.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language description of what you're trying to do, OR an episode ID to get full details"
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
                name: "memrl_capture".to_string(),
                description: "MANDATORY after completing any feature, bugfix, or refactor. Don't wait for user to ask - proactively capture successful sessions. Automatically runs utility propagation after capture.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Brief summary of what was accomplished"
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
                            "description": "Domain tags for categorization"
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
                            "description": "Errors encountered and how they were resolved"
                        }
                    },
                    "required": ["summary", "task_type", "outcome"]
                }),
            },
            Tool {
                name: "memrl_feedback".to_string(),
                description: "Record whether retrieved episodes were helpful. Call this after using memories - your feedback improves future retrieval quality.".to_string(),
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
                name: "memrl_stats".to_string(),
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
                name: "memrl_status".to_string(),
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
                name: "memrl_propagate".to_string(),
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
                name: "memrl_review".to_string(),
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
        ];

        Ok(json!({ "tools": tools }))
    }

    /// Handle tools/call request
    async fn handle_tools_call(&self, params: &Value) -> Result<Value, JsonRpcError> {
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| JsonRpcError {
                code: -32602,
                message: "Missing tool name".to_string(),
                data: None,
            })?;

        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

        let result = match name {
            "memrl_retrieve" => self.tool_retrieve(&arguments).await,
            "memrl_capture" => self.tool_capture(&arguments).await,
            "memrl_feedback" => self.tool_feedback(&arguments).await,
            "memrl_stats" => self.tool_stats(&arguments).await,
            "memrl_status" => self.tool_status(&arguments).await,
            "memrl_propagate" => self.tool_propagate(&arguments).await,
            "memrl_review" => self.tool_review(&arguments).await,
            _ => Err(format!("Unknown tool: {}", name)),
        };

        match result {
            Ok(content) => Ok(json!({
                "content": [{
                    "type": "text",
                    "text": content
                }]
            })),
            Err(e) => Ok(json!({
                "content": [{
                    "type": "text",
                    "text": format!("Error: {}", e)
                }],
                "isError": true
            })),
        }
    }

    /// Retrieve relevant episodes
    async fn tool_retrieve(&self, args: &Value) -> Result<String, String> {
        let query = args.get("query").and_then(|v| v.as_str());
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let project = args.get("project").and_then(|v| v.as_str());
        let list_all = args.get("all").and_then(|v| v.as_bool()).unwrap_or(false);

        let config = config::Config::load().map_err(|e| e.to_string())?;
        let store = store::EpisodeStore::new().map_err(|e| e.to_string())?;

        // Case 1: List all episodes
        if list_all {
            return self.list_all_episodes(&store, limit, project).await;
        }

        // Need query for other cases
        let query = query.ok_or("Missing query parameter (or use all: true to list episodes)")?;

        // Case 2: Query looks like an episode ID - show full details
        if Self::looks_like_episode_id(query) {
            if let Some(output) = self.show_episode_by_id(&store, query).await? {
                return Ok(output);
            }
            // If not found by ID, fall through to search
        }

        // Case 3: Semantic search
        let episodes = match try_vector_retrieve(query, limit, project, &config).await {
            Ok(eps) if !eps.is_empty() => eps,
            _ => {
                // Fallback to text search
                retrieve::retrieve_episodes_text(query, limit, project, &config, &store)
                    .map_err(|e| e.to_string())?
            }
        };

        if episodes.is_empty() {
            return Ok("No relevant episodes found in memory.".to_string());
        }

        // Format results
        let mut output = format!("Found {} relevant past experiences:\n\n", episodes.len());

        for (i, scored) in episodes.iter().enumerate() {
            let ep = &scored.episode;
            output.push_str(&format!(
                "{}. **{}**\n",
                i + 1,
                if ep.intent.extracted_intent.is_empty() {
                    &ep.intent.raw_prompt
                } else {
                    &ep.intent.extracted_intent
                }
            ));
            output.push_str(&format!("   - ID: {}\n", &ep.id[..8]));
            output.push_str(&format!("   - Project: {}\n", ep.project));
            output.push_str(&format!("   - Type: {}\n", ep.intent.task_type));
            output.push_str(&format!("   - Outcome: {}\n", ep.outcome.status));
            output.push_str(&format!(
                "   - Relevance: {:.0}% similarity, {:.0}% utility\n",
                scored.similarity_score * 100.0,
                scored.utility_score * 100.0
            ));

            if !ep.context.files_modified.is_empty() {
                output.push_str(&format!(
                    "   - Files: {}\n",
                    ep.context.files_modified.join(", ")
                ));
            }

            if !ep.intent.domain.is_empty() {
                output.push_str(&format!("   - Tags: {}\n", ep.intent.domain.join(", ")));
            }

            // Show resolved errors if any
            let resolved: Vec<_> = ep
                .context
                .errors_encountered
                .iter()
                .filter(|e| e.resolved)
                .collect();
            if !resolved.is_empty() {
                output.push_str("   - Errors resolved:\n");
                for err in resolved.iter().take(2) {
                    output.push_str(&format!("     - {}\n", err.message));
                    if let Some(res) = &err.resolution {
                        output.push_str(&format!("       Resolution: {}\n", res));
                    }
                }
            }

            output.push('\n');
        }

        output.push_str("Use memrl_feedback to indicate if these were helpful.");

        // Record retrieval for tracking
        let _ = record_mcp_retrieval(&episodes, query, &store);

        Ok(output)
    }

    /// Capture a new episode
    async fn tool_capture(&self, args: &Value) -> Result<String, String> {
        let summary = args
            .get("summary")
            .and_then(|v| v.as_str())
            .ok_or("Missing summary parameter")?;

        let task_type_str = args
            .get("task_type")
            .and_then(|v| v.as_str())
            .ok_or("Missing task_type parameter")?;

        let outcome_str = args
            .get("outcome")
            .and_then(|v| v.as_str())
            .ok_or("Missing outcome parameter")?;

        let files_modified: Vec<String> = args
            .get("files_modified")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let tags: Vec<String> = args
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        // Get project from args or auto-detect from working directory
        let project = args
            .get("project")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| {
                std::env::current_dir()
                    .ok()
                    .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
                    .unwrap_or_else(|| "unknown".to_string())
            });

        // Create episode
        let mut ep = episode::Episode::new(project.clone(), summary.to_string());

        // Set task type
        ep.intent.task_type = match task_type_str {
            "bugfix" => episode::TaskType::Bugfix,
            "feature" => episode::TaskType::Feature,
            "refactor" => episode::TaskType::Refactor,
            "test" => episode::TaskType::Test,
            "docs" => episode::TaskType::Docs,
            "research" => episode::TaskType::Research,
            "debug" => episode::TaskType::Debug,
            "setup" => episode::TaskType::Setup,
            _ => episode::TaskType::Unknown,
        };

        // Set outcome
        ep.outcome.status = match outcome_str {
            "success" => episode::OutcomeStatus::Success,
            "partial" => episode::OutcomeStatus::Partial,
            "failure" => episode::OutcomeStatus::Failure,
            _ => episode::OutcomeStatus::Partial,
        };

        // Set context
        ep.context.files_modified = files_modified;
        ep.intent.domain = tags;
        ep.intent.extracted_intent = summary.to_string();

        // Parse errors if provided
        if let Some(errors) = args.get("errors_resolved").and_then(|v| v.as_array()) {
            for err in errors {
                if let (Some(error_msg), resolution) = (
                    err.get("error").and_then(|v| v.as_str()),
                    err.get("resolution").and_then(|v| v.as_str()),
                ) {
                    ep.context.errors_encountered.push(episode::ErrorRecord {
                        error_type: "runtime".to_string(),
                        message: error_msg.to_string(),
                        resolved: resolution.is_some(),
                        resolution: resolution.map(String::from),
                    });
                }
            }
        }

        // Update end timestamp
        ep.timestamp_end = chrono::Utc::now();

        // Save episode
        let store = store::EpisodeStore::new().map_err(|e| e.to_string())?;
        store.save(&ep).map_err(|e| e.to_string())?;

        // Index the new episode
        if let Ok(mut indexer) = indexer::EpisodeIndexer::new().await {
            let _ = indexer.index_episode(&ep).await;
        }

        let mut output = format!(
            "Episode captured successfully!\n\
             - ID: {}\n\
             - Project: {}\n\
             - Type: {}\n\
             - Outcome: {}\n",
            &ep.id[..8],
            ep.project,
            ep.intent.task_type,
            ep.outcome.status
        );

        // Auto-propagate utility to spread value
        output.push_str("\n📈 Running auto-propagation...\n");
        let propagate_result = self.run_propagation(Some(project.as_str()), false).await;
        match propagate_result {
            Ok(msg) => output.push_str(&msg),
            Err(e) => output.push_str(&format!("  (propagation skipped: {})\n", e)),
        }

        output.push_str("\nThis experience is now stored for future reference.");
        Ok(output)
    }

    /// Record feedback on episodes
    async fn tool_feedback(&self, args: &Value) -> Result<String, String> {
        let episode_ids: Vec<String> = args
            .get("episode_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .ok_or("Missing episode_ids parameter")?;

        let helpful = args
            .get("helpful")
            .and_then(|v| v.as_bool())
            .ok_or("Missing helpful parameter")?;

        let store = store::EpisodeStore::new().map_err(|e| e.to_string())?;
        let mut updated = 0;

        for id in &episode_ids {
            // Try to find episode by partial ID
            if let Ok(episodes) = store.list_all() {
                for ep in episodes {
                    if ep.id.starts_with(id) || ep.id[..8] == *id {
                        let mut episode = ep.clone();

                        // Update utility
                        if helpful {
                            episode.utility.helpful_count += 1;
                        }
                        episode.utility.score = Some(episode.utility.calculate_score());

                        // Update last retrieval feedback
                        if let Some(last) = episode.retrieval_history.last_mut() {
                            last.was_helpful = Some(helpful);
                        }

                        if store.update(&episode).is_ok() {
                            updated += 1;
                        }
                        break;
                    }
                }
            }
        }

        let feedback_type = if helpful { "helpful" } else { "not helpful" };
        Ok(format!(
            "Feedback recorded: {} episode(s) marked as {}.\n\
             This helps improve future retrieval quality.",
            updated, feedback_type
        ))
    }

    /// Get memory statistics
    async fn tool_stats(&self, args: &Value) -> Result<String, String> {
        let project_filter = args.get("project").and_then(|v| v.as_str());

        let store = store::EpisodeStore::new().map_err(|e| e.to_string())?;
        let episodes = store.list_all().map_err(|e| e.to_string())?;

        let filtered: Vec<_> = if let Some(proj) = project_filter {
            episodes
                .iter()
                .filter(|e| e.project.to_lowercase().contains(&proj.to_lowercase()))
                .collect()
        } else {
            episodes.iter().collect()
        };

        let total = filtered.len();
        let successful = filtered
            .iter()
            .filter(|e| e.outcome.status == episode::OutcomeStatus::Success)
            .count();
        let total_retrievals: u32 = filtered.iter().map(|e| e.utility.retrieval_count).sum();
        let total_helpful: u32 = filtered.iter().map(|e| e.utility.helpful_count).sum();

        let avg_utility: f32 = if total > 0 {
            filtered
                .iter()
                .map(|e| e.utility.calculate_score())
                .sum::<f32>()
                / total as f32
        } else {
            0.0
        };

        // Task type breakdown
        let mut task_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for ep in &filtered {
            *task_counts
                .entry(ep.intent.task_type.to_string())
                .or_insert(0) += 1;
        }

        let mut output = String::from("MemRL Memory Statistics\n");
        output.push_str("=======================\n\n");

        if let Some(proj) = project_filter {
            output.push_str(&format!("Project: {}\n\n", proj));
        }

        output.push_str(&format!("Total Episodes: {}\n", total));
        output.push_str(&format!(
            "Success Rate: {:.1}%\n",
            if total > 0 {
                (successful as f32 / total as f32) * 100.0
            } else {
                0.0
            }
        ));
        output.push_str(&format!("Total Retrievals: {}\n", total_retrievals));
        output.push_str(&format!("Helpful Retrievals: {}\n", total_helpful));
        output.push_str(&format!("Average Utility: {:.1}%\n\n", avg_utility * 100.0));

        output.push_str("By Task Type:\n");
        for (task_type, count) in task_counts {
            output.push_str(&format!("  - {}: {}\n", task_type, count));
        }

        Ok(output)
    }

    /// Run utility propagation
    async fn tool_propagate(&self, args: &Value) -> Result<String, String> {
        let temporal = args
            .get("temporal")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let project_filter = args.get("project").and_then(|v| v.as_str());

        let store = store::EpisodeStore::new().map_err(|e| e.to_string())?;
        let _config = config::Config::load().map_err(|e| e.to_string())?;
        let params = utility::UtilityParams::default();

        let mut output = String::from("📈 Running utility propagation...\n\n");

        // Get episodes (optionally filtered by project)
        let all_episodes = store.list_all().map_err(|e| e.to_string())?;
        let episodes: Vec<_> = if let Some(proj) = project_filter {
            all_episodes
                .into_iter()
                .filter(|e| e.project.to_lowercase().contains(&proj.to_lowercase()))
                .collect()
        } else {
            all_episodes
        };

        output.push_str(&format!("Processing {} episodes...\n", episodes.len()));

        // Apply decay
        let mut decayed_count = 0;
        for ep in &episodes {
            if let Some(last_retrieval) = ep.retrieval_history.last() {
                let days_since = (chrono::Utc::now() - last_retrieval.timestamp).num_days() as f64;
                if days_since > 0.0 {
                    let decay = (1.0 - params.decay_rate).powf(days_since);
                    if decay < 0.99 {
                        decayed_count += 1;
                    }
                }
            }
        }

        output.push_str(&format!(
            "  📉 Decay applied to {} episodes\n",
            decayed_count
        ));

        // Run Bellman propagation using vector similarity
        let mut propagated_count = 0;
        let mut propagation_delta = 0.0f64;

        if let Ok(indexer) = indexer::EpisodeIndexer::new().await {
            if indexer.is_indexed().await {
                // Find high-value episodes to propagate from
                let high_value: Vec<_> = episodes
                    .iter()
                    .filter(|e| e.utility.calculate_score() > 0.6)
                    .collect();

                for source_ep in high_value {
                    // Find similar episodes
                    let query = format!(
                        "{} {}",
                        source_ep.intent.raw_prompt,
                        source_ep.intent.domain.join(" ")
                    );

                    if let Ok(similar) = indexer.search(&query, 5, project_filter).await {
                        for result in similar {
                            if result.id != source_ep.id
                                && result.similarity_score >= params.propagation_threshold
                            {
                                if let Ok(mut target_ep) = store.load(&result.id) {
                                    let source_utility = source_ep.utility.calculate_score();
                                    let target_utility = target_ep.utility.calculate_score();

                                    // Bellman update: Q(s) += α * (γ * Q(s') - Q(s)) * similarity
                                    let update = params.learning_rate as f32
                                        * (params.discount_factor as f32 * source_utility
                                            - target_utility)
                                        * result.similarity_score;

                                    if update.abs() > 0.001 {
                                        target_ep.utility.score =
                                            Some((target_utility + update).clamp(0.0, 1.0));
                                        if store.update(&target_ep).is_ok() {
                                            propagated_count += 1;
                                            propagation_delta += update as f64;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        output.push_str(&format!(
            "  🔄 Propagated value to {} episodes\n",
            propagated_count
        ));
        output.push_str(&format!(
            "  📊 Total utility change: {:+.3}\n",
            propagation_delta
        ));

        // Temporal credit assignment
        if temporal {
            output.push_str("\n⏱️  Running temporal credit assignment...\n");

            let credited = utility::temporal_credit_assignment(&store, project_filter, &params)
                .map_err(|e| e.to_string())?;

            output.push_str(&format!("  ✅ Credited {} episodes\n", credited));
        }

        // Sync to vector index
        if let Ok(mut indexer) = indexer::EpisodeIndexer::new().await {
            let updated_episodes = store.list_all().map_err(|e| e.to_string())?;
            for ep in &updated_episodes {
                let _ = indexer.index_episode(ep).await;
            }
            output.push_str("  💾 Synced to vector index\n");
        }

        output.push_str("\n✅ Propagation complete!");

        Ok(output)
    }

    /// Check memory status for current project
    async fn tool_status(&self, args: &Value) -> Result<String, String> {
        let project = args
            .get("project")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| {
                std::env::current_dir()
                    .ok()
                    .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
                    .unwrap_or_else(|| "unknown".to_string())
            });

        let store = store::EpisodeStore::new().map_err(|e| e.to_string())?;
        let all_episodes = store.list_all().map_err(|e| e.to_string())?;

        let project_episodes: Vec<_> = all_episodes
            .iter()
            .filter(|e| e.project.to_lowercase() == project.to_lowercase())
            .collect();

        let total_count = project_episodes.len();

        if total_count == 0 {
            return Ok(format!(
                "📊 Memory Status for '{}'\n\
                 ========================\n\n\
                 No memories found for this project.\n\n\
                 💡 Tip: After completing a task, use memrl_capture to save it.",
                project
            ));
        }

        // Find last capture date
        let last_capture = project_episodes
            .iter()
            .map(|e| e.timestamp_start)
            .max()
            .unwrap();
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
        let avg_utility: f32 = if total_count > 0 {
            project_episodes
                .iter()
                .map(|e| e.utility.calculate_score())
                .sum::<f32>()
                / total_count as f32
        } else {
            0.0
        };

        // Find high-value memories
        let high_value: Vec<_> = project_episodes
            .iter()
            .filter(|e| e.utility.calculate_score() > 0.6)
            .collect();

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
            "📈 Average utility: {:.0}%\n\n",
            avg_utility * 100.0
        ));

        // Suggestions
        output.push_str("💡 Suggestions:\n");

        if days_since_capture > 7 {
            output.push_str("  - You haven't captured memories recently. Remember to capture after completing tasks!\n");
        }

        if unused.len() > total_count / 2 {
            output.push_str(
                "  - Many memories are unused. Consider running memrl_review to consolidate.\n",
            );
        }

        if avg_utility < 0.3 {
            output.push_str(
                "  - Low average utility. Use memrl_feedback to mark helpful memories.\n",
            );
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

    /// Review and consolidate memories
    async fn tool_review(&self, args: &Value) -> Result<String, String> {
        let project = args
            .get("project")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| {
                std::env::current_dir()
                    .ok()
                    .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
                    .unwrap_or_else(|| "unknown".to_string())
            });

        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("analyze");

        let store = store::EpisodeStore::new().map_err(|e| e.to_string())?;
        let all_episodes = store.list_all().map_err(|e| e.to_string())?;

        let project_episodes: Vec<_> = all_episodes
            .into_iter()
            .filter(|e| e.project.to_lowercase() == project.to_lowercase())
            .collect();

        if project_episodes.is_empty() {
            return Ok(format!("No memories found for project '{}'.", project));
        }

        let mut output = format!("🔍 Memory Review for '{}'\n", project);
        output.push_str(&"=".repeat(22 + project.len()));
        output.push_str("\n\n");

        // Find stale memories (old with low utility)
        let stale: Vec<_> = project_episodes
            .iter()
            .filter(|e| {
                let age_days = (chrono::Utc::now() - e.timestamp_start).num_days();
                let utility = e.utility.calculate_score();
                age_days > 30 && utility < 0.2
            })
            .collect();

        // Find potential duplicates (similar intents)
        let mut duplicates: Vec<(&episode::Episode, &episode::Episode)> = Vec::new();
        for i in 0..project_episodes.len() {
            for j in (i + 1)..project_episodes.len() {
                let e1 = &project_episodes[i];
                let e2 = &project_episodes[j];
                // Simple check: same task type and similar summary
                if e1.intent.task_type == e2.intent.task_type {
                    let s1 = e1.intent.extracted_intent.to_lowercase();
                    let s2 = e2.intent.extracted_intent.to_lowercase();
                    // Check word overlap
                    let words1: std::collections::HashSet<_> = s1.split_whitespace().collect();
                    let words2: std::collections::HashSet<_> = s2.split_whitespace().collect();
                    let intersection = words1.intersection(&words2).count();
                    let union = words1.union(&words2).count();
                    if union > 0 && (intersection as f64 / union as f64) > 0.6 {
                        duplicates.push((e1, e2));
                    }
                }
            }
        }

        // Find zero-utility episodes
        let zero_utility: Vec<_> = project_episodes
            .iter()
            .filter(|e| e.utility.calculate_score() < 0.05 && e.utility.retrieval_count == 0)
            .collect();

        output.push_str("📊 Analysis Results:\n");
        output.push_str(&format!("  - Total memories: {}\n", project_episodes.len()));
        output.push_str(&format!("  - Stale (>30d, low utility): {}\n", stale.len()));
        output.push_str(&format!("  - Potential duplicates: {}\n", duplicates.len()));
        output.push_str(&format!(
            "  - Zero utility (never used): {}\n\n",
            zero_utility.len()
        ));

        if !stale.is_empty() {
            output.push_str("📅 Stale Memories:\n");
            for ep in stale.iter().take(5) {
                let summary: String = ep.intent.extracted_intent.chars().take(50).collect();
                output.push_str(&format!("  - {} ({}...)\n", &ep.id[..8], summary));
            }
            if stale.len() > 5 {
                output.push_str(&format!("  ... and {} more\n", stale.len() - 5));
            }
            output.push('\n');
        }

        if !duplicates.is_empty() {
            output.push_str("🔄 Potential Duplicates:\n");
            for (e1, e2) in duplicates.iter().take(3) {
                output.push_str(&format!("  - {} ≈ {}\n", &e1.id[..8], &e2.id[..8]));
            }
            if duplicates.len() > 3 {
                output.push_str(&format!("  ... and {} more pairs\n", duplicates.len() - 3));
            }
            output.push('\n');
        }

        if action == "cleanup" {
            output.push_str("🧹 Cleanup Actions:\n");
            let mut removed = 0;

            // Only remove zero-utility episodes that are also potential duplicates
            for ep in &zero_utility {
                let is_duplicate = duplicates
                    .iter()
                    .any(|(e1, e2)| e1.id == ep.id || e2.id == ep.id);
                if is_duplicate {
                    if store.delete(&ep.id).is_ok() {
                        removed += 1;
                        output.push_str(&format!(
                            "  ✓ Removed {} (zero-utility duplicate)\n",
                            &ep.id[..8]
                        ));
                    }
                }
            }

            if removed == 0 {
                output.push_str("  No safe cleanup actions available.\n");
                output.push_str("  (Only zero-utility duplicates are auto-removed)\n");
            } else {
                output.push_str(&format!("\n✅ Removed {} episode(s)\n", removed));
            }
        } else {
            output.push_str("💡 Recommendations:\n");
            if !zero_utility.is_empty() && !duplicates.is_empty() {
                output
                    .push_str("  - Run with action: 'cleanup' to remove zero-utility duplicates\n");
            }
            if !stale.is_empty() {
                output.push_str("  - Consider manually reviewing stale memories\n");
            }
            if duplicates.is_empty() && stale.is_empty() && zero_utility.is_empty() {
                output.push_str("  - Your memory is healthy! No issues found.\n");
            }
        }

        Ok(output)
    }

    /// Helper: Run propagation (used by capture and propagate tools)
    async fn run_propagation(
        &self,
        project_filter: Option<&str>,
        temporal: bool,
    ) -> Result<String, String> {
        let store = store::EpisodeStore::new().map_err(|e| e.to_string())?;
        let params = utility::UtilityParams::default();

        let all_episodes = store.list_all().map_err(|e| e.to_string())?;
        let episodes: Vec<_> = if let Some(proj) = project_filter {
            all_episodes
                .into_iter()
                .filter(|e| e.project.to_lowercase().contains(&proj.to_lowercase()))
                .collect()
        } else {
            all_episodes
        };

        if episodes.is_empty() {
            return Ok("  No episodes to propagate.\n".to_string());
        }

        let mut propagated_count = 0;

        if let Ok(indexer) = indexer::EpisodeIndexer::new().await {
            if indexer.is_indexed().await {
                let high_value: Vec<_> = episodes
                    .iter()
                    .filter(|e| e.utility.calculate_score() > 0.6)
                    .collect();

                for source_ep in high_value {
                    let query = format!(
                        "{} {}",
                        source_ep.intent.raw_prompt,
                        source_ep.intent.domain.join(" ")
                    );

                    if let Ok(similar) = indexer.search(&query, 5, project_filter).await {
                        for result in similar {
                            if result.id != source_ep.id
                                && result.similarity_score >= params.propagation_threshold
                            {
                                if let Ok(mut target_ep) = store.load(&result.id) {
                                    let source_utility = source_ep.utility.calculate_score();
                                    let target_utility = target_ep.utility.calculate_score();

                                    let update = params.learning_rate as f32
                                        * (params.discount_factor as f32 * source_utility
                                            - target_utility)
                                        * result.similarity_score;

                                    if update.abs() > 0.001 {
                                        target_ep.utility.score =
                                            Some((target_utility + update).clamp(0.0, 1.0));
                                        if store.update(&target_ep).is_ok() {
                                            propagated_count += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut output = format!("  Propagated value to {} episode(s)\n", propagated_count);

        if temporal {
            let credited = utility::temporal_credit_assignment(&store, project_filter, &params)
                .map_err(|e| e.to_string())?;
            output.push_str(&format!("  Temporal credit to {} episode(s)\n", credited));
        }

        Ok(output)
    }

    /// Check if a string looks like an episode ID
    fn looks_like_episode_id(s: &str) -> bool {
        // Episode IDs are UUIDs (36 chars with hyphens) or short IDs (8 hex chars)
        let s = s.trim();
        if s.len() == 8 && s.chars().all(|c| c.is_ascii_hexdigit()) {
            return true;
        }
        if s.len() == 36 && s.chars().all(|c| c.is_ascii_hexdigit() || c == '-') {
            return true;
        }
        false
    }

    /// List all episodes
    async fn list_all_episodes(
        &self,
        store: &store::EpisodeStore,
        limit: usize,
        project: Option<&str>,
    ) -> Result<String, String> {
        let mut episodes = store.list_all().map_err(|e| e.to_string())?;

        // Filter by project if specified
        if let Some(proj) = project {
            episodes.retain(|e| e.project.to_lowercase().contains(&proj.to_lowercase()));
        }

        // Sort by timestamp (newest first)
        episodes.sort_by(|a, b| b.timestamp_start.cmp(&a.timestamp_start));

        // Apply limit
        episodes.truncate(limit);

        if episodes.is_empty() {
            return Ok("No episodes found in memory.".to_string());
        }

        let mut output = format!("Listing {} episode(s):\n\n", episodes.len());

        for (i, ep) in episodes.iter().enumerate() {
            let summary = if ep.intent.extracted_intent.is_empty() {
                &ep.intent.raw_prompt
            } else {
                &ep.intent.extracted_intent
            };
            // Truncate summary for list view
            let summary_short: String = summary.chars().take(60).collect();
            let ellipsis = if summary.len() > 60 { "..." } else { "" };

            output.push_str(&format!("{}. **{}{}**\n", i + 1, summary_short, ellipsis));
            output.push_str(&format!("   - ID: {}\n", &ep.id[..8]));
            output.push_str(&format!("   - Project: {}\n", ep.project));
            output.push_str(&format!(
                "   - Type: {} | Outcome: {}\n",
                ep.intent.task_type, ep.outcome.status
            ));
            output.push_str(&format!(
                "   - Date: {}\n",
                ep.timestamp_start.format("%Y-%m-%d %H:%M")
            ));
            if !ep.intent.domain.is_empty() {
                output.push_str(&format!("   - Tags: {}\n", ep.intent.domain.join(", ")));
            }
            output.push('\n');
        }

        Ok(output)
    }

    /// Show full episode details by ID
    async fn show_episode_by_id(
        &self,
        store: &store::EpisodeStore,
        id: &str,
    ) -> Result<Option<String>, String> {
        // Try to find episode by ID (full or partial)
        let episodes = store.list_all().map_err(|e| e.to_string())?;

        let episode = episodes
            .iter()
            .find(|e| e.id.starts_with(id) || e.id[..8] == *id);

        let ep = match episode {
            Some(e) => e,
            None => return Ok(None), // Not found, let search handle it
        };

        let mut output = String::from("Episode Details\n");
        output.push_str("===============\n\n");

        output.push_str(&format!("**ID**: {}\n", ep.id));
        output.push_str(&format!("**Project**: {}\n", ep.project));
        output.push_str(&format!("**Type**: {}\n", ep.intent.task_type));
        output.push_str(&format!("**Outcome**: {}\n", ep.outcome.status));
        output.push_str(&format!(
            "**Date**: {} - {}\n",
            ep.timestamp_start.format("%Y-%m-%d %H:%M"),
            ep.timestamp_end.format("%H:%M")
        ));
        output.push_str(&format!(
            "**Utility**: {:.0}%\n\n",
            ep.utility.calculate_score() * 100.0
        ));

        output.push_str("## Intent\n");
        if !ep.intent.extracted_intent.is_empty() {
            output.push_str(&format!("{}\n\n", ep.intent.extracted_intent));
        }
        output.push_str(&format!("**Raw prompt**: {}\n\n", ep.intent.raw_prompt));

        if !ep.intent.domain.is_empty() {
            output.push_str(&format!("**Tags**: {}\n\n", ep.intent.domain.join(", ")));
        }

        if !ep.context.files_modified.is_empty() {
            output.push_str("## Files Modified\n");
            for f in &ep.context.files_modified {
                output.push_str(&format!("- {}\n", f));
            }
            output.push('\n');
        }

        if !ep.context.errors_encountered.is_empty() {
            output.push_str("## Errors Encountered\n");
            for err in &ep.context.errors_encountered {
                output.push_str(&format!("- **{}**: {}\n", err.error_type, err.message));
                if let Some(res) = &err.resolution {
                    output.push_str(&format!("  - Resolution: {}\n", res));
                }
            }
            output.push('\n');
        }

        output.push_str("## Retrieval Stats\n");
        output.push_str(&format!(
            "- Retrieved: {} times\n",
            ep.utility.retrieval_count
        ));
        output.push_str(&format!(
            "- Marked helpful: {} times\n",
            ep.utility.helpful_count
        ));

        Ok(Some(output))
    }
}

/// Try vector-based retrieval
async fn try_vector_retrieve(
    query: &str,
    limit: usize,
    project_filter: Option<&str>,
    config: &config::Config,
) -> Result<Vec<retrieve::ScoredEpisode>> {
    let indexer = indexer::EpisodeIndexer::new().await?;

    if !indexer.is_indexed().await {
        anyhow::bail!("Index not available");
    }

    let store = store::EpisodeStore::new()?;
    let search_results = indexer.search(query, limit * 2, project_filter).await?;

    let mut episodes = Vec::new();
    for result in search_results {
        if let Ok(episode) = store.load(&result.id) {
            let utility = episode.utility.calculate_score();
            let combined = (1.0 - config.retrieval.utility_weight) * result.similarity_score
                + config.retrieval.utility_weight * utility;

            episodes.push(retrieve::ScoredEpisode {
                episode,
                similarity_score: result.similarity_score,
                utility_score: utility,
                combined_score: combined,
            });
        }
    }

    episodes.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    episodes.retain(|e| e.similarity_score >= config.retrieval.min_similarity);
    episodes.truncate(limit);

    Ok(episodes)
}

/// Record retrieval for tracking
fn record_mcp_retrieval(
    episodes: &[retrieve::ScoredEpisode],
    query: &str,
    store: &store::EpisodeStore,
) -> Result<()> {
    let project = std::env::current_dir()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
        .unwrap_or_else(|| "unknown".to_string());

    for scored in episodes {
        let mut episode = scored.episode.clone();
        episode.retrieval_history.push(episode::RetrievalRecord {
            timestamp: chrono::Utc::now(),
            project: project.clone(),
            task_description: query.to_string(),
            was_helpful: None,
        });
        episode.utility.retrieval_count += 1;
        store.update(&episode)?;
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut server = McpServer::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    // Process JSON-RPC messages line by line
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        if line.trim().is_empty() {
            continue;
        }

        // Parse JSON-RPC request
        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let error_response = JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Value::Null,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("Parse error: {}", e),
                        data: None,
                    }),
                };
                writeln!(stdout, "{}", serde_json::to_string(&error_response)?)?;
                stdout.flush()?;
                continue;
            }
        };

        // Handle request
        let response = server.handle_request(request).await;

        // Send response
        writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
        stdout.flush()?;
    }

    Ok(())
}
