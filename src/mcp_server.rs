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
                description: "Search episodic memory for relevant past coding experiences. Use this at the start of a task to find similar problems you've solved before.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language description of what you're trying to do"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of episodes to retrieve (default: 5)",
                            "default": 5
                        },
                        "project": {
                            "type": "string",
                            "description": "Filter by project name (optional)"
                        }
                    },
                    "required": ["query"]
                }),
            },
            Tool {
                name: "memrl_capture".to_string(),
                description: "Capture the current coding session as an episode for future reference. Call this at the end of a successful task.".to_string(),
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
                description: "Record whether retrieved episodes were helpful. This improves future retrieval quality.".to_string(),
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
                name: "memrl_propagate".to_string(),
                description: "Run utility propagation to spread value from helpful episodes to similar ones. Use this periodically to improve memory quality.".to_string(),
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
            "memrl_propagate" => self.tool_propagate(&arguments).await,
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
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing query parameter")?;

        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

        let project = args.get("project").and_then(|v| v.as_str());

        let config = config::Config::load().map_err(|e| e.to_string())?;
        let store = store::EpisodeStore::new().map_err(|e| e.to_string())?;

        // Try vector search first
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

        // Get current project from working directory
        let project = std::env::current_dir()
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
            .unwrap_or_else(|| "unknown".to_string());

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

        // Try to index the new episode
        if let Ok(mut indexer) = indexer::EpisodeIndexer::new().await {
            let _ = indexer.index_episode(&ep).await;
        }

        Ok(format!(
            "Episode captured successfully!\n\
             - ID: {}\n\
             - Project: {}\n\
             - Type: {}\n\
             - Outcome: {}\n\n\
             This experience is now stored for future reference.",
            &ep.id[..8],
            ep.project,
            ep.intent.task_type,
            ep.outcome.status
        ))
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
