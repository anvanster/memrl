// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod handlers;
pub(crate) mod helpers;
pub(crate) mod protocol;
pub(crate) mod tools;

use protocol::{JsonRpcError, JsonRpcResponse};
use serde_json::{Value, json};

/// MCP Server implementation
pub(crate) struct McpServer {
    initialized: bool,
}

impl McpServer {
    pub fn new() -> Self {
        Self { initialized: false }
    }

    /// Handle incoming JSON-RPC request
    pub async fn handle_request(&mut self, request: protocol::JsonRpcRequest) -> JsonRpcResponse {
        let id = request.id.clone().unwrap_or(Value::Null);

        let result = match request.method.as_str() {
            "initialize" => self.handle_initialize(),
            "initialized"
            | "notifications/initialized"
            | "notifications/cancelled"
            | "notifications/roots/list_changed" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": tools::tool_definitions() })),
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
    fn handle_initialize(&mut self) -> Result<Value, JsonRpcError> {
        self.initialized = true;
        Ok(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "tempera",
                "version": env!("CARGO_PKG_VERSION")
            }
        }))
    }

    /// Handle tools/call request — dispatches to handler modules
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
            "tempera_retrieve" => handlers::retrieve::handle(&arguments).await,
            "tempera_capture" => handlers::capture::handle(&arguments).await,
            "tempera_feedback" => handlers::feedback::handle(&arguments).await,
            "tempera_stats" => handlers::stats::handle(&arguments).await,
            "tempera_status" => handlers::status::handle(&arguments).await,
            "tempera_propagate" => handlers::propagate::handle(&arguments).await,
            "tempera_review" => handlers::review::handle(&arguments).await,
            "tempera_log_correction" => handlers::log_correction::handle(&arguments).await,
            "tempera_template" => handlers::template::handle(&arguments).await,
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
}
