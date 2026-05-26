// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tempera MCP Server
//!
//! This binary implements the Model Context Protocol (MCP) server for Tempera,
//! allowing Claude Code to access episodic memory functionality.

#![allow(clippy::collapsible_if)]
#![allow(clippy::single_char_add_str)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::lines_filter_map_ok)]
#![allow(clippy::manual_ok_err)]
#![allow(clippy::for_kv_map)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::ptr_arg)]

use anyhow::Result;
use std::io::{self, BufRead, Write};

mod asks;
mod calibration;
mod config;
mod episode;
mod feedback;
mod fingerprint;
mod indexer;
mod keyword;
mod mcp;
mod mistakes;
mod retrieve;
mod stats;
mod store;
mod templates;
mod utility;

#[tokio::main]
async fn main() -> Result<()> {
    let mut server = mcp::McpServer::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        if line.trim().is_empty() {
            continue;
        }

        let request: mcp::protocol::JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let error_response = mcp::protocol::JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: serde_json::Value::Null,
                    result: None,
                    error: Some(mcp::protocol::JsonRpcError {
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

        // JSON-RPC notifications have no `id` — must not send a response
        let is_notification = request.id.is_none();
        let response = server.handle_request(request).await;

        if !is_notification {
            writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
            stdout.flush()?;
        }
    }

    Ok(())
}
