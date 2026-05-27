#!/usr/bin/env node
"use strict";

const { spawn } = require("child_process");
const path = require("path");
const os = require("os");
const fs = require("fs");

// ── Platform / arch mapping ──────────────────────────────────────────

const PLATFORM_MAP = { darwin: "darwin", linux: "linux", win32: "win32" };
const ARCH_MAP = { arm64: "arm64", x64: "x64", x86_64: "x64" };

function getBinaryName() {
  const platform = PLATFORM_MAP[os.platform()];
  const arch = ARCH_MAP[os.arch()];

  if (!platform || !arch) {
    console.error(`Unsupported platform: ${os.platform()}-${os.arch()}`);
    process.exit(1);
  }

  const ext = platform === "win32" ? ".exe" : "";
  return `tempera-mcp-${platform}-${arch}${ext}`;
}

function findBinary() {
  const binaryName = getBinaryName();
  const binDir = __dirname;
  const binaryPath = path.join(binDir, binaryName);

  if (fs.existsSync(binaryPath)) {
    return binaryPath;
  }

  console.error(`Binary not found: ${binaryPath}`);
  console.error(`Platform: ${os.platform()}-${os.arch()}`);
  console.error(
    `Available binaries: ${fs
      .readdirSync(binDir)
      .filter((f) => f.startsWith("tempera-mcp-"))
      .join(", ") || "none"}`
  );
  process.exit(1);
}

// ── ONNX Runtime DLL handling (Windows only) ─────────────────────────
//
// fastembed → ort → ONNX Runtime loads onnxruntime.dll at runtime.
// On Windows this DLL must be discoverable: either next to the binary,
// on PATH, or in a known location. We set ORT_DYLIB_PATH so ort finds
// it without requiring the user to put it on PATH themselves.

function ensureOnnxRuntime() {
  if (os.platform() !== "win32") return;

  const binDir = __dirname;
  const dllPath = path.join(binDir, "onnxruntime.dll");

  if (fs.existsSync(dllPath)) {
    // Tell ort exactly where the DLL is, so it doesn't search PATH.
    process.env.ORT_DYLIB_PATH = dllPath;
    return;
  }

  // DLL not bundled — warn but don't fail. The binary might still
  // find it via PATH or a system install of ONNX Runtime.
  console.error(
    "WARNING: onnxruntime.dll not found alongside the binary."
  );
  console.error(
    "  Windows users need this DLL for the embedding model."
  );
  console.error(
    "  Download from: https://github.com/anvanster/tempera/releases"
  );
}

// ── Spawn the Rust binary ────────────────────────────────────────────

ensureOnnxRuntime();

const binaryPath = findBinary();
const args = process.argv.slice(2);

const child = spawn(binaryPath, args, {
  stdio: ["inherit", "inherit", "inherit"],
  env: process.env,
});

child.on("error", (err) => {
  console.error(`Failed to start tempera-mcp: ${err.message}`);
  process.exit(1);
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.exit(1);
  }
  process.exit(code || 0);
});

// Forward signals to child
for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(sig, () => {
    child.kill(sig);
  });
}
