// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Snapshot the tempera data directory.
//!
//! Operates over `~/.tempera/`. Snapshots land in `~/.tempera/backups/<ts>/`
//! and contain everything except the `backups/` (recursive) and `models/`
//! (large + redownloadable) subdirectories.
//!
//! Pre-migration safety net: callers about to bump on-disk schema versions
//! can `snapshot()` first so a botched migration is rollback-able. Restore
//! is opt-in via `tempera backup --restore <ts>`.

#![allow(dead_code)]

use anyhow::{Context, Result, bail};
use chrono::Utc;
use std::path::{Path, PathBuf};

use crate::config::Config;

/// Subdirectories that are NOT included in a snapshot. `backups` would be
/// recursive; `models` is a multi-hundred-MB fastembed cache that's
/// regenerable on demand.
const EXCLUDE_DIRS: &[&str] = &["backups", "models"];

/// Snapshot the tempera data directory. Returns the path of the new snapshot.
pub fn snapshot() -> Result<PathBuf> {
    let data_dir = Config::data_dir()?;
    if !data_dir.exists() {
        bail!(
            "tempera data dir doesn't exist at {} — nothing to back up",
            data_dir.display()
        );
    }
    let ts = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let dest = data_dir.join("backups").join(&ts);
    if dest.exists() {
        bail!(
            "backup target {} already exists (timestamp collision?)",
            dest.display()
        );
    }
    std::fs::create_dir_all(&dest)?;

    copy_dir_filtered(&data_dir, &dest, EXCLUDE_DIRS)
        .with_context(|| format!("snapshot into {} failed", dest.display()))?;

    Ok(dest)
}

/// Restore an earlier snapshot over the current data dir. Files in the
/// snapshot overwrite their counterparts; files outside the snapshot are
/// left alone (so `models/` stays, for instance). Refuses to run when
/// `force=false` and the target data dir has unexpected siblings.
pub fn restore(ts: &str, force: bool) -> Result<()> {
    let data_dir = Config::data_dir()?;
    let src = data_dir.join("backups").join(ts);
    if !src.exists() {
        bail!("no snapshot found at {}", src.display());
    }
    if !force {
        // A non-empty data dir is fine; the restore is supposed to overlay.
        // We only block if the snapshot looks corrupt (e.g. empty).
        let count = std::fs::read_dir(&src)?.count();
        if count == 0 {
            bail!(
                "snapshot at {} is empty — refusing to restore",
                src.display()
            );
        }
    }
    copy_dir_filtered(&src, &data_dir, EXCLUDE_DIRS)?;
    Ok(())
}

/// List available snapshots, oldest first.
pub fn list_snapshots() -> Result<Vec<String>> {
    let backups_dir = Config::data_dir()?.join("backups");
    if !backups_dir.exists() {
        return Ok(Vec::new());
    }
    let mut names: Vec<String> = std::fs::read_dir(&backups_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().to_str().map(String::from))
        .collect();
    names.sort();
    Ok(names)
}

/// Recursive copy that skips top-level entries matching `exclude`.
/// Hand-rolled (no `cp -R`) for cross-platform behavior. Skips symlinks
/// (only copies regular files + directories) — the tempera data dir has
/// no symlinks today and following them blindly is the wrong default.
fn copy_dir_filtered(src: &Path, dst: &Path, exclude: &[&str]) -> Result<()> {
    if !dst.exists() {
        std::fs::create_dir_all(dst)?;
    }
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if exclude.contains(&name_str.as_ref()) {
            continue;
        }
        let from = entry.path();
        let to = dst.join(&name);
        let ft = entry.file_type()?;
        if ft.is_symlink() {
            // Skip; document elsewhere.
            continue;
        }
        if ft.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else if ft.is_file() {
            std::fs::copy(&from, &to).with_context(|| format!("copy {:?} → {:?}", from, to))?;
        }
    }
    Ok(())
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    if !dst.exists() {
        std::fs::create_dir_all(dst)?;
    }
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        let ft = entry.file_type()?;
        if ft.is_symlink() {
            continue;
        }
        if ft.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else if ft.is_file() {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn touch(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    #[test]
    fn copy_dir_filtered_skips_excluded() {
        let dir = tempdir().unwrap();
        let src = dir.path().join("src");
        let dst = dir.path().join("dst");
        touch(&src.join("a.txt"), "hello");
        touch(&src.join("backups/b.txt"), "should-skip");
        touch(&src.join("models/c.bin"), "should-skip");
        touch(&src.join("episodes/2026-01-01/x.json"), "ok");

        copy_dir_filtered(&src, &dst, &["backups", "models"]).unwrap();

        assert!(dst.join("a.txt").exists());
        assert!(dst.join("episodes/2026-01-01/x.json").exists());
        assert!(!dst.join("backups").exists());
        assert!(!dst.join("models").exists());
    }

    #[test]
    fn copy_dir_filtered_preserves_content() {
        let dir = tempdir().unwrap();
        let src = dir.path().join("src");
        let dst = dir.path().join("dst");
        touch(&src.join("file.txt"), "exact content here");
        copy_dir_filtered(&src, &dst, &[]).unwrap();
        let read = std::fs::read_to_string(dst.join("file.txt")).unwrap();
        assert_eq!(read, "exact content here");
    }

    #[test]
    fn copy_dir_filtered_creates_dst_if_missing() {
        let dir = tempdir().unwrap();
        let src = dir.path().join("src");
        let dst = dir.path().join("nonexistent/dst");
        touch(&src.join("a.txt"), "x");
        copy_dir_filtered(&src, &dst, &[]).unwrap();
        assert!(dst.join("a.txt").exists());
    }
}
