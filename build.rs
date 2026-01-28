//! Build script for MemRL
//!
//! Automatically downloads protoc (Protocol Buffers compiler) if not installed.
//! This is required by LanceDB dependencies.

fn main() {
    // Only download protoc if PROTOC env var is not already set
    if std::env::var("PROTOC").is_err() {
        match protoc_prebuilt::init("27.1") {
            Ok((protoc_bin, _include_path)) => {
                println!("cargo:rustc-env=PROTOC={}", protoc_bin.display());
                // SAFETY: This is single-threaded build script, no other threads accessing env
                unsafe { std::env::set_var("PROTOC", &protoc_bin); }
            }
            Err(e) => {
                println!("cargo:warning=Failed to download protoc: {}", e);
                println!("cargo:warning=Please install protoc manually");
            }
        }
    }
}
