#[tokio::main]
async fn main() {
    if let Err(error) = tool_runtime::code_mode::run_code_mode_host_stdio().await {
        eprintln!("code-mode-host failed: {error}");
        std::process::exit(1);
    }
}
