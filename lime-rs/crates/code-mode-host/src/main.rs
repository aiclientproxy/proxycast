#[tokio::main]
async fn main() {
    let listen_url = std::env::args()
        .skip(1)
        .find_map(|arg| arg.strip_prefix("--listen=").map(str::to_owned))
        .unwrap_or_else(|| "stdio".to_string());
    if let Err(error) = code_mode_host::run_main(&listen_url).await {
        eprintln!("code-mode-host failed: {error}");
        std::process::exit(1);
    }
}
