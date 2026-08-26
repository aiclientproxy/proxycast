fn main() {
    if let Err(error) = tool_runtime::execution_process::run_windows_sandbox_runner() {
        eprintln!("Windows sandbox runner failed: {error}");
        std::process::exit(1);
    }
}
