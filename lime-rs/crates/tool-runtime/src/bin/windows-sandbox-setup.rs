use std::path::PathBuf;

fn main() {
    let mut agent_root = None;
    let mut owner = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--agent-root" {
            agent_root = args.next().map(PathBuf::from);
        } else if argument == "--owner" {
            owner = args
                .next()
                .map(|value| value.to_string_lossy().into_owned());
        } else if argument == "--help" {
            println!("usage: windows-sandbox-setup --agent-root <absolute-path> --owner <account>");
            return;
        } else {
            eprintln!("unknown argument: {}", argument.to_string_lossy());
            std::process::exit(2);
        }
    }

    let Some(agent_root) = agent_root else {
        eprintln!("missing required --agent-root");
        std::process::exit(2);
    };
    let owner = owner
        .or_else(|| std::env::var("USERNAME").ok())
        .filter(|value| !value.trim().is_empty());
    let Some(owner) = owner else {
        eprintln!("missing required --owner and USERNAME is unavailable");
        std::process::exit(2);
    };
    match tool_runtime::windows_setup::run_windows_sandbox_setup(&agent_root, &owner) {
        Ok(result) => {
            println!(
                "windows sandbox setup completed: marker={}, users={}, firewall_rules={}, wfp_filters={}",
                result.marker_path.display(),
                result.users_path.display(),
                result.installed_firewall_rule_count,
                result.installed_wfp_filter_count
            );
        }
        Err(error) => {
            eprintln!("windows sandbox setup failed: {error}");
            std::process::exit(1);
        }
    }
}
