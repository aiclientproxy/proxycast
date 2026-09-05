use anyhow::Result;
use clap::Parser;
use execpolicy::ExecPolicyCheckCommand;

/// CLI for evaluating exec policies.
#[derive(Parser)]
#[command(name = "execpolicy")]
enum Cli {
    /// Evaluate a command against a policy.
    Check(ExecPolicyCheckCommand),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli {
        Cli::Check(cmd) => cmd.run(),
    }
}
