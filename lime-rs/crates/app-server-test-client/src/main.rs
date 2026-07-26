use app_server_client::AppServerClient;
use app_server_test_client::parse_cli_args;
use app_server_test_client::run_session_facade_stdio_smoke;
use app_server_test_client::run_stdio_smoke;
use app_server_test_client::sample_capability_list_line;
use app_server_test_client::sample_initialize_line;
use app_server_test_client::sample_initialized_line;
use app_server_test_client::sample_session_facade_lines;
use app_server_test_client::sample_session_facade_stdio_lines;
use app_server_test_client::sample_smoke_lines;
use app_server_test_client::HarnessCommand;
use app_server_test_client::StdioLaunchConfig;

fn main() -> anyhow::Result<()> {
    let cli = parse_cli_args(std::env::args().skip(1));
    match cli.command {
        HarnessCommand::InitializeLine { client_name } => {
            print!("{}", sample_initialize_line(client_name)?);
        }
        HarnessCommand::InitializedLine => {
            print!("{}", sample_initialized_line()?);
        }
        HarnessCommand::CapabilityListLine => {
            let mut client = AppServerClient::new();
            print!("{}", sample_capability_list_line(&mut client)?);
        }
        HarnessCommand::SmokeLines { client_name } => {
            for line in sample_smoke_lines(client_name)? {
                print!("{line}");
            }
        }
        HarnessCommand::SessionFacadeLines { client_name } => {
            for line in sample_session_facade_lines(client_name)? {
                print!("{line}");
            }
        }
        HarnessCommand::LaunchStdio {
            app_server_bin,
            extra_args,
        } => {
            let mut config = StdioLaunchConfig::new(app_server_bin);
            config.extra_args = extra_args;
            let lines = sample_smoke_lines("app-server-test-client")?;
            let report = run_stdio_smoke(config, &lines).map_err(anyhow::Error::msg)?;
            println!("{}", report.summary_line());
        }
        HarnessCommand::LaunchSessionFacadeStdio {
            app_server_bin,
            extra_args,
        } => {
            let mut config = StdioLaunchConfig::new(app_server_bin);
            config.extra_args = extra_args;
            let lines = sample_session_facade_stdio_lines("app-server-test-client")?;
            let report =
                run_session_facade_stdio_smoke(config, &lines).map_err(anyhow::Error::msg)?;
            println!("{}", report.summary_line());
        }
    }
    Ok(())
}
