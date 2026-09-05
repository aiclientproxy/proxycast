use std::process::ExitCode;

use anyhow::Result;
use app_server_protocol::protocol::v2::{
    ThreadQueueAddParams, ThreadQueueAddResponse, ThreadQueueListParams, ThreadQueueListResponse,
    UserInput, METHOD_THREAD_QUEUE_ADD, METHOD_THREAD_QUEUE_LIST,
};
use clap::builder::NonEmptyStringValueParser;
use clap::Args;

use crate::ConnectionArgs;

#[derive(Debug, Args)]
pub(crate) struct QueueCommand {
    #[command(subcommand)]
    subcommand: Option<QueueSubcommand>,
    /// Thread UUID or exact canonical Thread id.
    #[arg(long, value_name = "THREAD", required = false)]
    thread: Option<String>,
    /// Message text to queue.
    #[arg(long, value_name = "TEXT", value_parser = NonEmptyStringValueParser::new(), required = false)]
    message: Option<String>,
    #[command(flatten)]
    connection: ConnectionArgs,
}

#[derive(Debug, clap::Subcommand)]
enum QueueSubcommand {
    /// List canonical submissions waiting on a Thread.
    List(QueueListArgs),
}

#[derive(Debug, Args)]
struct QueueListArgs {
    /// Thread UUID or exact canonical Thread id.
    #[arg(long, value_name = "THREAD")]
    thread: String,
    /// Emit the typed response as JSON.
    #[arg(long)]
    json: bool,
    #[command(flatten)]
    connection: ConnectionArgs,
}

pub(crate) async fn run_queue_command(command: QueueCommand) -> ExitCode {
    match run_inner(command).await {
        Ok(message) => {
            println!("{message}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run_inner(command: QueueCommand) -> Result<String> {
    if let Some(QueueSubcommand::List(args)) = command.subcommand {
        return run_queue_list(args).await;
    }
    let thread = command
        .thread
        .ok_or_else(|| anyhow::anyhow!("--thread is required when queuing a message"))?;
    let message = command
        .message
        .ok_or_else(|| anyhow::anyhow!("--message is required when queuing a message"))?;
    let response = crate::request_value(
        &command.connection,
        METHOD_THREAD_QUEUE_ADD,
        serde_json::to_value(ThreadQueueAddParams {
            thread_id: thread.clone(),
            input: vec![UserInput::Text {
                text: message,
                text_elements: Vec::new(),
            }],
            client_user_message_id: format!("cli-{}", uuid::Uuid::new_v4()),
        })?,
    )
    .await?;
    let response: ThreadQueueAddResponse = serde_json::from_value(response)?;
    Ok(format!(
        "Queued message for thread `{}` as `{}`.",
        thread, response.queued_submission.id
    ))
}

async fn run_queue_list(args: QueueListArgs) -> Result<String> {
    let response = crate::request_value(
        &args.connection,
        METHOD_THREAD_QUEUE_LIST,
        serde_json::to_value(ThreadQueueListParams {
            thread_id: args.thread,
            cursor: None,
            limit: None,
        })?,
    )
    .await?;
    let response: ThreadQueueListResponse = serde_json::from_value(response)?;
    if args.json {
        return Ok(serde_json::to_string_pretty(&response)?);
    }
    if response.data.is_empty() {
        return Ok("No queued messages.".to_string());
    }
    Ok(response
        .data
        .into_iter()
        .map(|submission| {
            let text = submission
                .input
                .into_iter()
                .filter_map(|input| match input {
                    UserInput::Text { text, .. } => Some(text),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" ");
            format!("{}  {}", submission.id, text)
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    #[test]
    fn queue_parses_non_empty_thread_and_message() {
        let cli = crate::MultitoolCli::try_parse_from([
            "lime",
            "queue",
            "--thread",
            "thread-1",
            "--message",
            "follow up",
        ])
        .expect("parse queue command");
        assert!(matches!(cli.subcommand, Some(crate::Subcommand::Queue(_))));
    }

    #[test]
    fn queue_rejects_empty_message() {
        let error = crate::MultitoolCli::try_parse_from([
            "lime",
            "queue",
            "--thread",
            "thread-1",
            "--message",
            "",
        ])
        .expect_err("empty queue message must fail");
        assert_eq!(error.kind(), clap::error::ErrorKind::InvalidValue);
    }

    #[test]
    fn queue_list_parses_thread_and_json() {
        let cli = crate::MultitoolCli::try_parse_from([
            "lime", "queue", "list", "--thread", "thread-1", "--json",
        ])
        .expect("parse queue list command");
        assert!(matches!(cli.subcommand, Some(crate::Subcommand::Queue(_))));
    }
}
