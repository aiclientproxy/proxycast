mod catalog;
mod video;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use lime_core::config::load_config;
use lime_media_runtime::{
    build_image_generation_endpoint, execute_image_generation_task as execute_image_task_runtime,
    list_task_outputs, load_task_output, patch_task_artifact, retry_task_artifact,
    update_task_status, write_task_artifact, ImageGenerationRunnerConfig, MediaRuntimeError,
    MediaTaskErrorOutput, TaskArtifactPatch, TaskErrorRecord, TaskProgress, TaskRelationships,
    TaskType, TaskWriteOptions, DEFAULT_ARTIFACT_ROOT, IMAGE_TASK_RUNNER_WORKER_ID,
};
use serde_json::{json, Value};

const AUDIO_TRANSCRIPTION_CONTRACT_KEY: &str = "audio_transcription";
const AUDIO_TRANSCRIPTION_MODALITY: &str = "audio";
const AUDIO_TRANSCRIPTION_ROUTING_SLOT: &str = "audio_transcription_model";
const AUDIO_TRANSCRIPTION_REQUIRED_CAPABILITIES: &[&str] =
    &["text_generation", "audio_transcription"];

use crate::catalog::{
    find_skill_entry, find_task_entry, ROOT_AFTER_HELP, SKILL_AFTER_HELP, SKILL_ENTRIES,
    TASK_AFTER_HELP, TASK_ENTRIES,
};

const ROOT_LONG_ABOUT: &str = "\
Lime 官方任务 CLI。

默认输出结构化 JSON，优先服务 Agent 与技能调用。
当前主线覆盖内容生成、链接解析、排版优化与素材检索等任务型业务。";

#[derive(Debug, Parser)]
#[command(
    name = "lime",
    version,
    about = "Lime 官方任务 CLI",
    long_about = ROOT_LONG_ABOUT,
    after_help = ROOT_AFTER_HELP,
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Task(TaskCommand),
    Skill(SkillCommand),
    Doctor(DoctorArgs),
    #[command(hide = true)]
    Media(MediaCommand),
}

#[derive(Debug, Args)]
#[command(after_help = TASK_AFTER_HELP, arg_required_else_help = true)]
struct TaskCommand {
    #[command(subcommand)]
    command: TaskSubcommand,
}

#[derive(Debug, Subcommand)]
// 保持 clap derive 的直接子命令形态；这里是 CLI 解析结构，不在热路径上。
#[allow(clippy::large_enum_variant)]
enum TaskSubcommand {
    Create(TaskCreateCommand),
    Status(TaskLookupArgs),
    List(TaskListArgs),
    Attempts(TaskLookupArgs),
    Retry(TaskLookupArgs),
    Cancel(TaskLookupArgs),
    Result(TaskLookupArgs),
}

#[derive(Debug, Args)]
#[command(arg_required_else_help = true)]
struct TaskCreateCommand {
    #[command(subcommand)]
    command: TaskCreateSubcommand,
}

#[derive(Debug, Subcommand)]
enum TaskCreateSubcommand {
    Image(ImageGenerateArgs),
    Cover(CoverGenerateArgs),
    Video(VideoGenerateArgs),
    Transcription(TranscriptionGenerateArgs),
    Broadcast(BroadcastGenerateArgs),
    #[command(name = "url-parse")]
    UrlParse(UrlParseArgs),
    Typesetting(TypesettingArgs),
    #[command(name = "resource-search")]
    ResourceSearch(ResourceSearchArgs),
}

#[derive(Debug, Args)]
#[command(after_help = SKILL_AFTER_HELP, arg_required_else_help = true)]
struct SkillCommand {
    #[command(subcommand)]
    command: SkillSubcommand,
}

#[derive(Debug, Subcommand)]
enum SkillSubcommand {
    List,
    Show(SkillShowArgs),
}

#[derive(Debug, Args)]
struct SkillShowArgs {
    name: String,
}

#[derive(Debug, Args)]
struct DoctorArgs {
    #[arg(long)]
    workspace: Option<PathBuf>,
    #[arg(long = "artifact-dir")]
    artifact_dir: Option<String>,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args, Clone)]
struct SharedTaskWriteArgs {
    #[arg(long)]
    workspace: Option<PathBuf>,
    #[arg(long)]
    output: Option<String>,
    #[arg(long = "artifact-dir")]
    artifact_dir: Option<String>,
    #[arg(long = "idempotency-key")]
    idempotency_key: Option<String>,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args, Clone)]
struct SharedTaskReadArgs {
    #[arg(long)]
    workspace: Option<PathBuf>,
    #[arg(long = "artifact-dir")]
    artifact_dir: Option<String>,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct TaskLookupArgs {
    task_ref: String,
    #[command(flatten)]
    shared: SharedTaskReadArgs,
}

#[derive(Debug, Args)]
struct TaskListArgs {
    #[command(flatten)]
    shared: SharedTaskReadArgs,
    #[arg(long)]
    status: Option<String>,
    #[arg(long = "family")]
    task_family: Option<String>,
    #[arg(long = "type")]
    task_type: Option<String>,
    #[arg(long)]
    limit: Option<usize>,
}

#[derive(Debug, Args, Clone)]
struct ImageGenerateArgs {
    #[arg(long)]
    prompt: String,
    #[arg(long)]
    title: Option<String>,
    #[arg(long)]
    mode: Option<String>,
    #[arg(long = "raw-text")]
    raw_text: Option<String>,
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    style: Option<String>,
    #[arg(long)]
    size: Option<String>,
    #[arg(long = "aspect-ratio")]
    aspect_ratio: Option<String>,
    #[arg(long)]
    count: Option<u32>,
    #[arg(long)]
    usage: Option<String>,
    #[arg(long = "provider-id")]
    provider_id: Option<String>,
    #[arg(long = "session-id")]
    session_id: Option<String>,
    #[arg(long = "project-id")]
    project_id: Option<String>,
    #[arg(long = "content-id")]
    content_id: Option<String>,
    #[arg(long = "entry-source")]
    entry_source: Option<String>,
    #[arg(long = "requested-target")]
    requested_target: Option<String>,
    #[arg(long = "slot-id")]
    slot_id: Option<String>,
    #[arg(long = "anchor-hint")]
    anchor_hint: Option<String>,
    #[arg(long = "anchor-section-title")]
    anchor_section_title: Option<String>,
    #[arg(long = "anchor-text")]
    anchor_text: Option<String>,
    #[arg(long = "target-output-id")]
    target_output_id: Option<String>,
    #[arg(long = "target-output-ref-id")]
    target_output_ref_id: Option<String>,
    #[arg(long = "reference-image")]
    reference_images: Vec<String>,
    #[command(flatten)]
    output: SharedTaskWriteArgs,
}

#[derive(Debug, Args, Clone)]
struct CoverGenerateArgs {
    #[arg(long)]
    prompt: String,
    #[arg(long = "raw-text")]
    raw_text: Option<String>,
    #[arg(long)]
    title: Option<String>,
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    style: Option<String>,
    #[arg(long)]
    platform: Option<String>,
    #[arg(long)]
    size: Option<String>,
    #[arg(long = "image-url")]
    image_url: Option<String>,
    #[arg(long = "reference-image-url")]
    reference_image_url: Option<String>,
    #[arg(long)]
    usage: Option<String>,
    #[arg(long = "session-id")]
    session_id: Option<String>,
    #[arg(long = "project-id")]
    project_id: Option<String>,
    #[arg(long = "content-id")]
    content_id: Option<String>,
    #[arg(long = "entry-source")]
    entry_source: Option<String>,
    #[command(flatten)]
    output: SharedTaskWriteArgs,
}

#[derive(Debug, Args, Clone)]
struct VideoGenerateArgs {
    #[arg(long)]
    prompt: String,
    #[arg(long)]
    title: Option<String>,
    #[arg(long = "project-id")]
    project_id: Option<String>,
    #[arg(long = "provider-id")]
    provider_id: Option<String>,
    #[arg(long)]
    model: Option<String>,
    #[arg(long = "aspect-ratio")]
    aspect_ratio: Option<String>,
    #[arg(long)]
    resolution: Option<String>,
    #[arg(long)]
    duration: Option<i64>,
    #[arg(long = "image-url")]
    image_url: Option<String>,
    #[arg(long = "end-image-url")]
    end_image_url: Option<String>,
    #[arg(long)]
    seed: Option<i64>,
    #[arg(long = "generate-audio")]
    generate_audio: Option<bool>,
    #[arg(long = "camera-fixed")]
    camera_fixed: Option<bool>,
    #[command(flatten)]
    output: SharedTaskWriteArgs,
}

#[derive(Debug, Args, Clone)]
struct TranscriptionGenerateArgs {
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long)]
    title: Option<String>,
    #[arg(long = "raw-text")]
    raw_text: Option<String>,
    #[arg(long = "source-url")]
    source_url: Option<String>,
    #[arg(long = "source-path")]
    source_path: Option<String>,
    #[arg(long)]
    language: Option<String>,
    #[arg(long = "output-format")]
    output_format: Option<String>,
    #[arg(long = "speaker-labels")]
    speaker_labels: Option<bool>,
    #[arg(long)]
    timestamps: Option<bool>,
    #[arg(long = "provider-id")]
    provider_id: Option<String>,
    #[arg(long)]
    model: Option<String>,
    #[arg(long = "session-id")]
    session_id: Option<String>,
    #[arg(long = "project-id")]
    project_id: Option<String>,
    #[arg(long = "content-id")]
    content_id: Option<String>,
    #[arg(long = "entry-source")]
    entry_source: Option<String>,
    #[command(flatten)]
    output: SharedTaskWriteArgs,
}

#[derive(Debug, Args, Clone)]
struct BroadcastGenerateArgs {
    #[arg(long)]
    content: String,
    #[arg(long)]
    title: Option<String>,
    #[arg(long)]
    audience: Option<String>,
    #[arg(long)]
    tone: Option<String>,
    #[arg(long = "duration-hint-minutes")]
    duration_hint_minutes: Option<u32>,
    #[command(flatten)]
    output: SharedTaskWriteArgs,
}

#[derive(Debug, Args, Clone)]
struct UrlParseArgs {
    #[arg(long)]
    url: String,
    #[arg(long)]
    title: Option<String>,
    #[arg(long)]
    summary: Option<String>,
    #[arg(long = "key-point")]
    key_points: Vec<String>,
    #[arg(long = "extract-status", default_value = "ready")]
    extract_status: String,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long = "raw-text")]
    raw_text: Option<String>,
    #[arg(long = "extract-goal")]
    extract_goal: Option<String>,
    #[arg(long = "session-id")]
    session_id: Option<String>,
    #[arg(long = "project-id")]
    project_id: Option<String>,
    #[arg(long = "content-id")]
    content_id: Option<String>,
    #[arg(long = "entry-source")]
    entry_source: Option<String>,
    #[command(flatten)]
    output: SharedTaskWriteArgs,
}

#[derive(Debug, Args, Clone)]
struct TypesettingArgs {
    #[arg(long)]
    content: String,
    #[arg(long)]
    title: Option<String>,
    #[arg(long = "target-platform")]
    target_platform: String,
    #[arg(long = "rule")]
    rules: Vec<String>,
    #[command(flatten)]
    output: SharedTaskWriteArgs,
}

#[derive(Debug, Args, Clone)]
struct ResourceSearchArgs {
    #[arg(long = "resource-type")]
    resource_type: String,
    #[arg(long)]
    query: String,
    #[arg(long)]
    title: Option<String>,
    #[arg(long)]
    usage: String,
    #[arg(long, default_value_t = 6)]
    count: u32,
    #[arg(long = "constraint")]
    constraints: Vec<String>,
    #[command(flatten)]
    output: SharedTaskWriteArgs,
}

#[derive(Debug, Args)]
struct MediaCommand {
    #[command(subcommand)]
    command: MediaSubcommand,
}

#[derive(Debug, Subcommand)]
enum MediaSubcommand {
    Image(ImageCommand),
    Cover(CoverCommand),
    Video(VideoCommand),
    Transcription(TranscriptionCommand),
}

#[derive(Debug, Args)]
struct ImageCommand {
    #[command(subcommand)]
    command: ImageSubcommand,
}

#[derive(Debug, Args)]
struct CoverCommand {
    #[command(subcommand)]
    command: CoverSubcommand,
}

#[derive(Debug, Args)]
struct VideoCommand {
    #[command(subcommand)]
    command: VideoSubcommand,
}

#[derive(Debug, Subcommand)]
enum TranscriptionSubcommand {
    Generate(TranscriptionGenerateArgs),
}

#[derive(Debug, Args)]
struct TranscriptionCommand {
    #[command(subcommand)]
    command: TranscriptionSubcommand,
}

#[derive(Debug, Subcommand)]
enum ImageSubcommand {
    Generate(ImageGenerateArgs),
}

#[derive(Debug, Subcommand)]
enum CoverSubcommand {
    Generate(CoverGenerateArgs),
}

#[derive(Debug, Subcommand)]
enum VideoSubcommand {
    Generate(VideoGenerateArgs),
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(output) => {
            print_json(&output, false);
            ExitCode::SUCCESS
        }
        Err(error) => {
            let payload = MediaTaskErrorOutput::from_error(&error);
            let serialized =
                serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string());
            eprintln!("{serialized}");
            ExitCode::from(error.exit_code().clamp(1, 255) as u8)
        }
    }
}

fn print_json(value: &Value, stderr: bool) {
    let serialized = serde_json::to_string_pretty(value).unwrap_or_else(|_| "{}".to_string());
    if stderr {
        eprintln!("{serialized}");
    } else {
        println!("{serialized}");
    }
}

fn run(cli: Cli) -> Result<Value, MediaRuntimeError> {
    match cli.command {
        Command::Task(task) => run_task_command(task),
        Command::Skill(skill) => run_skill_command(skill),
        Command::Doctor(doctor) => run_doctor_command(doctor),
        Command::Media(media) => run_media_command(media),
    }
}

fn run_task_command(command: TaskCommand) -> Result<Value, MediaRuntimeError> {
    match command.command {
        TaskSubcommand::Create(create) => run_task_create_command(create),
        TaskSubcommand::Status(args) => {
            let workspace_root = resolve_workspace_root(args.shared.workspace)?;
            let output = load_task_output(
                &workspace_root,
                &args.task_ref,
                args.shared.artifact_dir.as_deref(),
            )?;
            Ok(json!(output))
        }
        TaskSubcommand::Attempts(args) => {
            let workspace_root = resolve_workspace_root(args.shared.workspace)?;
            let output = load_task_output(
                &workspace_root,
                &args.task_ref,
                args.shared.artifact_dir.as_deref(),
            )?;
            Ok(json!({
                "success": true,
                "task_id": output.task_id,
                "task_type": output.task_type,
                "task_family": output.task_family,
                "status": output.status,
                "normalized_status": output.normalized_status,
                "current_attempt_id": output.current_attempt_id,
                "attempt_count": output.attempt_count,
                "attempts": output.record.attempts,
                "path": output.path,
                "absolute_path": output.absolute_path,
            }))
        }
        TaskSubcommand::List(args) => {
            let workspace_root = resolve_workspace_root(args.shared.workspace)?;
            let task_type_filter = match args.task_type.as_deref() {
                Some(raw) => Some(raw.parse::<TaskType>().map_err(|_| {
                    MediaRuntimeError::InvalidParams(format!("未知任务类型: {raw}"))
                })?),
                None => None,
            };
            let task_family_filter = args
                .task_family
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty());
            let tasks = list_task_outputs(
                &workspace_root,
                args.shared.artifact_dir.as_deref(),
                args.status.as_deref(),
                task_family_filter,
                task_type_filter,
                args.limit,
            )?;
            Ok(json!({
                "success": true,
                "workspace_root": workspace_root.to_string_lossy().to_string(),
                "artifact_root": resolve_artifact_root(&workspace_root, args.shared.artifact_dir.as_deref()).to_string_lossy().to_string(),
                "filters": {
                    "status": args.status,
                    "task_family": task_family_filter,
                    "task_type": task_type_filter.map(|value| value.as_str().to_string()),
                    "limit": args.limit,
                },
                "total": tasks.len(),
                "tasks": tasks,
            }))
        }
        TaskSubcommand::Retry(args) => {
            let workspace_root = resolve_workspace_root(args.shared.workspace)?;
            let output = retry_task_artifact(
                &workspace_root,
                &args.task_ref,
                args.shared.artifact_dir.as_deref(),
            )?;
            Ok(json!(output))
        }
        TaskSubcommand::Cancel(args) => {
            let workspace_root = resolve_workspace_root(args.shared.workspace)?;
            let output = update_task_status(
                &workspace_root,
                &args.task_ref,
                args.shared.artifact_dir.as_deref(),
                "cancelled",
            )?;
            Ok(json!(output))
        }
        TaskSubcommand::Result(args) => {
            let workspace_root = resolve_workspace_root(args.shared.workspace)?;
            let output = load_task_output(
                &workspace_root,
                &args.task_ref,
                args.shared.artifact_dir.as_deref(),
            )?;
            let has_result = output.record.result.is_some();
            let result = output.record.result.clone();
            let record = output.record.clone();
            Ok(json!({
                "success": true,
                "task_id": output.task_id,
                "task_type": output.task_type,
                "task_family": output.task_family,
                "status": output.status,
                "normalized_status": output.normalized_status,
                "current_attempt_id": output.current_attempt_id,
                "attempt_count": output.attempt_count,
                "last_error": output.last_error,
                "progress": output.progress,
                "ui_hints": output.ui_hints,
                "path": output.path,
                "absolute_path": output.absolute_path,
                "artifact_path": output.artifact_path,
                "absolute_artifact_path": output.absolute_artifact_path,
                "has_result": has_result,
                "result": result,
                "record": record,
            }))
        }
    }
}

fn run_task_create_command(command: TaskCreateCommand) -> Result<Value, MediaRuntimeError> {
    match command.command {
        TaskCreateSubcommand::Image(args) => create_image_task(args),
        TaskCreateSubcommand::Cover(args) => create_cover_task(args),
        TaskCreateSubcommand::Video(args) => video::create_video_task(args),
        TaskCreateSubcommand::Transcription(args) => create_transcription_task(args),
        TaskCreateSubcommand::Broadcast(args) => create_broadcast_task(args),
        TaskCreateSubcommand::UrlParse(args) => create_url_parse_task(args),
        TaskCreateSubcommand::Typesetting(args) => create_typesetting_task(args),
        TaskCreateSubcommand::ResourceSearch(args) => create_resource_search_task(args),
    }
}

fn run_media_command(command: MediaCommand) -> Result<Value, MediaRuntimeError> {
    match command.command {
        MediaSubcommand::Image(image) => match image.command {
            ImageSubcommand::Generate(args) => generate_image_task(args),
        },
        MediaSubcommand::Cover(cover) => match cover.command {
            CoverSubcommand::Generate(args) => create_cover_task(args),
        },
        MediaSubcommand::Video(video) => match video.command {
            VideoSubcommand::Generate(args) => video::generate_video_task(args),
        },
        MediaSubcommand::Transcription(transcription) => match transcription.command {
            TranscriptionSubcommand::Generate(args) => create_transcription_task(args),
        },
    }
}

fn run_skill_command(command: SkillCommand) -> Result<Value, MediaRuntimeError> {
    match command.command {
        SkillSubcommand::List => Ok(json!({
            "success": true,
            "total": SKILL_ENTRIES.len(),
            "skills": SKILL_ENTRIES.iter().map(|entry| {
                json!({
                    "name": entry.name,
                    "description": entry.description,
                    "recommended_command": entry.recommended_command,
                    "skill_path": entry.skill_path,
                    "references": entry.references,
                })
            }).collect::<Vec<_>>(),
        })),
        SkillSubcommand::Show(args) => {
            let entry = find_skill_entry(&args.name)
                .or_else(|| {
                    find_task_entry(&args.name).and_then(|task| find_skill_entry(task.skill_name))
                })
                .ok_or_else(|| {
                    MediaRuntimeError::InvalidParams(format!("未知 skill: {}", args.name))
                })?;
            Ok(json!({
                "success": true,
                "skill": {
                    "name": entry.name,
                    "description": entry.description,
                    "recommended_command": entry.recommended_command,
                    "skill_path": entry.skill_path,
                    "references": entry.references,
                }
            }))
        }
    }
}

fn run_doctor_command(args: DoctorArgs) -> Result<Value, MediaRuntimeError> {
    let _ = args.json;
    let workspace_root = resolve_workspace_root(args.workspace)?;
    let artifact_root = resolve_artifact_root(&workspace_root, args.artifact_dir.as_deref());
    let docs_root = workspace_root.join("tools/lime-cli");
    let current_dir = std::env::current_dir()
        .map_err(|error| MediaRuntimeError::Io(format!("读取当前工作目录失败: {error}")))?;
    let current_exe = std::env::current_exe()
        .map_err(|error| MediaRuntimeError::Io(format!("读取当前可执行路径失败: {error}")))?;
    let task_count = list_task_outputs(
        &workspace_root,
        args.artifact_dir.as_deref(),
        None,
        None,
        None,
        None,
    )?
    .len();

    Ok(json!({
        "success": true,
        "cli_version": env!("CARGO_PKG_VERSION"),
        "current_dir": current_dir.to_string_lossy().to_string(),
        "current_exe": current_exe.to_string_lossy().to_string(),
        "workspace_root": workspace_root.to_string_lossy().to_string(),
        "artifact_root": artifact_root.to_string_lossy().to_string(),
        "artifact_root_exists": artifact_root.exists(),
        "docs_root": docs_root.to_string_lossy().to_string(),
        "docs_root_exists": docs_root.exists(),
        "known_task_count": task_count,
        "supported_tasks": TASK_ENTRIES.iter().map(|entry| {
            json!({
                "command_name": entry.command_name,
                "task_type": entry.task_type.as_str(),
                "task_family": entry.task_type.family(),
                "description": entry.description,
                "skill_name": entry.skill_name,
                "docs_dir": entry.docs_dir,
                "example": entry.example,
            })
        }).collect::<Vec<_>>(),
        "supported_skills": SKILL_ENTRIES.iter().map(|entry| entry.name).collect::<Vec<_>>(),
    }))
}

fn create_image_task_artifact(
    workspace_root: &Path,
    args: &ImageGenerateArgs,
) -> Result<lime_media_runtime::MediaTaskOutput, MediaRuntimeError> {
    write_task_artifact(
        workspace_root,
        TaskType::ImageGenerate,
        args.title.clone(),
        build_image_task_payload(args),
        task_write_options(&args.output),
    )
}

fn create_image_task(args: ImageGenerateArgs) -> Result<Value, MediaRuntimeError> {
    generate_image_task(args)
}

fn build_image_task_payload(args: &ImageGenerateArgs) -> Value {
    json!({
        "prompt": args.prompt,
        "mode": args.mode,
        "raw_text": args.raw_text,
        "model": args.model,
        "style": args.style,
        "size": args.size,
        "aspect_ratio": args.aspect_ratio,
        "count": args.count,
        "usage": args.usage,
        "provider_id": args.provider_id,
        "session_id": args.session_id,
        "project_id": args.project_id,
        "content_id": args.content_id,
        "entry_source": args.entry_source,
        "requested_target": args.requested_target,
        "slot_id": args.slot_id,
        "anchor_hint": args.anchor_hint,
        "anchor_section_title": args.anchor_section_title,
        "anchor_text": args.anchor_text,
        "target_output_id": args.target_output_id,
        "target_output_ref_id": args.target_output_ref_id,
        "reference_images": args.reference_images,
    })
}

fn read_non_empty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn read_env_port(name: &str) -> Result<Option<u16>, String> {
    let Some(raw) = read_non_empty_env(name) else {
        return Ok(None);
    };
    raw.parse::<u16>()
        .map(Some)
        .map_err(|error| format!("{name} 不是合法端口: {error}"))
}

fn resolve_cli_image_generation_runner_config() -> Result<ImageGenerationRunnerConfig, String> {
    let endpoint_override = read_non_empty_env("LIME_MEDIA_IMAGE_ENDPOINT");
    let api_key_override = read_non_empty_env("LIME_MEDIA_IMAGE_API_KEY")
        .or_else(|| read_non_empty_env("LIME_SERVER_API_KEY"));
    let host_override = read_non_empty_env("LIME_SERVER_HOST");
    let port_override = read_env_port("LIME_SERVER_PORT")?;

    let (loaded_config, config_load_error) = match load_config() {
        Ok(config) => (Some(config), None),
        Err(error) => (None, Some(error.to_string())),
    };

    let host = host_override
        .or_else(|| {
            loaded_config
                .as_ref()
                .map(|config| config.server.host.clone())
        })
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let port = port_override
        .or_else(|| loaded_config.as_ref().map(|config| config.server.port))
        .unwrap_or(9000);
    let api_key = api_key_override
        .or_else(|| {
            loaded_config
                .as_ref()
                .map(|config| config.server.api_key.trim().to_string())
                .filter(|value| !value.is_empty())
        })
        .ok_or_else(|| match config_load_error {
            Some(error) => format!("Lime 本地图片服务未配置 API Key，且加载本地配置失败: {error}"),
            None => "Lime 本地图片服务未配置 API Key".to_string(),
        })?;

    Ok(ImageGenerationRunnerConfig {
        endpoint: endpoint_override.unwrap_or_else(|| build_image_generation_endpoint(&host, port)),
        api_key,
        request_body_format: Default::default(),
    })
}

fn build_image_task_progress(
    phase: &str,
    message: impl Into<String>,
    percent: Option<u32>,
) -> TaskProgress {
    TaskProgress {
        phase: Some(phase.to_string()),
        percent,
        message: Some(message.into()),
        preview_slots: Vec::new(),
    }
}

fn build_image_task_error(
    code: &str,
    message: impl Into<String>,
    retryable: bool,
    stage: &str,
) -> TaskErrorRecord {
    TaskErrorRecord {
        code: code.to_string(),
        message: message.into(),
        retryable,
        stage: Some(stage.to_string()),
        provider_code: None,
        occurred_at: None,
    }
}

fn mark_cli_image_task_failed(
    workspace_root: &Path,
    task_id: &str,
    message: impl Into<String>,
) -> Result<Value, MediaRuntimeError> {
    let task_error =
        build_image_task_error("image_worker_unavailable", message, false, "bootstrap");
    let output = patch_task_artifact(
        workspace_root,
        task_id,
        None,
        TaskArtifactPatch {
            status: Some("failed".to_string()),
            last_error: Some(Some(task_error.clone())),
            progress: Some(build_image_task_progress(
                "failed",
                task_error.message.clone(),
                None,
            )),
            current_attempt_worker_id: Some(Some(IMAGE_TASK_RUNNER_WORKER_ID.to_string())),
            ..TaskArtifactPatch::default()
        },
    )?;
    Ok(json!(output))
}

fn generate_image_task(args: ImageGenerateArgs) -> Result<Value, MediaRuntimeError> {
    let workspace_root = resolve_workspace_root(args.output.workspace.clone())?;
    let created = create_image_task_artifact(&workspace_root, &args)?;
    let runner_config = match resolve_cli_image_generation_runner_config() {
        Ok(config) => config,
        Err(error_message) => {
            return mark_cli_image_task_failed(&workspace_root, &created.task_id, error_message);
        }
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| MediaRuntimeError::Io(format!("初始化图片任务运行时失败: {error}")))?;
    let output = runtime.block_on(execute_image_task_runtime(
        &workspace_root,
        &created.task_id,
        &runner_config,
    ))?;
    Ok(json!(output))
}

fn create_cover_task(args: CoverGenerateArgs) -> Result<Value, MediaRuntimeError> {
    let workspace_root = resolve_workspace_root(args.output.workspace.clone())?;
    let output = write_task_artifact(
        &workspace_root,
        TaskType::CoverGenerate,
        args.title,
        json!({
            "prompt": args.prompt,
            "raw_text": args.raw_text,
            "model": args.model,
            "style": args.style,
            "platform": args.platform,
            "size": args.size,
            "imageUrl": args.image_url,
            "referenceImageUrl": args.reference_image_url,
            "usage": args.usage.or(Some("cover".to_string())),
            "session_id": args.session_id,
            "project_id": args.project_id,
            "content_id": args.content_id,
            "entry_source": args.entry_source,
        }),
        task_write_options(&args.output),
    )?;
    Ok(json!(output))
}

fn create_transcription_task(args: TranscriptionGenerateArgs) -> Result<Value, MediaRuntimeError> {
    let workspace_root = resolve_workspace_root(args.output.workspace.clone())?;
    let source_url = args
        .source_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let source_path = args
        .source_path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());

    if source_url.is_none() && source_path.is_none() {
        return Err(MediaRuntimeError::InvalidParams(
            "source_url 或 source_path 至少需要提供一个".to_string(),
        ));
    }
    let transcript = json!({
        "kind": "transcript",
        "status": "pending",
        "source_url": source_url,
        "source_path": source_path,
        "language": args.language.as_deref(),
        "output_format": args.output_format.as_deref(),
        "speaker_labels": args.speaker_labels,
        "timestamps": args.timestamps,
        "provider_id": args.provider_id.as_deref(),
        "model": args.model.as_deref(),
        "modality_contract_key": AUDIO_TRANSCRIPTION_CONTRACT_KEY,
        "modality": AUDIO_TRANSCRIPTION_MODALITY,
        "routing_slot": AUDIO_TRANSCRIPTION_ROUTING_SLOT,
    });

    let output = write_task_artifact(
        &workspace_root,
        TaskType::TranscriptionGenerate,
        args.title,
        json!({
            "prompt": args.prompt,
            "raw_text": args.raw_text,
            "source_url": source_url,
            "source_path": source_path,
            "language": args.language,
            "output_format": args.output_format,
            "speaker_labels": args.speaker_labels,
            "timestamps": args.timestamps,
            "provider_id": args.provider_id,
            "model": args.model,
            "session_id": args.session_id,
            "project_id": args.project_id,
            "content_id": args.content_id,
            "entry_source": args.entry_source,
            "modality_contract_key": AUDIO_TRANSCRIPTION_CONTRACT_KEY,
            "modality": AUDIO_TRANSCRIPTION_MODALITY,
            "required_capabilities": AUDIO_TRANSCRIPTION_REQUIRED_CAPABILITIES,
            "routing_slot": AUDIO_TRANSCRIPTION_ROUTING_SLOT,
            "runtime_contract": {
                "contract_key": AUDIO_TRANSCRIPTION_CONTRACT_KEY,
                "modality": AUDIO_TRANSCRIPTION_MODALITY,
                "required_capabilities": AUDIO_TRANSCRIPTION_REQUIRED_CAPABILITIES,
                "routing_slot": AUDIO_TRANSCRIPTION_ROUTING_SLOT,
                "executor_binding": {
                    "executor_kind": "app_server",
                    "binding_key": "mediaTaskArtifact/transcription/create"
                },
                "truth_source": ["transcript_artifact", "runtime_timeline_event"],
                "artifact_kinds": ["transcript"],
                "viewer_surface": ["transcript_viewer", "document_viewer"],
                "owner_surface": "app_server_runtime"
            },
            "requested_target": "transcript",
            "transcript": transcript,
        }),
        task_write_options(&args.output),
    )?;
    Ok(json!(output))
}

fn create_broadcast_task(args: BroadcastGenerateArgs) -> Result<Value, MediaRuntimeError> {
    let workspace_root = resolve_workspace_root(args.output.workspace.clone())?;
    if args.content.trim().is_empty() {
        return Err(MediaRuntimeError::InvalidParams(
            "content 不能为空字符串".to_string(),
        ));
    }
    let output = write_task_artifact(
        &workspace_root,
        TaskType::BroadcastGenerate,
        args.title,
        json!({
            "content": args.content,
            "audience": args.audience,
            "tone": args.tone,
            "durationHintMinutes": args.duration_hint_minutes,
        }),
        task_write_options(&args.output),
    )?;
    Ok(json!(output))
}

fn create_url_parse_task(args: UrlParseArgs) -> Result<Value, MediaRuntimeError> {
    let workspace_root = resolve_workspace_root(args.output.workspace.clone())?;
    if args.url.trim().is_empty() {
        return Err(MediaRuntimeError::InvalidParams(
            "url 不能为空字符串".to_string(),
        ));
    }
    let summary = args
        .summary
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let extract_status = if summary.is_none() && args.extract_status.trim() == "ready" {
        "pending_extract".to_string()
    } else {
        args.extract_status
    };
    let output = write_task_artifact(
        &workspace_root,
        TaskType::UrlParse,
        args.title,
        json!({
            "url": args.url,
            "summary": summary,
            "keyPoints": args.key_points,
            "extractStatus": extract_status,
            "prompt": args.prompt,
            "raw_text": args.raw_text,
            "extractGoal": args.extract_goal,
            "session_id": args.session_id,
            "project_id": args.project_id,
            "content_id": args.content_id,
            "entry_source": args.entry_source,
        }),
        task_write_options(&args.output),
    )?;
    Ok(json!(output))
}

fn create_typesetting_task(args: TypesettingArgs) -> Result<Value, MediaRuntimeError> {
    let workspace_root = resolve_workspace_root(args.output.workspace.clone())?;
    if args.content.trim().is_empty() {
        return Err(MediaRuntimeError::InvalidParams(
            "content 不能为空字符串".to_string(),
        ));
    }
    if args.target_platform.trim().is_empty() {
        return Err(MediaRuntimeError::InvalidParams(
            "targetPlatform 不能为空字符串".to_string(),
        ));
    }
    let output = write_task_artifact(
        &workspace_root,
        TaskType::Typesetting,
        args.title,
        json!({
            "targetPlatform": args.target_platform,
            "rules": args.rules,
            "content": args.content,
        }),
        task_write_options(&args.output),
    )?;
    Ok(json!(output))
}

fn create_resource_search_task(args: ResourceSearchArgs) -> Result<Value, MediaRuntimeError> {
    let workspace_root = resolve_workspace_root(args.output.workspace.clone())?;
    if args.resource_type.trim().is_empty() {
        return Err(MediaRuntimeError::InvalidParams(
            "resourceType 不能为空字符串".to_string(),
        ));
    }
    if args.query.trim().is_empty() {
        return Err(MediaRuntimeError::InvalidParams(
            "query 不能为空字符串".to_string(),
        ));
    }
    if args.usage.trim().is_empty() {
        return Err(MediaRuntimeError::InvalidParams(
            "usage 不能为空字符串".to_string(),
        ));
    }
    let output = write_task_artifact(
        &workspace_root,
        TaskType::ModalResourceSearch,
        args.title,
        json!({
            "resourceType": args.resource_type,
            "query": args.query,
            "usage": args.usage,
            "count": args.count,
            "constraints": args.constraints,
        }),
        task_write_options(&args.output),
    )?;
    Ok(json!(output))
}

fn task_write_options(args: &SharedTaskWriteArgs) -> TaskWriteOptions<'_> {
    let _ = args.json;
    TaskWriteOptions {
        status: None,
        output_path: args.output.as_deref(),
        artifact_dir: args.artifact_dir.as_deref(),
        idempotency_key: args.idempotency_key.as_deref(),
        relationships: TaskRelationships::default(),
    }
}

fn resolve_workspace_root(
    explicit_workspace: Option<PathBuf>,
) -> Result<PathBuf, MediaRuntimeError> {
    if let Some(workspace) = explicit_workspace {
        return Ok(workspace);
    }

    std::env::current_dir()
        .map_err(|error| MediaRuntimeError::Io(format!("读取当前工作目录失败: {error}")))
}

fn resolve_artifact_root(workspace_root: &Path, artifact_dir: Option<&str>) -> PathBuf {
    match artifact_dir {
        Some(raw) if !raw.trim().is_empty() => workspace_root.join(raw),
        _ => workspace_root.join(DEFAULT_ARTIFACT_ROOT),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_workspace_root_defaults_to_current_dir() {
        let root = resolve_workspace_root(None).expect("resolve current dir");
        assert!(root.is_dir());
    }

    #[test]
    fn run_broadcast_create_supports_idempotency_key() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");

        let first = run(Cli {
            command: Command::Task(TaskCommand {
                command: TaskSubcommand::Create(TaskCreateCommand {
                    command: TaskCreateSubcommand::Broadcast(BroadcastGenerateArgs {
                        content: "原文".to_string(),
                        title: Some("播客".to_string()),
                        audience: Some("开发者".to_string()),
                        tone: Some("理性".to_string()),
                        duration_hint_minutes: Some(8),
                        output: SharedTaskWriteArgs {
                            workspace: Some(temp_dir.path().to_path_buf()),
                            output: None,
                            artifact_dir: None,
                            idempotency_key: Some("broadcast-1".to_string()),
                            json: true,
                        },
                    }),
                }),
            }),
        })
        .expect("run first");

        let second = run(Cli {
            command: Command::Task(TaskCommand {
                command: TaskSubcommand::Create(TaskCreateCommand {
                    command: TaskCreateSubcommand::Broadcast(BroadcastGenerateArgs {
                        content: "原文".to_string(),
                        title: Some("播客".to_string()),
                        audience: Some("开发者".to_string()),
                        tone: Some("理性".to_string()),
                        duration_hint_minutes: Some(8),
                        output: SharedTaskWriteArgs {
                            workspace: Some(temp_dir.path().to_path_buf()),
                            output: None,
                            artifact_dir: None,
                            idempotency_key: Some("broadcast-1".to_string()),
                            json: true,
                        },
                    }),
                }),
            }),
        })
        .expect("run second");

        assert_eq!(first["task_id"], second["task_id"]);
        assert_eq!(second["reused_existing"], json!(true));
    }

    #[test]
    fn skill_show_accepts_task_command_name() {
        let output = run(Cli {
            command: Command::Skill(SkillCommand {
                command: SkillSubcommand::Show(SkillShowArgs {
                    name: "broadcast".to_string(),
                }),
            }),
        })
        .expect("show skill");

        assert_eq!(output["skill"]["name"], "broadcast_generate");
    }

    #[test]
    fn skill_show_image_generate_prefers_media_command() {
        let output = run(Cli {
            command: Command::Skill(SkillCommand {
                command: SkillSubcommand::Show(SkillShowArgs {
                    name: "image_generate".to_string(),
                }),
            }),
        })
        .expect("show image skill");

        assert_eq!(
            output["skill"]["recommended_command"],
            "lime media image generate --prompt \"...\""
        );
    }

    #[test]
    fn skill_show_video_generate_prefers_media_command() {
        let output = run(Cli {
            command: Command::Skill(SkillCommand {
                command: SkillSubcommand::Show(SkillShowArgs {
                    name: "video_generate".to_string(),
                }),
            }),
        })
        .expect("show video skill");

        assert_eq!(
            output["skill"]["recommended_command"],
            "lime media video generate --prompt \"...\" --aspect-ratio 9:16"
        );
    }

    #[test]
    fn skill_show_webpage_generate_returns_builtin_skill() {
        let output = run(Cli {
            command: Command::Skill(SkillCommand {
                command: SkillSubcommand::Show(SkillShowArgs {
                    name: "webpage_generate".to_string(),
                }),
            }),
        })
        .expect("show webpage skill");

        assert_eq!(output["skill"]["name"], "webpage_generate");
        assert_eq!(
            output["skill"]["skill_path"],
            "lime-rs/resources/default-skills/webpage_generate/SKILL.md"
        );
    }

    #[test]
    fn skill_show_presentation_generate_returns_builtin_skill() {
        let output = run(Cli {
            command: Command::Skill(SkillCommand {
                command: SkillSubcommand::Show(SkillShowArgs {
                    name: "presentation_generate".to_string(),
                }),
            }),
        })
        .expect("show presentation skill");

        assert_eq!(output["skill"]["name"], "presentation_generate");
        assert_eq!(
            output["skill"]["skill_path"],
            "lime-rs/resources/default-skills/presentation_generate/SKILL.md"
        );
    }

    #[test]
    fn skill_show_form_generate_returns_builtin_skill() {
        let output = run(Cli {
            command: Command::Skill(SkillCommand {
                command: SkillSubcommand::Show(SkillShowArgs {
                    name: "form_generate".to_string(),
                }),
            }),
        })
        .expect("show form skill");

        assert_eq!(output["skill"]["name"], "form_generate");
        assert_eq!(
            output["skill"]["skill_path"],
            "lime-rs/resources/default-skills/form_generate/SKILL.md"
        );
    }

    #[test]
    fn create_image_task_preserves_extended_context_fields() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let output = json!(create_image_task_artifact(
            temp_dir.path(),
            &ImageGenerateArgs {
                prompt: "城市夜景".to_string(),
                title: Some("夜景修图".to_string()),
                mode: Some("edit".to_string()),
                raw_text: Some("@配图 编辑 #img-2 去掉角标".to_string()),
                model: Some("fal-ai/nano-banana-pro".to_string()),
                style: Some("写实".to_string()),
                size: Some("1024x1024".to_string()),
                aspect_ratio: Some("1:1".to_string()),
                count: Some(1),
                usage: Some("claw-image-workbench".to_string()),
                provider_id: Some("fal".to_string()),
                session_id: Some("session-1".to_string()),
                project_id: Some("project-1".to_string()),
                content_id: Some("content-1".to_string()),
                entry_source: Some("at_image_command".to_string()),
                requested_target: Some("generate".to_string()),
                slot_id: Some("slot-1".to_string()),
                anchor_hint: Some("section_end".to_string()),
                anchor_section_title: Some("核心观点".to_string()),
                anchor_text: Some("这里是核心观点段落。".to_string()),
                target_output_id: Some("task-image-1:output:1".to_string()),
                target_output_ref_id: Some("img-2".to_string()),
                reference_images: vec![
                    "https://example.com/image-2.png".to_string(),
                    "/tmp/input-1.png".to_string(),
                ],
                output: SharedTaskWriteArgs {
                    workspace: Some(temp_dir.path().to_path_buf()),
                    output: None,
                    artifact_dir: None,
                    idempotency_key: None,
                    json: true,
                },
            },
        )
        .expect("create image"));

        assert_eq!(output["record"]["payload"]["mode"], "edit");
        assert_eq!(output["record"]["payload"]["target_output_ref_id"], "img-2");
        assert_eq!(
            output["record"]["payload"]["reference_images"],
            json!(["https://example.com/image-2.png", "/tmp/input-1.png"])
        );
    }

    #[test]
    fn create_transcription_task_preserves_source_and_format_fields() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let output = create_transcription_task(TranscriptionGenerateArgs {
            prompt: Some("生成逐字稿".to_string()),
            title: Some("会议转写".to_string()),
            raw_text: Some("@转写 /tmp/interview.wav 生成逐字稿".to_string()),
            source_url: None,
            source_path: Some("/tmp/interview.wav".to_string()),
            language: Some("zh".to_string()),
            output_format: Some("srt".to_string()),
            speaker_labels: Some(true),
            timestamps: Some(true),
            provider_id: None,
            model: None,
            session_id: Some("session-1".to_string()),
            project_id: Some("project-1".to_string()),
            content_id: Some("content-1".to_string()),
            entry_source: Some("at_transcription_command".to_string()),
            output: SharedTaskWriteArgs {
                workspace: Some(temp_dir.path().to_path_buf()),
                output: None,
                artifact_dir: None,
                idempotency_key: None,
                json: true,
            },
        })
        .expect("create transcription");

        assert_eq!(
            output["record"]["payload"]["source_path"],
            "/tmp/interview.wav"
        );
        assert_eq!(output["record"]["payload"]["output_format"], "srt");
        assert_eq!(output["record"]["payload"]["speaker_labels"], true);
        assert_eq!(output["record"]["payload"]["timestamps"], true);
        assert_eq!(
            output["record"]["payload"]["modality_contract_key"],
            AUDIO_TRANSCRIPTION_CONTRACT_KEY
        );
        assert_eq!(
            output["record"]["payload"]["routing_slot"],
            AUDIO_TRANSCRIPTION_ROUTING_SLOT
        );
        assert_eq!(
            output["record"]["payload"]["runtime_contract"]["contract_key"],
            AUDIO_TRANSCRIPTION_CONTRACT_KEY
        );
        assert_eq!(
            output["record"]["payload"]["transcript"]["kind"],
            "transcript"
        );
        assert_eq!(
            output["record"]["payload"]["transcript"]["status"],
            "pending"
        );
        assert_eq!(
            output["record"]["payload"]["transcript"]["source_path"],
            "/tmp/interview.wav"
        );
    }

    #[test]
    fn task_list_returns_created_items() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let _ = create_image_task_artifact(
            temp_dir.path(),
            &ImageGenerateArgs {
                prompt: "城市".to_string(),
                title: None,
                mode: None,
                raw_text: None,
                model: None,
                style: None,
                size: None,
                aspect_ratio: None,
                count: None,
                usage: None,
                provider_id: None,
                session_id: None,
                project_id: None,
                content_id: None,
                entry_source: None,
                requested_target: None,
                slot_id: None,
                anchor_hint: None,
                anchor_section_title: None,
                anchor_text: None,
                target_output_id: None,
                target_output_ref_id: None,
                reference_images: Vec::new(),
                output: SharedTaskWriteArgs {
                    workspace: Some(temp_dir.path().to_path_buf()),
                    output: None,
                    artifact_dir: None,
                    idempotency_key: None,
                    json: true,
                },
            },
        )
        .expect("create image");

        let output = run(Cli {
            command: Command::Task(TaskCommand {
                command: TaskSubcommand::List(TaskListArgs {
                    shared: SharedTaskReadArgs {
                        workspace: Some(temp_dir.path().to_path_buf()),
                        artifact_dir: None,
                        json: true,
                    },
                    status: None,
                    task_family: None,
                    task_type: Some("image".to_string()),
                    limit: Some(10),
                }),
            }),
        })
        .expect("list tasks");

        assert_eq!(output["total"], 1);
        assert_eq!(output["tasks"][0]["task_type"], "image_generate");
    }

    #[test]
    fn task_list_supports_family_filter() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let _ = create_image_task_artifact(
            temp_dir.path(),
            &ImageGenerateArgs {
                prompt: "城市".to_string(),
                title: None,
                mode: None,
                raw_text: None,
                model: None,
                style: None,
                size: None,
                aspect_ratio: None,
                count: None,
                usage: None,
                provider_id: None,
                session_id: None,
                project_id: None,
                content_id: None,
                entry_source: None,
                requested_target: None,
                slot_id: None,
                anchor_hint: None,
                anchor_section_title: None,
                anchor_text: None,
                target_output_id: None,
                target_output_ref_id: None,
                reference_images: Vec::new(),
                output: SharedTaskWriteArgs {
                    workspace: Some(temp_dir.path().to_path_buf()),
                    output: None,
                    artifact_dir: None,
                    idempotency_key: None,
                    json: true,
                },
            },
        )
        .expect("create image");
        let _ = create_url_parse_task(UrlParseArgs {
            url: "https://example.com".to_string(),
            title: None,
            summary: Some("摘要".to_string()),
            key_points: Vec::new(),
            extract_status: "ready".to_string(),
            prompt: None,
            raw_text: None,
            extract_goal: None,
            session_id: None,
            project_id: None,
            content_id: None,
            entry_source: None,
            output: SharedTaskWriteArgs {
                workspace: Some(temp_dir.path().to_path_buf()),
                output: None,
                artifact_dir: None,
                idempotency_key: None,
                json: true,
            },
        })
        .expect("create url parse");

        let output = run(Cli {
            command: Command::Task(TaskCommand {
                command: TaskSubcommand::List(TaskListArgs {
                    shared: SharedTaskReadArgs {
                        workspace: Some(temp_dir.path().to_path_buf()),
                        artifact_dir: None,
                        json: true,
                    },
                    status: None,
                    task_family: Some("image".to_string()),
                    task_type: None,
                    limit: Some(10),
                }),
            }),
        })
        .expect("list image family");

        assert_eq!(output["total"], 1);
        assert_eq!(output["tasks"][0]["task_family"], "image");
    }

    #[test]
    fn create_url_parse_task_without_summary_should_fallback_to_pending_extract() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let output = create_url_parse_task(UrlParseArgs {
            url: "https://example.com/post".to_string(),
            title: None,
            summary: None,
            key_points: Vec::new(),
            extract_status: "ready".to_string(),
            prompt: Some("提取重点".to_string()),
            raw_text: Some("@链接解析 https://example.com/post 提取重点".to_string()),
            extract_goal: Some("key_points".to_string()),
            session_id: Some("session-1".to_string()),
            project_id: Some("project-1".to_string()),
            content_id: Some("content-1".to_string()),
            entry_source: Some("at_url_parse_command".to_string()),
            output: SharedTaskWriteArgs {
                workspace: Some(temp_dir.path().to_path_buf()),
                output: None,
                artifact_dir: None,
                idempotency_key: None,
                json: true,
            },
        })
        .expect("create url parse task");

        assert_eq!(output["status"], "pending_submit");
        assert_eq!(
            output["record"]["payload"]["extractStatus"],
            "pending_extract"
        );
        assert_eq!(
            output["record"]["payload"]["entry_source"],
            "at_url_parse_command"
        );
    }

    #[test]
    fn create_typesetting_task_preserves_platform_and_rules() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let output = create_typesetting_task(TypesettingArgs {
            content: "这是一段待排版正文".to_string(),
            target_platform: "小红书".to_string(),
            title: Some("小红书排版".to_string()),
            rules: vec!["语气:轻快".to_string(), "段落:短句".to_string()],
            output: SharedTaskWriteArgs {
                workspace: Some(temp_dir.path().to_path_buf()),
                output: None,
                artifact_dir: None,
                idempotency_key: None,
                json: true,
            },
        })
        .expect("create typesetting task");

        assert_eq!(output["record"]["payload"]["targetPlatform"], "小红书");
        assert_eq!(output["record"]["payload"]["content"], "这是一段待排版正文");
        assert_eq!(
            output["record"]["payload"]["rules"],
            json!(["语气:轻快", "段落:短句"])
        );
    }

    #[test]
    fn task_attempts_returns_attempt_history() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let failed = write_task_artifact(
            temp_dir.path(),
            TaskType::ModalResourceSearch,
            Some("素材检索".to_string()),
            json!({ "query": "城市夜景" }),
            TaskWriteOptions {
                status: Some("failed".to_string()),
                ..TaskWriteOptions::default()
            },
        )
        .expect("write failed task");
        let _ = retry_task_artifact(temp_dir.path(), &failed.task_id, None).expect("retry task");

        let output = run(Cli {
            command: Command::Task(TaskCommand {
                command: TaskSubcommand::Attempts(TaskLookupArgs {
                    task_ref: failed.task_id.clone(),
                    shared: SharedTaskReadArgs {
                        workspace: Some(temp_dir.path().to_path_buf()),
                        artifact_dir: None,
                        json: true,
                    },
                }),
            }),
        })
        .expect("get attempts");

        assert_eq!(output["attempt_count"], 2);
        assert_eq!(output["attempts"].as_array().map(Vec::len), Some(2));
        assert_eq!(output["task_family"], "resource");
    }
}
