use std::fs::{read_dir, read_to_string};
use std::path::{Path, PathBuf};

const ALLOWED_EVENT_CONVERTER_FILES: &[&str] = &[
    "crates/agent/src/agent_runtime_projection.rs",
    "crates/agent/src/event_converter.rs",
];

const ALLOWED_RUNTIME_CONVERTER_FILES: &[&str] = &[
    "crates/agent/src/agent_runtime_projection.rs",
    "crates/agent/src/event_converter.rs",
];

fn collect_rust_files(dir: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };

        if file_type.is_dir() {
            if path.file_name().and_then(|name| name.to_str()) == Some("tests") {
                continue;
            }
            collect_rust_files(&path, files);
            continue;
        }

        if path.extension().and_then(|value| value.to_str()) != Some("rs") {
            continue;
        }

        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".test.rs"))
        {
            continue;
        }

        files.push(path);
    }
}

#[test]
fn production_rust_code_should_not_bypass_protocol_projection() {
    let rust_workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut files = Vec::new();

    collect_rust_files(&rust_workspace_root.join("src"), &mut files);
    collect_rust_files(&rust_workspace_root.join("crates"), &mut files);

    let mut offenders = Vec::new();
    for file_path in files {
        let relative_path = file_path
            .strip_prefix(&rust_workspace_root)
            .expect("相对路径转换失败");
        let relative_path_string = relative_path.to_string_lossy().replace('\\', "/");
        let Ok(content) = read_to_string(&file_path) else {
            continue;
        };
        let event_converter_allowed = ALLOWED_EVENT_CONVERTER_FILES
            .iter()
            .any(|allowed| *allowed == relative_path_string);
        let runtime_converter_allowed = ALLOWED_RUNTIME_CONVERTER_FILES
            .iter()
            .any(|allowed| *allowed == relative_path_string);
        if !event_converter_allowed
            && (content.contains("event_converter::") || content.contains("convert_agent_event("))
        {
            offenders.push(relative_path_string.clone());
            continue;
        }
        if !runtime_converter_allowed
            && (content.contains("convert_turn_runtime(")
                || content.contains("convert_item_runtime("))
        {
            offenders.push(relative_path_string);
        }
    }

    assert_eq!(offenders, Vec::<String>::new());
}
