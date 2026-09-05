//! Local file-link parsing and display adapted from Codex TUI.

use url::Url;

pub(super) fn is_local_path_like_link(destination: &str) -> bool {
    destination.starts_with("file://")
        || destination.starts_with('/')
        || destination.starts_with("~/")
        || destination.starts_with("./")
        || destination.starts_with("../")
        || destination.starts_with("\\\\")
        || matches!(
            destination.as_bytes(),
            [drive, b':', separator, ..]
                if drive.is_ascii_alphabetic() && matches!(separator, b'/' | b'\\')
        )
}

pub(super) fn render_local_link_target(destination: &str) -> Option<String> {
    let (path, suffix) = parse_local_link_target(destination)?;
    let mut rendered = normalize_path(&path);
    if let Some(suffix) = suffix {
        rendered.push_str(&suffix);
    }
    Some(rendered)
}

pub(super) fn should_render_local_link_label(label: &str, destination: &str) -> bool {
    let label = label.trim();
    if label.is_empty() {
        return false;
    }
    let Some(label_path) = comparable_path(label) else {
        return true;
    };
    let Some(target_path) = comparable_path(destination) else {
        return true;
    };
    let label_path = trim_trailing_separator(label_path.trim_start_matches("./"));
    let target_path = trim_trailing_separator(target_path.trim_start_matches("./"));
    let boundary_suffix = |path: &str, suffix: &str| {
        !suffix.is_empty()
            && path
                .strip_suffix(suffix)
                .is_some_and(|prefix| prefix.is_empty() || prefix.ends_with('/'))
    };

    !(boundary_suffix(target_path, label_path)
        || (is_absolute_path(label_path) && boundary_suffix(label_path, target_path)))
}

fn comparable_path(text: &str) -> Option<String> {
    let (path, _) = parse_local_link_target(text)?;
    Some(normalize_path(&path).to_lowercase())
}

fn parse_local_link_target(destination: &str) -> Option<(String, Option<String>)> {
    if destination.starts_with("file://") {
        let url = Url::parse(destination).ok()?;
        let path = file_url_path(&url)?;
        let suffix = url.fragment().and_then(normalize_hash_location);
        return Some((path, suffix));
    }

    let mut path = destination;
    let mut suffix = None;
    if let Some((candidate, fragment)) = destination.rsplit_once('#') {
        if let Some(location) = normalize_hash_location(fragment) {
            path = candidate;
            suffix = Some(location);
        }
    }
    if suffix.is_none() {
        if let Some((candidate, location)) = split_colon_location(path) {
            path = candidate;
            suffix = Some(location.to_string());
        }
    }
    let decoded = urlencoding::decode(path).unwrap_or_else(|_| path.into());
    Some((normalize_path(&decoded), suffix))
}

fn file_url_path(url: &Url) -> Option<String> {
    if let Ok(path) = url.to_file_path() {
        return Some(normalize_path(&path.to_string_lossy()));
    }

    let mut path = urlencoding::decode(url.path())
        .unwrap_or_else(|_| url.path().into())
        .into_owned();
    if let Some(host) = url.host_str() {
        if !host.is_empty() && host != "localhost" {
            path = format!("//{host}{path}");
        } else if matches!(
            path.as_bytes(),
            [b'/', drive, b':', b'/', ..] if drive.is_ascii_alphabetic()
        ) {
            path.remove(0);
        }
    }
    Some(normalize_path(&path))
}

fn normalize_hash_location(fragment: &str) -> Option<String> {
    let rest = fragment.strip_prefix('L')?;
    let (start, end) = rest
        .split_once("-L")
        .map_or((rest, None), |(start, end)| (start, Some(end)));
    let start = normalize_line_column(start)?;
    match end {
        Some(end) => Some(format!("{start}-{}", normalize_line_column(end)?)),
        None => Some(start),
    }
}

fn normalize_line_column(value: &str) -> Option<String> {
    let (line, column) = value
        .split_once('C')
        .map_or((value, None), |(line, column)| (line, Some(column)));
    if line.is_empty() || !line.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    match column {
        Some(column) if !column.is_empty() && column.bytes().all(|byte| byte.is_ascii_digit()) => {
            Some(format!(":{line}:{column}"))
        }
        Some(_) => None,
        None => Some(format!(":{line}")),
    }
}

fn split_colon_location(path: &str) -> Option<(&str, &str)> {
    let bytes = path.as_bytes();
    let mut start = bytes.len();
    let mut colon_count = 0usize;
    while start > 0 {
        let byte = bytes[start - 1];
        if byte.is_ascii_digit() {
            start -= 1;
            continue;
        }
        if byte == b':' && colon_count < 2 {
            colon_count += 1;
            start -= 1;
            continue;
        }
        break;
    }
    let location = &path[start..];
    let valid = matches!(colon_count, 1 | 2)
        && location
            .split(':')
            .skip(1)
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()));
    valid.then_some((&path[..start], location))
}

fn normalize_path(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("\\\\") {
        format!("//{}", rest.replace('\\', "/").trim_start_matches('/'))
    } else {
        path.replace('\\', "/")
    }
}

fn is_absolute_path(path: &str) -> bool {
    path.starts_with('/')
        || path.starts_with("//")
        || matches!(
            path.as_bytes(),
            [drive, b':', b'/', ..] if drive.is_ascii_alphabetic()
        )
}

fn trim_trailing_separator(path: &str) -> &str {
    if path == "/" || path == "//" {
        return path;
    }
    if matches!(path.as_bytes(), [drive, b':', b'/'] if drive.is_ascii_alphabetic()) {
        return path;
    }
    path.trim_end_matches('/')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_unix_windows_unc_and_file_url_paths() {
        for path in [
            "file:///tmp/a.rs",
            "/tmp/a.rs",
            "~/a.rs",
            "./a.rs",
            "../a.rs",
            r"C:\repo\a.rs",
            r"\\server\share\a.rs",
        ] {
            assert!(is_local_path_like_link(path), "{path}");
        }
        assert!(!is_local_path_like_link("https://example.com/a.rs"));
    }

    #[test]
    fn renders_encoded_paths_and_location_suffixes_without_filesystem_access() {
        assert_eq!(
            render_local_link_target("file:///tmp/My%20File.rs#L12C3"),
            Some("/tmp/My File.rs:12:3".to_string())
        );
        assert_eq!(
            render_local_link_target(r"C:\Repo\src\lib.rs:8"),
            Some("C:/Repo/src/lib.rs:8".to_string())
        );
        assert_eq!(
            render_local_link_target(r"\\server\share\My%20File.rs"),
            Some("//server/share/My File.rs".to_string())
        );
    }

    #[test]
    fn matching_path_labels_collapse_but_descriptive_labels_remain() {
        assert!(!should_render_local_link_label(
            "src/lib.rs",
            "./src/lib.rs"
        ));
        assert!(!should_render_local_link_label(
            "My File.rs",
            "file:///tmp/My%20File.rs"
        ));
        assert!(should_render_local_link_label(
            "open generated source",
            "./src/lib.rs"
        ));
        assert!(should_render_local_link_label(
            "other/src/lib.rs",
            "./src/lib.rs"
        ));
    }

    #[test]
    fn invalid_percent_encoding_stays_visible() {
        assert_eq!(
            render_local_link_target("/tmp/bad%FF.rs"),
            Some("/tmp/bad%FF.rs".to_string())
        );
    }
}
