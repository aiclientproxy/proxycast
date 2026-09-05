use std::io::Cursor;
use std::path::PathBuf;

use image::{DynamicImage, ImageFormat, RgbaImage};
use tempfile::{Builder, NamedTempFile};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PasteImageError {
    ClipboardUnavailable(String),
    NoImage(String),
    EncodeFailed(String),
    Io(String),
}

/// Normalize pasted text for a single-line search query.
pub(crate) fn normalize_pasted_search_query(pasted: &str) -> Option<String> {
    let normalized = pasted.split_whitespace().collect::<Vec<_>>().join(" ");
    (!normalized.is_empty()).then_some(normalized)
}

impl std::fmt::Display for PasteImageError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ClipboardUnavailable(message) => {
                write!(formatter, "clipboard unavailable: {message}")
            }
            Self::NoImage(message) => write!(formatter, "no image on clipboard: {message}"),
            Self::EncodeFailed(message) => write!(formatter, "could not encode image: {message}"),
            Self::Io(message) => write!(formatter, "io error: {message}"),
        }
    }
}

impl std::error::Error for PasteImageError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PastedImageInfo {
    pub(crate) width: u32,
    pub(crate) height: u32,
}

#[cfg(not(target_os = "android"))]
pub(crate) fn paste_image_to_temp_png() -> Result<(PathBuf, PastedImageInfo), PasteImageError> {
    let mut clipboard = arboard::Clipboard::new()
        .map_err(|error| PasteImageError::ClipboardUnavailable(error.to_string()))?;
    let result = read_clipboard_image(&mut clipboard).and_then(|(image, info)| {
        let mut png = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
            .map_err(|error| PasteImageError::EncodeFailed(error.to_string()))?;
        persist_temp_png(&png).map(|path| (path, info))
    });

    #[cfg(target_os = "linux")]
    if let Err(error) = &result {
        if let Ok(image) = try_wsl_clipboard_fallback(error) {
            return Ok(image);
        }
    }

    result
}

#[cfg(target_os = "android")]
pub(crate) fn paste_image_to_temp_png() -> Result<(PathBuf, PastedImageInfo), PasteImageError> {
    Err(PasteImageError::ClipboardUnavailable(
        "clipboard image paste is unsupported on Android".to_string(),
    ))
}

#[cfg(test)]
fn encode_rgba_as_png(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, PasteImageError> {
    let image = RgbaImage::from_raw(width, height, rgba.to_vec())
        .ok_or_else(|| PasteImageError::EncodeFailed("invalid RGBA buffer".to_string()))?;
    let mut png = Vec::new();
    DynamicImage::ImageRgba8(image)
        .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
        .map_err(|error| PasteImageError::EncodeFailed(error.to_string()))?;
    Ok(png)
}

#[cfg(not(target_os = "android"))]
fn read_clipboard_image(
    clipboard: &mut arboard::Clipboard,
) -> Result<(DynamicImage, PastedImageInfo), PasteImageError> {
    if let Some(image) = clipboard
        .get()
        .file_list()
        .unwrap_or_default()
        .into_iter()
        .find_map(|path| image::open(path).ok())
    {
        let info = PastedImageInfo {
            width: image.width(),
            height: image.height(),
        };
        return Ok((image, info));
    }

    let image = clipboard
        .get_image()
        .map_err(|error| PasteImageError::NoImage(error.to_string()))?;
    let width = u32::try_from(image.width)
        .map_err(|_| PasteImageError::EncodeFailed("image width exceeds u32".to_string()))?;
    let height = u32::try_from(image.height)
        .map_err(|_| PasteImageError::EncodeFailed("image height exceeds u32".to_string()))?;
    let rgba = RgbaImage::from_raw(width, height, image.bytes.into_owned())
        .ok_or_else(|| PasteImageError::EncodeFailed("invalid RGBA buffer".to_string()))?;
    Ok((
        DynamicImage::ImageRgba8(rgba),
        PastedImageInfo { width, height },
    ))
}

fn persist_temp_png(png: &[u8]) -> Result<PathBuf, PasteImageError> {
    let file = write_temp_png(png)?;
    let (_file, path) = file
        .keep()
        .map_err(|error| PasteImageError::Io(error.error.to_string()))?;
    Ok(path)
}

fn write_temp_png(png: &[u8]) -> Result<NamedTempFile, PasteImageError> {
    let file = Builder::new()
        .prefix("tui-clipboard-")
        .suffix(".png")
        .tempfile()
        .map_err(|error| PasteImageError::Io(error.to_string()))?;
    std::fs::write(file.path(), png).map_err(|error| PasteImageError::Io(error.to_string()))?;
    Ok(file)
}

#[cfg(target_os = "linux")]
fn try_wsl_clipboard_fallback(
    error: &PasteImageError,
) -> Result<(PathBuf, PastedImageInfo), PasteImageError> {
    if !is_wsl_session()
        || !matches!(
            error,
            PasteImageError::ClipboardUnavailable(_) | PasteImageError::NoImage(_)
        )
    {
        return Err(error.clone());
    }
    let windows_path = dump_windows_clipboard_image().ok_or_else(|| error.clone())?;
    let path = windows_path_to_wsl(&windows_path).ok_or_else(|| error.clone())?;
    let (width, height) = image::image_dimensions(&path).map_err(|_| error.clone())?;
    Ok((path, PastedImageInfo { width, height }))
}

#[cfg(target_os = "linux")]
fn dump_windows_clipboard_image() -> Option<String> {
    let script = r#"[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; $img = Get-Clipboard -Format Image; if ($img -ne $null) { $p=[System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(),'png'); $img.Save($p,[System.Drawing.Imaging.ImageFormat]::Png); Write-Output $p } else { exit 1 }"#;
    for command in ["powershell.exe", "pwsh", "powershell"] {
        let Ok(output) = std::process::Command::new(command)
            .args(["-NoProfile", "-Command", script])
            .output()
        else {
            continue;
        };
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(path);
            }
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn is_wsl_session() -> bool {
    std::env::var_os("WSL_DISTRO_NAME").is_some()
        || std::env::var_os("WSL_INTEROP").is_some()
        || std::fs::read_to_string("/proc/version").is_ok_and(|version| {
            let version = version.to_ascii_lowercase();
            version.contains("microsoft") || version.contains("wsl")
        })
}

#[cfg(target_os = "linux")]
fn windows_path_to_wsl(path: &str) -> Option<PathBuf> {
    if path.starts_with("\\\\") || path.get(1..2) != Some(":") {
        return None;
    }
    let drive = path.chars().next()?.to_ascii_lowercase();
    if !drive.is_ascii_lowercase() {
        return None;
    }
    let mut mapped = PathBuf::from(format!("/mnt/{drive}"));
    for component in path
        .get(2..)?
        .trim_start_matches(['\\', '/'])
        .split(['\\', '/'])
        .filter(|component| !component.is_empty())
    {
        mapped.push(component);
    }
    Some(mapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_clipboard_pixels_encode_as_png() {
        let png = encode_rgba_as_png(1, 1, &[255, 0, 0, 255]).expect("encode PNG");
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        assert!(encode_rgba_as_png(2, 1, &[255, 0, 0, 255]).is_err());
    }

    #[test]
    fn temporary_clipboard_image_uses_png_suffix_and_bytes() {
        let png = encode_rgba_as_png(1, 1, &[0, 0, 0, 0]).expect("encode PNG");
        let file = write_temp_png(&png).expect("write PNG");
        let path = file.path();
        assert_eq!(
            path.extension().and_then(|value| value.to_str()),
            Some("png")
        );
        assert_eq!(std::fs::read(path).expect("read PNG"), png);
    }

    #[test]
    fn pasted_search_query_collapses_whitespace() {
        assert_eq!(
            normalize_pasted_search_query("  alpha\n\tbeta\r\n gamma  "),
            Some(String::from("alpha beta gamma"))
        );
        assert_eq!(normalize_pasted_search_query(" \n\t "), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn maps_windows_temp_paths_for_wsl() {
        assert_eq!(
            windows_path_to_wsl(r"C:\Users\Alice\AppData\Local\Temp\image.png"),
            Some(PathBuf::from(
                "/mnt/c/Users/Alice/AppData/Local/Temp/image.png"
            ))
        );
        assert_eq!(windows_path_to_wsl(r"\\server\share\image.png"), None);
    }
}
