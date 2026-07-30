pub(super) const MAX_IMAGE_MEDIA_BYTES: usize = 32 * 1024 * 1024;

pub(super) fn validate_image_bytes<'a>(
    bytes: &[u8],
    declared_media_type: Option<&'a str>,
) -> Result<&'a str, String> {
    let detected = detected_image_media_type(bytes)
        .ok_or_else(|| "image payload is not a supported image".to_string())?;
    if let Some(declared) = declared_media_type {
        if declared != detected {
            return Err(format!(
                "image media type mismatch: declared {declared}, detected {detected}"
            ));
        }
        return Ok(declared);
    }
    Ok(detected)
}

pub(super) fn extension_for_media_type(media_type: &str) -> &'static str {
    match media_type {
        "image/jpeg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        _ => "png",
    }
}

fn detected_image_media_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some("image/png");
    }
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return Some("image/jpeg");
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some("image/gif");
    }
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return Some("image/webp");
    }
    None
}
