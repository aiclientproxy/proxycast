use super::ExecutionOutputKind;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::PathBuf;

pub(super) const RUNNER_PROTOCOL_VERSION: u8 = 1;
const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct RunnerSpawnRequest {
    pub command: Vec<String>,
    pub cwd: PathBuf,
    pub env: HashMap<String, String>,
    pub capability_sid: String,
    pub expected_account_sid: String,
    pub tty: bool,
    pub stdin_open: bool,
    pub pty_size: Option<(u16, u16)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum RunnerMessage {
    Spawn {
        payload: RunnerSpawnRequest,
    },
    Ready {
        process_id: u32,
    },
    Output {
        kind: ExecutionOutputKind,
        data_base64: String,
    },
    Stdin {
        data_base64: String,
    },
    CloseStdin,
    Resize {
        rows: u16,
        cols: u16,
    },
    Terminate,
    Exit {
        exit_code: i32,
    },
    Error {
        message: String,
        windows_error_code: Option<u32>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct RunnerFrame {
    pub version: u8,
    #[serde(flatten)]
    pub message: RunnerMessage,
}

impl RunnerFrame {
    pub(super) fn new(message: RunnerMessage) -> Self {
        Self {
            version: RUNNER_PROTOCOL_VERSION,
            message,
        }
    }
}

pub(super) fn encode_bytes(bytes: &[u8]) -> String {
    BASE64_STANDARD.encode(bytes)
}

pub(super) fn decode_bytes(value: &str) -> io::Result<Vec<u8>> {
    BASE64_STANDARD
        .decode(value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

pub(super) fn write_frame(writer: &mut impl Write, message: RunnerMessage) -> io::Result<()> {
    let payload = serde_json::to_vec(&RunnerFrame::new(message))
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if payload.len() > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Windows sandbox runner frame exceeds {MAX_FRAME_BYTES} bytes"),
        ));
    }
    let length = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "runner frame is too large"))?;
    writer.write_all(&length.to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()
}

pub(super) fn read_frame(reader: &mut impl Read) -> io::Result<Option<RunnerMessage>> {
    let mut length = [0u8; 4];
    match reader.read_exact(&mut length) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error),
    }
    let length = u32::from_le_bytes(length) as usize;
    if length > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Windows sandbox runner frame exceeds {MAX_FRAME_BYTES} bytes"),
        ));
    }
    let mut payload = vec![0u8; length];
    reader.read_exact(&mut payload)?;
    let frame: RunnerFrame = serde_json::from_slice(&payload)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if frame.version != RUNNER_PROTOCOL_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported Windows sandbox runner protocol version {}",
                frame.version
            ),
        ));
    }
    Ok(Some(frame.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framed_output_round_trips_raw_bytes() {
        let mut bytes = Vec::new();
        write_frame(
            &mut bytes,
            RunnerMessage::Output {
                kind: ExecutionOutputKind::Stdout,
                data_base64: encode_bytes(&[0, 1, 2, 255]),
            },
        )
        .expect("frame should encode");

        let message = read_frame(&mut bytes.as_slice())
            .expect("frame should decode")
            .expect("frame should be present");
        let RunnerMessage::Output { data_base64, .. } = message else {
            panic!("unexpected runner message")
        };
        assert_eq!(decode_bytes(&data_base64).unwrap(), [0, 1, 2, 255]);
    }

    #[test]
    fn oversized_frame_is_rejected_before_allocation() {
        let mut encoded = ((MAX_FRAME_BYTES + 1) as u32).to_le_bytes().to_vec();
        encoded.extend_from_slice(b"{}");
        let error = read_frame(&mut encoded.as_slice()).expect_err("oversized frame must fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }
}
