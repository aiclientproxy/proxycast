//! Length-prefixed JSON framing for the Code Mode stdio transport.
//!
//! The implementation remains owned by the protocol module so every host
//! transport applies the same size and decoding rules.

use serde::de::DeserializeOwned;
use serde::Serialize;
use std::io;
use std::mem::size_of;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Maximum JSON payload accepted for one frame.
pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct EncodedFrame {
    payload: Vec<u8>,
}

impl EncodedFrame {
    pub fn encode<T: Serialize>(message: &T) -> io::Result<Self> {
        let payload = serde_json::to_vec(message).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to encode code-mode IPC frame: {error}"),
            )
        })?;
        if payload.len() > MAX_FRAME_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "code-mode IPC frame length {} exceeds {MAX_FRAME_BYTES} bytes",
                    payload.len()
                ),
            ));
        }
        Ok(Self { payload })
    }

    pub fn into_framed_bytes(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(size_of::<u32>() + self.payload.len());
        bytes.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.payload);
        bytes
    }

    /// Decodes exactly one complete length-prefixed frame.
    pub fn decode_framed<T: DeserializeOwned>(bytes: &[u8]) -> io::Result<T> {
        let length_bytes: [u8; size_of::<u32>()] = bytes
            .get(..size_of::<u32>())
            .and_then(|value| value.try_into().ok())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "code-mode IPC frame is missing its length prefix",
                )
            })?;
        let length = u32::from_le_bytes(length_bytes) as usize;
        if length > MAX_FRAME_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("code-mode IPC frame length {length} exceeds {MAX_FRAME_BYTES} bytes"),
            ));
        }
        let payload = bytes.get(size_of::<u32>()..).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "code-mode IPC frame is missing its payload",
            )
        })?;
        if payload.len() != length {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "code-mode IPC frame declares {length} payload bytes but contains {}",
                    payload.len()
                ),
            ));
        }
        serde_json::from_slice(payload).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to decode code-mode IPC frame: {error}"),
            )
        })
    }
}

pub struct FramedReader<R> {
    reader: R,
}

impl<R: AsyncRead + Unpin> FramedReader<R> {
    pub fn new(reader: R) -> Self {
        Self { reader }
    }

    pub async fn read<T: DeserializeOwned>(&mut self) -> io::Result<Option<T>> {
        let mut length_bytes = [0_u8; size_of::<u32>()];
        if self.reader.read(&mut length_bytes[..1]).await? == 0 {
            return Ok(None);
        }
        self.reader.read_exact(&mut length_bytes[1..]).await?;
        let length = u32::from_le_bytes(length_bytes) as usize;
        if length > MAX_FRAME_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("code-mode IPC frame length {length} exceeds {MAX_FRAME_BYTES} bytes"),
            ));
        }
        let mut payload = vec![0; length];
        self.reader.read_exact(&mut payload).await?;
        serde_json::from_slice(&payload).map(Some).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to decode code-mode IPC frame: {error}"),
            )
        })
    }
}

pub struct FramedWriter<W> {
    writer: W,
}

impl<W: AsyncWrite + Unpin> FramedWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    pub async fn write<T: Serialize>(&mut self, message: &T) -> io::Result<()> {
        let frame = EncodedFrame::encode(message)?;
        let length = u32::try_from(frame.payload.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "code-mode IPC frame exceeds u32",
            )
        })?;
        self.writer.write_all(&length.to_le_bytes()).await?;
        self.writer.write_all(&frame.payload).await?;
        self.writer.flush().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Message {
        value: String,
    }

    #[test]
    fn encoded_frame_round_trips_and_rejects_trailing_bytes() {
        let frame = EncodedFrame::encode(&Message { value: "ok".into() })
            .expect("message should encode")
            .into_framed_bytes();
        assert_eq!(
            EncodedFrame::decode_framed::<Message>(&frame).expect("frame should decode"),
            Message { value: "ok".into() }
        );
        let mut trailing = frame.clone();
        trailing.push(0);
        assert!(EncodedFrame::decode_framed::<Message>(&trailing).is_err());
    }

    #[test]
    fn encoded_frame_rejects_truncated_prefix_and_payload() {
        assert!(EncodedFrame::decode_framed::<Message>(&[]).is_err());
        assert!(EncodedFrame::decode_framed::<Message>(&[1, 0, 0, 0]).is_err());
    }
}
