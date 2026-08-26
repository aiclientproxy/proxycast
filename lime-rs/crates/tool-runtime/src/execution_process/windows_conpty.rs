use super::{create_pipe_pair, io, OwnedHandle};
use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
use windows_sys::Win32::System::Console::{
    ClosePseudoConsole, CreatePseudoConsole, ResizePseudoConsole, COORD, HPCON,
};

const PSEUDOCONSOLE_RESIZE_QUIRK: u32 = 0x2;

pub(super) struct RestrictedConpty {
    handle: HPCON,
    _input_read: OwnedHandle,
    _output_write: OwnedHandle,
}

impl RestrictedConpty {
    pub(super) fn create(rows: u16, cols: u16) -> io::Result<(Self, OwnedHandle, OwnedHandle)> {
        let size = checked_size(rows, cols)?;
        let (input_read, input_write) = create_pipe_pair(true)?;
        let (output_read, output_write) = create_pipe_pair(false)?;
        let mut handle = 0;
        let result = unsafe {
            CreatePseudoConsole(
                size,
                input_read.raw(),
                output_write.raw(),
                PSEUDOCONSOLE_RESIZE_QUIRK,
                &mut handle,
            )
        };
        if result < 0 || handle == 0 || handle == INVALID_HANDLE_VALUE {
            return Err(hresult_error("CreatePseudoConsole", result));
        }
        Ok((
            Self {
                handle,
                _input_read: input_read,
                _output_write: output_write,
            },
            input_write,
            output_read,
        ))
    }

    pub(super) fn raw(&self) -> HPCON {
        self.handle
    }

    pub(super) fn resize(&self, rows: u16, cols: u16) -> io::Result<()> {
        let result = unsafe { ResizePseudoConsole(self.handle, checked_size(rows, cols)?) };
        if result < 0 {
            Err(hresult_error("ResizePseudoConsole", result))
        } else {
            Ok(())
        }
    }
}

impl Drop for RestrictedConpty {
    fn drop(&mut self) {
        if self.handle != 0 && self.handle != INVALID_HANDLE_VALUE {
            unsafe {
                ClosePseudoConsole(self.handle);
            }
            self.handle = 0;
        }
    }
}

fn checked_size(rows: u16, cols: u16) -> io::Result<COORD> {
    let rows = i16::try_from(rows)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "ConPTY rows exceed i16::MAX"))?;
    let cols = i16::try_from(cols).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "ConPTY columns exceed i16::MAX",
        )
    })?;
    if rows == 0 || cols == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "ConPTY size must be non-zero",
        ));
    }
    Ok(COORD { X: cols, Y: rows })
}

fn hresult_error(context: &str, result: i32) -> io::Error {
    io::Error::other(format!("{context} failed with HRESULT {result:#010x}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conpty_size_rejects_zero_and_overflow() {
        assert!(checked_size(0, 80).is_err());
        assert!(checked_size(24, 0).is_err());
        assert!(checked_size(i16::MAX as u16 + 1, 80).is_err());
        let size = checked_size(24, 120).unwrap();
        assert_eq!(size.X, 120);
        assert_eq!(size.Y, 24);
    }

    #[test]
    fn hresult_failure_preserves_operation_and_code() {
        let error = hresult_error("CreatePseudoConsole", 0x8007_0057u32 as i32);
        let message = error.to_string();
        assert!(message.contains("CreatePseudoConsole"));
        assert!(message.contains("0x80070057"));
    }
}
