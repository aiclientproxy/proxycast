use std::process::ExitCode;

pub(crate) fn handle_exit_status(code: i32) -> ExitCode {
    ExitCode::from(u8::try_from(code).unwrap_or(1))
}
