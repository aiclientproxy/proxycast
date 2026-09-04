use std::io::{self, stdout, Stdout};
use std::panic;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

use crossterm::cursor::Show;
use crossterm::event::{DisableBracketedPaste, EnableBracketedPaste};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

pub(crate) type TuiTerminal = Terminal<CrosstermBackend<Stdout>>;

static PANIC_HOOK: Once = Once::new();
static TERMINAL_ACTIVE: AtomicBool = AtomicBool::new(false);

fn install_panic_hook() {
    PANIC_HOOK.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |panic_info| {
            if TERMINAL_ACTIVE.swap(false, Ordering::AcqRel) {
                let _ = restore_terminal_state();
            }
            previous(panic_info);
        }));
    });
}

fn restore_terminal_state() -> io::Result<()> {
    let mut output = stdout();
    let mut first_error = crossterm::execute!(output, Show).err();
    if let Err(error) = crossterm::execute!(output, DisableBracketedPaste, LeaveAlternateScreen) {
        first_error.get_or_insert(error);
    }
    if let Err(error) = disable_raw_mode() {
        first_error.get_or_insert(error);
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

pub(crate) struct TerminalGuard {
    terminal: TuiTerminal,
    restored: bool,
}

impl TerminalGuard {
    pub(crate) fn enter() -> io::Result<Self> {
        install_panic_hook();
        enable_raw_mode()?;
        let mut output = stdout();
        if let Err(error) = execute!(output, EnterAlternateScreen, EnableBracketedPaste) {
            let _ = disable_raw_mode();
            return Err(error);
        }
        let terminal = match Terminal::new(CrosstermBackend::new(output)) {
            Ok(terminal) => terminal,
            Err(error) => {
                let mut output = stdout();
                let _ = execute!(output, DisableBracketedPaste, LeaveAlternateScreen);
                let _ = disable_raw_mode();
                return Err(error);
            }
        };
        TERMINAL_ACTIVE.store(true, Ordering::Release);
        Ok(Self {
            terminal,
            restored: false,
        })
    }

    pub(crate) fn terminal_mut(&mut self) -> &mut TuiTerminal {
        &mut self.terminal
    }

    pub(crate) fn suspend(&mut self) -> io::Result<()> {
        if self.restored {
            return Ok(());
        }
        self.terminal.show_cursor()?;
        execute!(
            self.terminal.backend_mut(),
            DisableBracketedPaste,
            LeaveAlternateScreen
        )?;
        disable_raw_mode()?;
        TERMINAL_ACTIVE.store(false, Ordering::Release);
        Ok(())
    }

    pub(crate) fn resume(&mut self) -> io::Result<()> {
        if self.restored {
            return Ok(());
        }
        enable_raw_mode()?;
        if let Err(error) = execute!(
            self.terminal.backend_mut(),
            EnterAlternateScreen,
            EnableBracketedPaste
        ) {
            let _ = disable_raw_mode();
            return Err(error);
        }
        self.terminal.clear()?;
        TERMINAL_ACTIVE.store(true, Ordering::Release);
        Ok(())
    }

    pub(crate) fn restore(&mut self) -> io::Result<()> {
        if self.restored {
            return Ok(());
        }
        self.restored = true;
        TERMINAL_ACTIVE.store(false, Ordering::Release);
        let mut first_error = self.terminal.show_cursor().err();
        if let Err(error) = execute!(
            self.terminal.backend_mut(),
            DisableBracketedPaste,
            LeaveAlternateScreen
        ) {
            first_error.get_or_insert(error);
        }
        if let Err(error) = disable_raw_mode() {
            first_error.get_or_insert(error);
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}
