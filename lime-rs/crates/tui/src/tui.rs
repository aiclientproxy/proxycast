use std::io::{self, Stdout, stdout};
use std::panic;
use std::sync::Arc;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

use crossterm::cursor::Show;
use crossterm::event::{DisableBracketedPaste, EnableBracketedPaste, KeyEvent};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::broadcast;

pub(crate) mod event_stream;
mod frame_rate_limiter;
mod frame_requester;

pub(crate) use event_stream::{EventBroker, TuiEventStream};
pub(crate) use frame_requester::FrameRequester;

/// Normalized events consumed by the TUI runtime.
#[derive(Debug, Clone)]
pub enum TuiEvent {
    Key(KeyEvent),
    Paste(String),
    Resize(ratatui::layout::Size),
    Draw,
    #[allow(dead_code)]
    Resume,
    FocusGained,
    FocusLost,
}

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
    event_broker: Arc<EventBroker>,
    draw_tx: broadcast::Sender<()>,
    frame_requester: FrameRequester,
    terminal_focused: Arc<AtomicBool>,
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
        let event_broker = Arc::new(EventBroker::new());
        let (draw_tx, _) = broadcast::channel(8);
        let frame_requester = FrameRequester::new(draw_tx.clone());
        let terminal_focused = Arc::new(AtomicBool::new(true));
        TERMINAL_ACTIVE.store(true, Ordering::Release);
        Ok(Self {
            terminal,
            restored: false,
            event_broker,
            draw_tx,
            frame_requester,
            terminal_focused,
        })
    }

    pub(crate) fn terminal_mut(&mut self) -> &mut TuiTerminal {
        &mut self.terminal
    }

    pub(crate) fn event_stream(&self) -> TuiEventStream {
        TuiEventStream::new(
            self.event_broker.clone(),
            self.draw_tx.subscribe(),
            self.terminal_focused.clone(),
        )
    }

    pub(crate) fn frame_requester(&self) -> FrameRequester {
        self.frame_requester.clone()
    }

    pub(crate) fn pause_events(&self) {
        self.event_broker.pause_events();
    }

    pub(crate) fn resume_events(&self) {
        self.event_broker.resume_events();
    }

    #[allow(dead_code)]
    pub(crate) fn is_terminal_focused(&self) -> bool {
        self.terminal_focused
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn suspend(&mut self) -> io::Result<()> {
        if self.restored {
            return Ok(());
        }
        self.pause_events();
        let result = (|| {
            self.terminal.show_cursor()?;
            execute!(
                self.terminal.backend_mut(),
                DisableBracketedPaste,
                LeaveAlternateScreen
            )?;
            disable_raw_mode()?;
            TERMINAL_ACTIVE.store(false, Ordering::Release);
            Ok(())
        })();
        if result.is_err() {
            self.resume_events();
        }
        result
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
        // Re-entering the alternate screen restores the previous surface; the next draw
        // reconciles the current frame without synchronously querying stdin.
        self.event_broker.resume_events();
        TERMINAL_ACTIVE.store(true, Ordering::Release);
        Ok(())
    }

    pub(crate) fn restore(&mut self) -> io::Result<()> {
        if self.restored {
            return Ok(());
        }
        self.restored = true;
        self.event_broker.pause_events();
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
