import AppKit
import CoreGraphics
import Foundation

private let displayWatcherLock = NSLock()
private var displayWatcherRunning = false

private func displayChangePayload(displayID: CGDirectDisplayID, flags: CGDisplayChangeSummaryFlags) -> [String: Any] {
    [
        "displayId": displayID,
        "flags": flags.rawValue,
        "displays": readDisplays()["displays"] ?? [],
    ]
}
private func displayReconfigurationCallback(
    _ display: CGDirectDisplayID,
    _ flags: CGDisplayChangeSummaryFlags,
    _ userInfo: UnsafeMutableRawPointer?
) {
    displayWatcherLock.lock()
    let active = displayWatcherRunning
    displayWatcherLock.unlock()
    guard active else {
        return
    }
    DispatchQueue.main.async {
        displayWatcherLock.lock()
        let stillActive = displayWatcherRunning
        displayWatcherLock.unlock()
        guard stillActive else {
            return
        }
        writeEvent(
            event: "display.changed",
            payload: displayChangePayload(displayID: display, flags: flags)
        )
    }
}

func startDisplayWatcher() throws -> [String: Any] {
    displayWatcherLock.lock()
    if displayWatcherRunning {
        displayWatcherLock.unlock()
        return ["started": true, "alreadyRunning": true]
    }
    let result = CGDisplayRegisterReconfigurationCallback(
        displayReconfigurationCallback,
        nil
    )
    guard result == .success else {
        displayWatcherLock.unlock()
        throw HostError.operationFailed("macOS display reconfiguration watcher could not be installed.")
    }
    displayWatcherRunning = true
    displayWatcherLock.unlock()
    return [
        "started": true,
        "alreadyRunning": false,
        "displays": readDisplays()["displays"] ?? [],
    ]
}

func stopDisplayWatcher() -> [String: Any] {
    displayWatcherLock.lock()
    guard displayWatcherRunning else {
        displayWatcherLock.unlock()
        return ["stopped": false]
    }
    displayWatcherRunning = false
    displayWatcherLock.unlock()
    CGDisplayRemoveReconfigurationCallback(displayReconfigurationCallback, nil)
    return ["stopped": true]
}
