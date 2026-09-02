import AppKit
import Darwin
import Foundation

final class WindowFixtureDelegate: NSObject, NSApplicationDelegate {
    private let outputURL: URL
    private var windows: [NSWindow] = []

    init(outputPath: String) {
        self.outputURL = URL(fileURLWithPath: outputPath)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
        for index in 0..<2 {
            let frame = NSRect(
                x: 160 + CGFloat(index * 360),
                y: 360,
                width: 300,
                height: 180
            )
            let window = NSWindow(
                contentRect: frame,
                styleMask: [.titled, .closable],
                backing: .buffered,
                defer: false
            )
            window.title = "Lime Gate B Fixture \(index + 1)"
            window.isReleasedWhenClosed = false
            window.makeKeyAndOrderFront(nil)
            windows.append(window)
        }

        let record: [String: Any] = [
            "pid": ProcessInfo.processInfo.processIdentifier,
            "titles": windows.map(\.title),
        ]
        if JSONSerialization.isValidJSONObject(record),
           let data = try? JSONSerialization.data(withJSONObject: record) {
            try? data.write(to: outputURL, options: .atomic)
        }
    }
}

guard CommandLine.arguments.count >= 2 else {
    exit(64)
}
let application = NSApplication.shared
let delegate = WindowFixtureDelegate(outputPath: CommandLine.arguments[1])
application.delegate = delegate
application.run()
