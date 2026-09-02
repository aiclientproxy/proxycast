import AppKit
import ApplicationServices
import CoreGraphics
import Darwin
import Foundation
import IOKit.hid

private let accessibilitySettingsURL =
    "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
private let inputMonitoringSettingsURL =
    "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"
private let screenCaptureSettingsURL =
    "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"

private let outputLock = NSLock()
private let hidWatcherLock = NSLock()
private var hidWatcherTimer: DispatchSourceTimer?
private var hidWatcherFingerprint: String?
private var bareModifierMonitorToken: Any?

func response(id: Any, result: [String: Any]) -> [String: Any] {
    ["id": id, "ok": true, "result": result]
}

func failure(id: Any, code: String, message: String, data: [String: Any] = [:])
    -> [String: Any]
{
    [
        "id": id,
        "ok": false,
        "error": ["code": code, "message": message, "data": data],
    ]
}

func string(_ value: Any?, _ name: String) throws -> String {
    guard let value = value as? String, !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw HostError.invalidArgument("Missing required string field: \(name)")
    }
    return value.trimmingCharacters(in: .whitespacesAndNewlines)
}

func uint32(_ value: Any?, _ name: String) throws -> UInt32 {
    guard let number = value as? NSNumber, number.int64Value >= 0,
          number.int64Value <= Int64(UInt32.max) else {
        throw HostError.invalidArgument("Missing or invalid numeric field: \(name)")
    }
    return number.uint32Value
}

func dictionary(_ value: Any?) -> [String: Any] {
    value as? [String: Any] ?? [:]
}

enum HostError: Error {
    case invalidArgument(String)
    case unavailable(String)
    case notGranted(String)
    case operationFailed(String)
}

func errorDetails(_ error: HostError) -> (String, String, [String: Any]) {
    switch error {
    case .invalidArgument(let message):
        return ("invalid_argument", message, [:])
    case .unavailable(let message):
        return ("unavailable", message, [:])
    case .notGranted(let message):
        return ("not_granted", message, [:])
    case .operationFailed(let message):
        return ("operation_failed", message, [:])
    }
}

func capabilityStatus(
    _ granted: Bool,
    grantedReason: String,
    deniedReason: String,
    settingsURL: String
)
    -> [String: Any]
{
    [
        "status": granted ? "ready" : "not_granted",
        "reason": granted ? grantedReason : deniedReason,
        "settingsUrl": settingsURL,
    ]
}

func screenCaptureStatus(_ granted: Bool, reason: String? = nil) -> [String: Any] {
    [
        "status": granted ? "ready" : "not_granted",
        "reason": reason
            ?? (granted
                ? "macOS Screen Recording permission is granted."
                : "macOS Screen Recording permission is not granted."),
        "settingsUrl": screenCaptureSettingsURL,
    ]
}

func readCapabilities() -> [String: Any] {
    let bundleIdentifier = Bundle.main.bundleIdentifier ?? "com.limecloud.lime.native-host"
    let applicationGroup: [String: Any] = [
        "status": "not_configured",
        "reason": "The native host has no Application Group consumer.",
        "identifiers": [],
    ]
    let nativeReady: [String: Any] = [
        "status": "ready",
        "reason": "The native host API is available; packaged entitlement and user authorization are reported separately.",
    ]
    return [
        "protocolVersion": 1,
        "helperId": "macos-native-host",
        "platform": "darwin",
        "applicationId": bundleIdentifier,
        "accessibility": capabilityStatus(
            AXIsProcessTrusted(),
            grantedReason: "macOS Accessibility permission is granted.",
            deniedReason: "macOS Accessibility permission is not granted.",
            settingsURL: accessibilitySettingsURL
        ),
        "inputMonitoring": capabilityStatus(
            CGPreflightListenEventAccess(),
            grantedReason: "macOS Input Monitoring permission is granted.",
            deniedReason: "macOS Input Monitoring permission is not granted.",
            settingsURL: inputMonitoringSettingsURL
        ),
        "screenCapture": screenCaptureStatus(CGPreflightScreenCaptureAccess()),
        "appleEvents": [
            "status": "ready",
            "reason": "Apple Events authorization queries are provided by the macOS native host.",
            "settingsUrl": "x-apple.systempreferences:com.apple.preference.security?Privacy_Automation",
        ],
        "applicationGroups": applicationGroup,
        "windowHandles": nativeReady,
        "windowOrchestration": nativeReady,
        "accessibilityTree": nativeReady,
        "displays": nativeReady,
        "displayWatcher": nativeReady,
        "mediaPermissions": readMediaPermissions(),
        "hidTopology": nativeReady,
        "bareModifierMonitor": capabilityStatus(
            CGPreflightListenEventAccess(),
            grantedReason: "Bare modifier monitoring can be enabled after Input Monitoring authorization.",
            deniedReason: "Input Monitoring authorization is required for bare modifier monitoring.",
            settingsURL: inputMonitoringSettingsURL
        ),
        "securityScopedBookmarks": nativeReady,
        "localAuthentication": readLocalAuthentication(),
        "deviceKey": [
            "status": "unavailable",
            "reason": "Secure Enclave availability depends on compatible hardware and signed key entitlements.",
        ],
    ]
}

func readScreenCapture() -> [String: Any] {
    screenCaptureStatus(CGPreflightScreenCaptureAccess())
}

func requestScreenCapture() -> [String: Any] {
    screenCaptureStatus(CGRequestScreenCaptureAccess())
}

func openSettings(_ settingsURL: String) throws -> [String: Any] {
    guard let url = URL(string: settingsURL), NSWorkspace.shared.open(url) else {
        throw HostError.operationFailed("Unable to open macOS privacy settings.")
    }
    return ["opened": true, "settingsUrl": settingsURL]
}

func readBundleIdentifier(path: String) throws -> [String: Any] {
    let fileURL = URL(fileURLWithPath: path)
    guard let applicationURL = NSWorkspace.shared.urlForApplication(toOpen: fileURL) else {
        throw HostError.unavailable("No application is registered for the path.")
    }
    guard let bundle = Bundle(url: applicationURL), let identifier = bundle.bundleIdentifier else {
        throw HostError.unavailable("The registered application has no bundle identifier.")
    }
    return [
        "path": applicationURL.path,
        "bundleIdentifier": identifier,
    ]
}

func readURLHandlers(urlString: String) throws -> [String: Any] {
    guard let url = URL(string: urlString), url.scheme != nil else {
        throw HostError.invalidArgument("The URL must include a scheme.")
    }
    let applications = NSWorkspace.shared.urlsForApplications(toOpen: url).compactMap { applicationURL -> [String: Any]? in
        guard let bundle = Bundle(url: applicationURL), let identifier = bundle.bundleIdentifier else {
            return nil
        }
        return ["path": applicationURL.path, "bundleIdentifier": identifier]
    }
    return ["url": urlString, "applications": applications]
}

func readWindows() -> [String: Any] {
    let windowInfo = CGWindowListCopyWindowInfo(
        [.optionOnScreenOnly, .excludeDesktopElements],
        kCGNullWindowID
    ) as? [[String: Any]] ?? []
    let windows = windowInfo.compactMap { info -> [String: Any]? in
        guard let number = info[kCGWindowNumber as String] as? NSNumber else {
            return nil
        }
        let ownerPID = (info[kCGWindowOwnerPID as String] as? NSNumber)?.int32Value ?? 0
        let application = NSRunningApplication(processIdentifier: pid_t(ownerPID))
        let bounds = info[kCGWindowBounds as String] as? [String: Any]
        let frame: [String: Any] = [
            "x": (bounds?["X"] as? NSNumber)?.doubleValue ?? 0,
            "y": (bounds?["Y"] as? NSNumber)?.doubleValue ?? 0,
            "width": (bounds?["Width"] as? NSNumber)?.doubleValue ?? 0,
            "height": (bounds?["Height"] as? NSNumber)?.doubleValue ?? 0,
        ]
        var window: [String: Any] = [
            "windowId": number.uint32Value,
            "ownerPid": ownerPID,
            "ownerName": info[kCGWindowOwnerName as String] as? String ?? "",
            "title": info[kCGWindowName as String] as? String ?? "",
            "layer": (info[kCGWindowLayer as String] as? NSNumber)?.intValue ?? 0,
            "isOnScreen": info[kCGWindowIsOnscreen as String] as? Bool ?? true,
            "bounds": frame,
        ]
        if let bundleIdentifier = application?.bundleIdentifier {
            window["bundleIdentifier"] = bundleIdentifier
        }
        return window
    }
    return ["windows": windows]
}

func windowOwnerPID(windowID: UInt32) throws -> pid_t {
    let windowInfo = CGWindowListCopyWindowInfo(
        [.optionIncludingWindow],
        CGWindowID(windowID)
    ) as? [[String: Any]] ?? []
    guard let info = windowInfo.first,
          let ownerPID = (info[kCGWindowOwnerPID as String] as? NSNumber)?.int32Value,
          ownerPID > 0 else {
        throw HostError.unavailable("The requested macOS window is no longer available.")
    }
    return pid_t(ownerPID)
}

func focusWindow(windowID: UInt32) throws -> [String: Any] {
    let ownerPID = try windowOwnerPID(windowID: windowID)
    guard let application = NSRunningApplication(processIdentifier: ownerPID) else {
        throw HostError.unavailable("The requested macOS window owner is no longer running.")
    }
    guard application.activate(options: [.activateIgnoringOtherApps]) else {
        throw HostError.operationFailed("The macOS window owner could not be activated.")
    }
    return ["windowId": windowID, "ownerPid": ownerPID, "focused": true]
}

func raiseWindow(windowID: UInt32) throws -> [String: Any] {
    let ownerPID = try windowOwnerPID(windowID: windowID)
    let window = try accessibilityWindowMatching(
        ownerPID: ownerPID,
        frame: try windowBounds(windowID: windowID)
    )
    guard AXUIElementPerformAction(window, kAXRaiseAction as CFString) == .success else {
        throw HostError.operationFailed("The macOS window could not be raised.")
    }
    return ["windowId": windowID, "ownerPid": ownerPID, "raised": true]
}

func setWindowOwnerVisibility(windowID: UInt32, hidden: Bool) throws -> [String: Any] {
    let ownerPID = try windowOwnerPID(windowID: windowID)
    guard let application = NSRunningApplication(processIdentifier: ownerPID) else {
        throw HostError.unavailable("The requested macOS window owner is no longer running.")
    }
    let alreadyHidden = application.isHidden
    let changed = alreadyHidden == hidden || setApplicationHidden(application, hidden: hidden)
    guard changed else {
        throw HostError.operationFailed("The macOS window owner visibility could not be updated.")
    }
    return [
        "windowId": windowID,
        "ownerPid": ownerPID,
        "hidden": hidden,
        "changed": alreadyHidden != hidden,
    ]
}

func setWindowFrame(windowID: UInt32, frame: [String: Any]) throws -> [String: Any] {
    let ownerPID = try windowOwnerPID(windowID: windowID)
    let window = try accessibilityWindowMatching(
        ownerPID: ownerPID,
        frame: try windowBounds(windowID: windowID)
    )
    let x = (frame["x"] as? NSNumber)?.doubleValue
    let y = (frame["y"] as? NSNumber)?.doubleValue
    let width = (frame["width"] as? NSNumber)?.doubleValue
    let height = (frame["height"] as? NSNumber)?.doubleValue
    if let x, let y {
        var point = CGPoint(x: x, y: y)
        guard let position = AXValueCreate(.cgPoint, &point),
              AXUIElementSetAttributeValue(window, kAXPositionAttribute as CFString, position) == .success else {
            throw HostError.operationFailed("The macOS window position could not be updated.")
        }
    }
    if let width, let height {
        var size = CGSize(width: width, height: height)
        guard let sizeValue = AXValueCreate(.cgSize, &size),
              AXUIElementSetAttributeValue(window, kAXSizeAttribute as CFString, sizeValue) == .success else {
            throw HostError.operationFailed("The macOS window size could not be updated.")
        }
    }
    return ["windowId": windowID, "ownerPid": ownerPID, "updated": true]
}

func readDisplays() -> [String: Any] {
    let displays = NSScreen.screens.map { screen -> [String: Any] in
        let number = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber
        let frame = screen.frame
        let visibleFrame = screen.visibleFrame
        var display: [String: Any] = [
            "name": screen.localizedName,
            "scaleFactor": screen.backingScaleFactor,
            "frame": [
                "x": frame.origin.x,
                "y": frame.origin.y,
                "width": frame.size.width,
                "height": frame.size.height,
            ],
            "visibleFrame": [
                "x": visibleFrame.origin.x,
                "y": visibleFrame.origin.y,
                "width": visibleFrame.size.width,
                "height": visibleFrame.size.height,
            ],
        ]
        if let displayId = number?.uint32Value {
            display["displayId"] = displayId
        }
        return display
    }
    return ["displays": displays]
}

func hidStringProperty(_ device: IOHIDDevice, _ key: CFString) -> String? {
    guard let value = IOHIDDeviceGetProperty(device, key) else {
        return nil
    }
    if let string = value as? String {
        return string
    }
    return String(describing: value)
}

func hidNumberProperty(_ device: IOHIDDevice, _ key: CFString) -> Int? {
    guard let value = IOHIDDeviceGetProperty(device, key) else {
        return nil
    }
    return (value as? NSNumber)?.intValue
}

func readHIDTopology() -> [String: Any] {
    let manager = IOHIDManagerCreate(kCFAllocatorDefault, IOOptionBits(kIOHIDOptionsTypeNone))
    guard let deviceSet = IOHIDManagerCopyDevices(manager) else {
        return ["devices": []]
    }
    let devices = (deviceSet as NSSet).compactMap { item -> [String: Any]? in
        let device = item as! IOHIDDevice
        var record: [String: Any] = [:]
        if let product = hidStringProperty(device, kIOHIDProductKey as CFString) {
            record["product"] = product
        }
        if let manufacturer = hidStringProperty(device, kIOHIDManufacturerKey as CFString) {
            record["manufacturer"] = manufacturer
        }
        if let transport = hidStringProperty(device, kIOHIDTransportKey as CFString) {
            record["transport"] = transport
        }
        if let vendorId = hidNumberProperty(device, kIOHIDVendorIDKey as CFString) {
            record["vendorId"] = vendorId
        }
        if let productId = hidNumberProperty(device, kIOHIDProductIDKey as CFString) {
            record["productId"] = productId
        }
        if let locationId = hidNumberProperty(device, kIOHIDLocationIDKey as CFString) {
            record["locationId"] = locationId
        }
        if let usagePage = hidNumberProperty(device, kIOHIDPrimaryUsagePageKey as CFString) {
            record["usagePage"] = usagePage
        }
        if let usage = hidNumberProperty(device, kIOHIDPrimaryUsageKey as CFString) {
            record["usage"] = usage
        }
        return record
    }
    return ["devices": devices]
}

func hidTopologyFingerprint(_ topology: [String: Any]) -> String {
    let devices = topology["devices"] as? [[String: Any]] ?? []
    let keys = [
        "manufacturer",
        "product",
        "transport",
        "vendorId",
        "productId",
        "locationId",
        "usagePage",
        "usage",
    ]
    return devices.map { device in
        keys.map { key in
            "\(key)=\(String(describing: device[key] ?? ""))"
        }.joined(separator: ";")
    }.sorted().joined(separator: "|")
}

func startHIDTopologyWatcher(intervalMs: Int?) throws -> [String: Any] {
    let requestedInterval = intervalMs ?? 500
    guard (100...60_000).contains(requestedInterval) else {
        throw HostError.invalidArgument("HID watcher interval must be between 100 and 60000 ms.")
    }

    hidWatcherLock.lock()
    defer { hidWatcherLock.unlock() }
    if hidWatcherTimer != nil {
        return [
            "started": true,
            "alreadyRunning": true,
            "intervalMs": requestedInterval,
        ]
    }

    let initialTopology = readHIDTopology()
    hidWatcherFingerprint = hidTopologyFingerprint(initialTopology)
    let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
    timer.schedule(
        deadline: .now() + .milliseconds(requestedInterval),
        repeating: .milliseconds(requestedInterval)
    )
    timer.setEventHandler {
        let topology = readHIDTopology()
        let fingerprint = hidTopologyFingerprint(topology)
        hidWatcherLock.lock()
        let changed = fingerprint != hidWatcherFingerprint
        if changed {
            hidWatcherFingerprint = fingerprint
        }
        hidWatcherLock.unlock()
        if changed {
            writeEvent(event: "hidTopology.changed", payload: topology)
        }
    }
    hidWatcherTimer = timer
    timer.resume()
    return [
        "started": true,
        "alreadyRunning": false,
        "intervalMs": requestedInterval,
        "topology": initialTopology,
    ]
}

func stopHIDTopologyWatcher() -> [String: Any] {
    hidWatcherLock.lock()
    let timer = hidWatcherTimer
    hidWatcherTimer = nil
    hidWatcherFingerprint = nil
    hidWatcherLock.unlock()
    timer?.cancel()
    return ["stopped": timer != nil]
}

func modifierName(for keyCode: UInt16) -> String? {
    switch keyCode {
    case 54, 55:
        return "command"
    case 58, 61:
        return "option"
    case 59, 62:
        return "control"
    case 56, 60:
        return "shift"
    case 63:
        return "function"
    case 57:
        return "capsLock"
    default:
        return nil
    }
}

func modifierFlag(for name: String) -> NSEvent.ModifierFlags {
    switch name {
    case "command":
        return .command
    case "option":
        return .option
    case "control":
        return .control
    case "shift":
        return .shift
    case "function":
        return .function
    case "capsLock":
        return .capsLock
    default:
        return []
    }
}

func startBareModifierMonitor(requestPermission: Bool) throws -> [String: Any] {
    if !CGPreflightListenEventAccess() && requestPermission {
        _ = CGRequestListenEventAccess()
    }
    guard CGPreflightListenEventAccess() else {
        throw HostError.notGranted("macOS Input Monitoring permission is required for bare modifier monitoring.")
    }
    if bareModifierMonitorToken != nil {
        return ["started": true, "alreadyRunning": true]
    }
    guard let token = NSEvent.addGlobalMonitorForEvents(matching: .flagsChanged, handler: { event in
        guard let name = modifierName(for: event.keyCode) else {
            return
        }
        let active = event.modifierFlags.contains(modifierFlag(for: name))
        writeEvent(
            event: "bareModifier.changed",
            payload: [
                "modifier": name,
                "phase": active ? "down" : "up",
                "keyCode": event.keyCode,
                "timestamp": event.timestamp,
                "bare": true,
            ]
        )
    }) else {
        throw HostError.unavailable("macOS global modifier monitor could not be installed.")
    }
    bareModifierMonitorToken = token
    return ["started": true, "alreadyRunning": false]
}

func stopBareModifierMonitor() -> [String: Any] {
    guard let token = bareModifierMonitorToken else {
        return ["stopped": false]
    }
    NSEvent.removeMonitor(token)
    bareModifierMonitorToken = nil
    return ["stopped": true]
}

func openPath(path: String) throws -> [String: Any] {
    let fileURL = URL(fileURLWithPath: path)
    guard NSWorkspace.shared.open(fileURL) else {
        throw HostError.operationFailed("Unable to open the requested path.")
    }
    return ["opened": true, "path": fileURL.path]
}

func handle(method: String, params: [String: Any]) throws -> [String: Any] {
    switch method {
    case "capabilities.read":
        return readCapabilities()
    case "accessibility.read":
        return capabilityStatus(
            AXIsProcessTrusted(),
            grantedReason: "macOS Accessibility permission is granted.",
            deniedReason: "macOS Accessibility permission is not granted.",
            settingsURL: accessibilitySettingsURL
        )
    case "accessibility.openSettings":
        return try openSettings(accessibilitySettingsURL)
    case "inputMonitoring.openSettings":
        return try openSettings(inputMonitoringSettingsURL)
    case "inputMonitoring.read":
        return capabilityStatus(
            CGPreflightListenEventAccess(),
            grantedReason: "macOS Input Monitoring permission is granted.",
            deniedReason: "macOS Input Monitoring permission is not granted.",
            settingsURL: inputMonitoringSettingsURL
        )
    case "inputMonitoring.request":
        let granted = CGRequestListenEventAccess()
        return capabilityStatus(
            granted,
            grantedReason: "macOS Input Monitoring permission is granted.",
            deniedReason: "macOS Input Monitoring permission was not granted.",
            settingsURL: inputMonitoringSettingsURL
        )
    case "screenCapture.read":
        return readScreenCapture()
    case "screenCapture.request":
        return requestScreenCapture()
    case "screenCapture.snapshot":
        return try screenCaptureSnapshot(params: params)
    case "appleEvents.read":
        return try readAppleEventsPermission(params: params)
    case "appleEvents.request":
        return try requestAppleEventsPermission(params: params)
    case "appleEvents.targets":
        return readAppleEventsTargets()
    case "appleEvents.openSettings":
        return try openAppleEventsSettings()
    case "launchServices.openPath":
        return try openPath(path: string(params["path"], "path"))
    case "launchServices.bundleIdentifier":
        return try readBundleIdentifier(path: string(params["path"], "path"))
    case "launchServices.urlHandlers":
        return try readURLHandlers(urlString: string(params["url"], "url"))
    case "window.read":
        return readWindows()
    case "window.focus":
        let windowID = try uint32(params["windowId"], "windowId")
        return try focusWindow(windowID: windowID)
    case "window.raise":
        let windowID = try uint32(params["windowId"], "windowId")
        return try raiseWindow(windowID: windowID)
    case "window.setOwnerVisibility":
        let windowID = try uint32(params["windowId"], "windowId")
        let hidden = (params["hidden"] as? NSNumber)?.boolValue ?? true
        return try setWindowOwnerVisibility(windowID: windowID, hidden: hidden)
    case "window.setFrame":
        let windowID = try uint32(params["windowId"], "windowId")
        return try setWindowFrame(windowID: windowID, frame: dictionary(params["frame"]))
    case "window.anchor":
        let windowID = try uint32(params["windowId"], "windowId")
        let anchorWindowID = try uint32(params["anchorWindowId"], "anchorWindowId")
        return try anchorWindow(windowID: windowID, anchorWindowID: anchorWindowID, params: params)
    case "window.stack":
        return try stackWindows(windowIDs: windowIDList(params["windowIds"]))
    case "window.hideForTask.start":
        return try startHideForTask(
            taskID: taskIdentifier(params["taskId"]),
            windowIDs: windowIDList(params["windowIds"])
        )
    case "window.hideForTask.stop":
        return try stopHideForTask(taskID: taskIdentifier(params["taskId"]))
    case "window.hideForTask.read":
        return readHideForTasks()
    case "accessibilityTree.read":
        let windowID = try uint32(params["windowId"], "windowId")
        return try readAccessibilityTree(windowID: windowID, params: params)
    case "display.read":
        return readDisplays()
    case "display.watch.start":
        return try startDisplayWatcher()
    case "display.watch.stop":
        return stopDisplayWatcher()
    case "mediaPermissions.read":
        return readMediaPermissions()
    case "mediaPermissions.request":
        return try requestMediaPermission(kind: string(params["kind"], "kind"))
    case "hidTopology.read":
        return readHIDTopology()
    case "hidTopology.watch.start":
        let interval = params["intervalMs"] as? NSNumber
        return try startHIDTopologyWatcher(intervalMs: interval?.intValue)
    case "hidTopology.watch.stop":
        return stopHIDTopologyWatcher()
    case "bareModifierMonitor.start":
        let requestPermission = (params["requestPermission"] as? NSNumber)?.boolValue ?? false
        return try startBareModifierMonitor(requestPermission: requestPermission)
    case "bareModifierMonitor.stop":
        return stopBareModifierMonitor()
    case "deviceKey.create":
        return try createDeviceKey(identifier: string(params["identifier"], "identifier"))
    case "deviceKey.read":
        return try readDeviceKey(identifier: string(params["identifier"], "identifier"))
    case "deviceKey.sign":
        return try signWithDeviceKey(
            identifier: string(params["identifier"], "identifier"),
            message: string(params["message"], "message")
        )
    case "deviceKey.delete":
        return try deleteDeviceKey(identifier: string(params["identifier"], "identifier"))
    case "localAuthentication.read":
        return readLocalAuthentication()
    case "localAuthentication.request":
        return try requestLocalAuthentication(reason: string(params["reason"], "reason"))
    case "bookmark.create":
        return try createBookmark(path: string(params["path"], "path"))
    case "bookmark.resolve":
        return try resolveBookmark(bookmark: string(params["bookmark"], "bookmark")).1
    case "bookmark.start":
        return try startBookmark(bookmark: string(params["bookmark"], "bookmark"))
    case "bookmark.stop":
        return try stopBookmark(token: string(params["token"], "token"))
    case "applicationGroup.read":
        let group = try string(params["identifier"], "identifier")
        guard group == "com.limecloud.lime" || group.hasPrefix("com.limecloud.lime.") else {
            throw HostError.invalidArgument("Application Group identifiers must use the Lime namespace.")
        }
        guard let containerURL = FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: group) else {
            return [
                "status": "not_configured",
                "reason": "The requested Application Group is not configured for this signed app.",
                "identifier": group,
                "path": NSNull(),
            ]
        }
        return [
            "status": "ready",
            "reason": "The Application Group container is available.",
            "identifier": group,
            "path": containerURL.path,
        ]
    default:
        throw HostError.invalidArgument("Unsupported macOS native host method: \(method)")
    }
}

func writeJSON(_ object: [String: Any]) {
    guard JSONSerialization.isValidJSONObject(object), let data = try? JSONSerialization.data(withJSONObject: object), let line = String(data: data, encoding: .utf8) else {
        return
    }
    if let data = (line + "\n").data(using: .utf8) {
        outputLock.lock()
        FileHandle.standardOutput.write(data)
        outputLock.unlock()
    }
}

func writeEvent(event: String, payload: [String: Any]) {
    writeJSON(["event": event, "payload": payload])
}

func processRequestLine(_ line: String) {
    guard let data = line.data(using: .utf8), let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        writeJSON(failure(id: NSNull(), code: "invalid_json", message: "Request is not valid JSON."))
        return
    }
    let id: Any = object["id"] ?? NSNull()
    do {
        let method = try string(object["method"], "method")
        let result = try handle(method: method, params: dictionary(object["params"]))
        writeJSON(response(id: id, result: result))
    } catch {
        let hostError = error as? HostError ?? .operationFailed(String(describing: error))
        let (code, message, details) = errorDetails(hostError)
        writeJSON(failure(id: id, code: code, message: message, data: details))
    }
}

@main
struct NativeHostMain {
    static func main() {
        DispatchQueue.main.async {
            DispatchQueue.global(qos: .userInitiated).async {
                while let line = readLine() {
                    let normalizedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
                    DispatchQueue.main.sync {
                        processRequestLine(normalizedLine)
                    }
                }
                DispatchQueue.main.sync {
                    _ = stopHIDTopologyWatcher()
                    _ = stopBareModifierMonitor()
                    _ = stopDisplayWatcher()
                    for taskID in (readHideForTasks()["tasks"] as? [String] ?? []) {
                        _ = try? stopHideForTask(taskID: taskID)
                    }
                    stopAllSecurityResources()
                    Darwin.exit(0)
                }
            }
        }
        RunLoop.main.run()
    }
}
