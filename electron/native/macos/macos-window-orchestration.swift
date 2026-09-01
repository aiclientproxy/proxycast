import AppKit
import ApplicationServices
import CoreGraphics
import Foundation

private let hideForTaskLock = NSLock()
private var hideForTaskStates: [String: [pid_t: Bool]] = [:]

private func finiteDouble(_ value: Any?, _ name: String, defaultValue: Double? = nil) throws -> Double {
    guard let value else {
        if let defaultValue {
            return defaultValue
        }
        throw HostError.invalidArgument("Missing required numeric field: \(name)")
    }
    guard let number = value as? NSNumber, number.doubleValue.isFinite else {
        throw HostError.invalidArgument("Missing or invalid numeric field: \(name)")
    }
    return number.doubleValue
}

func taskIdentifier(_ value: Any?) throws -> String {
    let identifier = try string(value, "taskId")
    guard identifier.range(of: "^[A-Za-z0-9._-]{1,96}$", options: .regularExpression) != nil else {
        throw HostError.invalidArgument("Task identifiers contain unsupported characters.")
    }
    return identifier
}

func windowIDList(_ value: Any?, _ name: String = "windowIds") throws -> [UInt32] {
    guard let values = value as? [Any], !values.isEmpty, values.count <= 64 else {
        throw HostError.invalidArgument("\(name) must contain between 1 and 64 window IDs.")
    }
    var result: [UInt32] = []
    var seen = Set<UInt32>()
    for value in values {
        let windowID = try uint32(value, name)
        if seen.insert(windowID).inserted {
            result.append(windowID)
        }
    }
    return result
}

func windowBounds(windowID: UInt32) throws -> CGRect {
    let windowInfo = CGWindowListCopyWindowInfo(
        [.optionIncludingWindow],
        CGWindowID(windowID)
    ) as? [[String: Any]] ?? []
    guard let info = windowInfo.first,
          let bounds = info[kCGWindowBounds as String] as? [String: Any],
          let x = (bounds["X"] as? NSNumber)?.doubleValue,
          let y = (bounds["Y"] as? NSNumber)?.doubleValue,
          let width = (bounds["Width"] as? NSNumber)?.doubleValue,
          let height = (bounds["Height"] as? NSNumber)?.doubleValue,
          width > 0,
          height > 0 else {
        throw HostError.unavailable("The requested macOS window bounds are unavailable.")
    }
    return CGRect(x: x, y: y, width: width, height: height)
}

private func framePayload(_ frame: CGRect) -> [String: Any] {
    [
        "x": frame.origin.x,
        "y": frame.origin.y,
        "width": frame.size.width,
        "height": frame.size.height,
    ]
}

func accessibilityWindowMatching(ownerPID: pid_t, frame: CGRect) throws -> AXUIElement {
    guard AXIsProcessTrusted() else {
        throw HostError.notGranted("macOS Accessibility permission is required for window control.")
    }
    let application = AXUIElementCreateApplication(ownerPID)
    var value: CFTypeRef?
    guard AXUIElementCopyAttributeValue(application, kAXWindowsAttribute as CFString, &value) == .success,
          let windows = value as? [AXUIElement],
          !windows.isEmpty else {
        throw HostError.unavailable("The requested macOS window has no Accessibility element.")
    }
    var closest: (window: AXUIElement, distance: CGFloat)?
    for window in windows {
        var positionValue: CFTypeRef?
        var sizeValue: CFTypeRef?
        guard AXUIElementCopyAttributeValue(window, kAXPositionAttribute as CFString, &positionValue) == .success,
              AXUIElementCopyAttributeValue(window, kAXSizeAttribute as CFString, &sizeValue) == .success,
              let positionValue,
              let sizeValue,
              CFGetTypeID(positionValue) == AXValueGetTypeID(),
              CFGetTypeID(sizeValue) == AXValueGetTypeID() else {
            continue
        }
        let positionAXValue = positionValue as! AXValue
        let sizeAXValue = sizeValue as! AXValue
        var position = CGPoint.zero
        var size = CGSize.zero
        guard AXValueGetValue(positionAXValue, .cgPoint, &position),
              AXValueGetValue(sizeAXValue, .cgSize, &size) else {
            continue
        }
        let candidate = CGRect(origin: position, size: size)
        let distance = abs(candidate.minX - frame.minX)
            + abs(candidate.minY - frame.minY)
            + abs(candidate.width - frame.width)
            + abs(candidate.height - frame.height)
        if closest == nil || distance < closest!.distance {
            closest = (window, distance)
        }
    }
    guard let closest else {
        throw HostError.unavailable("The requested macOS window could not be matched to an Accessibility element.")
    }
    return closest.window
}

func anchorWindow(windowID: UInt32, anchorWindowID: UInt32, params: [String: Any]) throws -> [String: Any] {
    guard windowID != anchorWindowID else {
        throw HostError.invalidArgument("A window cannot be anchored to itself.")
    }
    let targetFrame = try windowBounds(windowID: windowID)
    let anchorFrame = try windowBounds(windowID: anchorWindowID)
    let edge = (params["edge"] as? String ?? "bottom").lowercased()
    guard ["top", "bottom", "left", "right"].contains(edge) else {
        throw HostError.invalidArgument("Window anchor edge must be top, bottom, left or right.")
    }
    let alignment = (params["alignment"] as? String ?? "start").lowercased()
    guard ["start", "center", "end"].contains(alignment) else {
        throw HostError.invalidArgument("Window anchor alignment must be start, center or end.")
    }
    let gap = try finiteDouble(params["gap"], "gap", defaultValue: 8)
    guard (0...4096).contains(gap) else {
        throw HostError.invalidArgument("Window anchor gap must be between 0 and 4096 points.")
    }
    let gapPoints = CGFloat(gap)

    var frame = targetFrame
    switch edge {
    case "top":
        frame.origin.y = anchorFrame.maxY + gapPoints
        frame.origin.x = alignedOrigin(
            alignment: alignment,
            start: anchorFrame.minX,
            center: anchorFrame.midX,
            end: anchorFrame.maxX,
            extent: targetFrame.width
        )
    case "bottom":
        frame.origin.y = anchorFrame.minY - targetFrame.height - gapPoints
        frame.origin.x = alignedOrigin(
            alignment: alignment,
            start: anchorFrame.minX,
            center: anchorFrame.midX,
            end: anchorFrame.maxX,
            extent: targetFrame.width
        )
    case "left":
        frame.origin.x = anchorFrame.minX - targetFrame.width - gapPoints
        frame.origin.y = alignedOrigin(
            alignment: alignment,
            start: anchorFrame.minY,
            center: anchorFrame.midY,
            end: anchorFrame.maxY,
            extent: targetFrame.height
        )
    case "right":
        frame.origin.x = anchorFrame.maxX + gapPoints
        frame.origin.y = alignedOrigin(
            alignment: alignment,
            start: anchorFrame.minY,
            center: anchorFrame.midY,
            end: anchorFrame.maxY,
            extent: targetFrame.height
        )
    default:
        throw HostError.invalidArgument("Window anchor edge is unsupported.")
    }

    let ownerPID = try windowOwnerPID(windowID: windowID)
    let window = try accessibilityWindowMatching(ownerPID: ownerPID, frame: targetFrame)
    var position = frame.origin
    guard let positionValue = AXValueCreate(.cgPoint, &position),
          AXUIElementSetAttributeValue(window, kAXPositionAttribute as CFString, positionValue) == .success else {
        throw HostError.operationFailed("The anchored macOS window position could not be updated.")
    }
    return [
        "anchored": true,
        "windowId": windowID,
        "anchorWindowId": anchorWindowID,
        "edge": edge,
        "alignment": alignment,
        "gap": gap,
        "frame": framePayload(frame),
    ]
}

private func alignedOrigin(alignment: String, start: CGFloat, center: CGFloat, end: CGFloat, extent: CGFloat) -> CGFloat {
    switch alignment {
    case "center":
        return center - (extent / 2)
    case "end":
        return end - extent
    default:
        return start
    }
}

func stackWindows(windowIDs: [UInt32]) throws -> [String: Any] {
    guard AXIsProcessTrusted() else {
        throw HostError.notGranted("macOS Accessibility permission is required for window stacking.")
    }
    var raised: [UInt32] = []
    for windowID in windowIDs.reversed() {
        _ = try raiseWindow(windowID: windowID)
        raised.append(windowID)
    }
    return [
        "stacked": true,
        "order": windowIDs,
        "raisedOrder": raised,
    ]
}

func startHideForTask(taskID: String, windowIDs: [UInt32]) throws -> [String: Any] {
    hideForTaskLock.lock()
    defer { hideForTaskLock.unlock() }
    if hideForTaskStates[taskID] != nil {
        return [
            "taskId": taskID,
            "started": true,
            "alreadyRunning": true,
        ]
    }

    var originalVisibility: [pid_t: Bool] = [:]
    var changedApplications: [NSRunningApplication] = []
    do {
        for windowID in windowIDs {
            let ownerPID = try windowOwnerPID(windowID: windowID)
            guard let application = NSRunningApplication(processIdentifier: ownerPID) else {
                throw HostError.unavailable("The requested macOS window owner is no longer running.")
            }
            if originalVisibility[ownerPID] == nil {
                originalVisibility[ownerPID] = application.isHidden
                if !application.isHidden {
                    guard application.hide() else {
                        throw HostError.operationFailed("The macOS window owner could not be hidden for the task.")
                    }
                    changedApplications.append(application)
                }
            }
        }
    } catch {
        for application in changedApplications {
            _ = application.unhide()
        }
        throw error
    }
    hideForTaskStates[taskID] = originalVisibility
    return [
        "taskId": taskID,
        "started": true,
        "alreadyRunning": false,
        "hiddenCount": originalVisibility.count,
        "windowIds": windowIDs,
    ]
}

func stopHideForTask(taskID: String) throws -> [String: Any] {
    hideForTaskLock.lock()
    guard let originalVisibility = hideForTaskStates.removeValue(forKey: taskID) else {
        hideForTaskLock.unlock()
        throw HostError.invalidArgument("Unknown macOS hide-for-task identifier.")
    }
    hideForTaskLock.unlock()

    var restoredCount = 0
    for (ownerPID, wasHidden) in originalVisibility {
        guard let application = NSRunningApplication(processIdentifier: ownerPID) else {
            continue
        }
        let success = wasHidden ? application.hide() : application.unhide()
        if success {
            restoredCount += 1
        }
    }
    return [
        "taskId": taskID,
        "stopped": true,
        "restoredCount": restoredCount,
        "ownerCount": originalVisibility.count,
    ]
}

func readHideForTasks() -> [String: Any] {
    hideForTaskLock.lock()
    let taskIDs = hideForTaskStates.keys.sorted()
    hideForTaskLock.unlock()
    return ["tasks": taskIDs]
}
