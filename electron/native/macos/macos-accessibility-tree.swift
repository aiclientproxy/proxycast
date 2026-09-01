import AppKit
import ApplicationServices
import CoreGraphics
import Foundation

private let accessibilityTreeDefaultDepth = 8
private let accessibilityTreeMaximumDepth = 32
private let accessibilityTreeDefaultNodes = 2_000
private let accessibilityTreeMaximumNodes = 10_000
private let accessibilityTreeMaximumText = 512

private struct AccessibilityTreeBudget {
    let maximumDepth: Int
    let maximumNodes: Int
    var nodes = 0
    var truncated = false
}
private func boundedText(_ value: CFTypeRef?) -> String? {
    guard let value else {
        return nil
    }
    let text: String
    if let string = value as? String {
        text = string
    } else if let number = value as? NSNumber {
        text = number.stringValue
    } else {
        return nil
    }
    guard !text.isEmpty else {
        return nil
    }
    if text.count <= accessibilityTreeMaximumText {
        return text
    }
    return String(text.prefix(accessibilityTreeMaximumText))
}

private func accessibilityAttribute(_ element: AXUIElement, _ key: CFString) -> CFTypeRef? {
    var value: CFTypeRef?
    guard AXUIElementCopyAttributeValue(element, key, &value) == .success else {
        return nil
    }
    return value
}

private func accessibilityFrame(_ element: AXUIElement) -> CGRect? {
    guard let positionValue = accessibilityAttribute(element, kAXPositionAttribute as CFString),
          let sizeValue = accessibilityAttribute(element, kAXSizeAttribute as CFString),
          CFGetTypeID(positionValue) == AXValueGetTypeID(),
          CFGetTypeID(sizeValue) == AXValueGetTypeID() else {
        return nil
    }
    let positionAXValue = positionValue as! AXValue
    let sizeAXValue = sizeValue as! AXValue
    var position = CGPoint.zero
    var size = CGSize.zero
    guard AXValueGetValue(positionAXValue, .cgPoint, &position),
          AXValueGetValue(sizeAXValue, .cgSize, &size),
          size.width >= 0,
          size.height >= 0 else {
        return nil
    }
    return CGRect(origin: position, size: size)
}

private func accessibilityNode(
    _ element: AXUIElement,
    path: String,
    depth: Int,
    budget: inout AccessibilityTreeBudget
) -> [String: Any] {
    if budget.nodes >= budget.maximumNodes {
        budget.truncated = true
        return ["path": path, "truncated": true]
    }
    budget.nodes += 1

    var node: [String: Any] = [
        "path": path,
        "depth": depth,
    ]
    let textAttributes: [(String, CFString)] = [
        ("role", kAXRoleAttribute as CFString),
        ("subrole", kAXSubroleAttribute as CFString),
        ("title", kAXTitleAttribute as CFString),
        ("description", kAXDescriptionAttribute as CFString),
        ("value", kAXValueAttribute as CFString),
        ("identifier", kAXIdentifierAttribute as CFString),
    ]
    for (name, key) in textAttributes {
        if let text = boundedText(accessibilityAttribute(element, key)) {
            node[name] = text
        }
    }
    if let enabled = accessibilityAttribute(element, kAXEnabledAttribute as CFString) as? NSNumber {
        node["enabled"] = enabled.boolValue
    }
    if let focused = accessibilityAttribute(element, kAXFocusedAttribute as CFString) as? NSNumber {
        node["focused"] = focused.boolValue
    }
    if let frame = accessibilityFrame(element) {
        node["frame"] = [
            "x": frame.origin.x,
            "y": frame.origin.y,
            "width": frame.size.width,
            "height": frame.size.height,
        ]
    }

    guard depth < budget.maximumDepth,
          let children = accessibilityAttribute(element, kAXChildrenAttribute as CFString) as? [AXUIElement],
          !children.isEmpty else {
        return node
    }
    var childNodes: [[String: Any]] = []
    for (index, child) in children.enumerated() {
        if budget.nodes >= budget.maximumNodes {
            budget.truncated = true
            break
        }
        childNodes.append(
            accessibilityNode(
                child,
                path: "\(path).\(index)",
                depth: depth + 1,
                budget: &budget
            )
        )
    }
    if !childNodes.isEmpty {
        node["children"] = childNodes
    }
    if childNodes.count < children.count {
        budget.truncated = true
    }
    return node
}

func readAccessibilityTree(windowID: UInt32, params: [String: Any]) throws -> [String: Any] {
    guard AXIsProcessTrusted() else {
        throw HostError.notGranted("macOS Accessibility permission is required for the accessibility tree.")
    }
    let requestedDepth = (params["maxDepth"] as? NSNumber)?.intValue ?? accessibilityTreeDefaultDepth
    let requestedNodes = (params["maxNodes"] as? NSNumber)?.intValue ?? accessibilityTreeDefaultNodes
    guard (0...accessibilityTreeMaximumDepth).contains(requestedDepth) else {
        throw HostError.invalidArgument("Accessibility tree maxDepth must be between 0 and 32.")
    }
    guard (1...accessibilityTreeMaximumNodes).contains(requestedNodes) else {
        throw HostError.invalidArgument("Accessibility tree maxNodes must be between 1 and 10000.")
    }

    let ownerPID = try windowOwnerPID(windowID: windowID)
    let frame = try windowBounds(windowID: windowID)
    let root = try accessibilityWindowMatching(ownerPID: ownerPID, frame: frame)
    var budget = AccessibilityTreeBudget(
        maximumDepth: requestedDepth,
        maximumNodes: requestedNodes
    )
    let tree = accessibilityNode(root, path: "0", depth: 0, budget: &budget)
    return [
        "windowId": windowID,
        "ownerPid": ownerPID,
        "maxDepth": requestedDepth,
        "maxNodes": requestedNodes,
        "nodeCount": budget.nodes,
        "truncated": budget.truncated,
        "tree": tree,
    ]
}
