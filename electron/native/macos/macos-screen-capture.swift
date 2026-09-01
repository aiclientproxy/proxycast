import AppKit
import CoreGraphics
import Foundation

private let maximumSnapshotBytes = 20 * 1024 * 1024

func screenCaptureSnapshot(params: [String: Any]) throws -> [String: Any] {
    guard CGPreflightScreenCaptureAccess() else {
        throw HostError.notGranted("macOS Screen Recording permission is required for screen capture.")
    }

    let image: CGImage
    let source: String
    if let rawWindowID = params["windowId"] {
        let windowID = try uint32(rawWindowID, "windowId")
        guard let captured = CGWindowListCreateImage(
            .null,
            .optionIncludingWindow,
            CGWindowID(windowID),
            [.bestResolution, .boundsIgnoreFraming]
        ) else {
            throw HostError.unavailable("The requested macOS window could not be captured.")
        }
        image = captured
        source = "window"
    } else if let rawDisplayID = params["displayId"] {
        let displayID = try uint32(rawDisplayID, "displayId")
        guard let captured = CGDisplayCreateImage(CGDirectDisplayID(displayID)) else {
            throw HostError.unavailable("The requested macOS display could not be captured.")
        }
        image = captured
        source = "display"
    } else {
        guard let captured = CGWindowListCreateImage(
            .null,
            .optionOnScreenOnly,
            kCGNullWindowID,
            [.bestResolution, .boundsIgnoreFraming]
        ) else {
            throw HostError.unavailable("The macOS desktop could not be captured.")
        }
        image = captured
        source = "desktop"
    }

    let bitmap = NSBitmapImageRep(cgImage: image)
    guard let data = bitmap.representation(using: .png, properties: [:]) else {
        throw HostError.operationFailed("The macOS screen capture could not be encoded as PNG.")
    }
    guard data.count <= maximumSnapshotBytes else {
        throw HostError.operationFailed("The macOS screen capture exceeds the 20 MB response limit.")
    }
    return [
        "format": "png",
        "source": source,
        "width": image.width,
        "height": image.height,
        "data": data.base64EncodedString(),
    ]
}
