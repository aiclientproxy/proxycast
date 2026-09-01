import AVFoundation
import Foundation

private let microphoneSettingsURL =
    "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
private let cameraSettingsURL =
    "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"

private func mediaPermissionStatus(
    _ mediaType: AVMediaType,
    settingsURL: String,
    label: String
) -> [String: Any] {
    let authorization = AVCaptureDevice.authorizationStatus(for: mediaType)
    let status: String
    let reason: String
    switch authorization {
    case .authorized:
        status = "ready"
        reason = "macOS \(label) permission is granted."
    case .denied:
        status = "not_granted"
        reason = "macOS \(label) permission is denied."
    case .restricted:
        status = "unavailable"
        reason = "macOS \(label) permission is restricted by the system or policy."
    case .notDetermined:
        status = "not_granted"
        reason = "macOS \(label) permission has not been requested."
    @unknown default:
        status = "unavailable"
        reason = "macOS \(label) permission returned an unknown status."
    }
    return [
        "status": status,
        "reason": reason,
        "settingsUrl": settingsURL,
        "mediaType": mediaType.rawValue,
    ]
}
func readMediaPermissions() -> [String: Any] {
    [
        "microphone": mediaPermissionStatus(
            .audio,
            settingsURL: microphoneSettingsURL,
            label: "microphone"
        ),
        "camera": mediaPermissionStatus(
            .video,
            settingsURL: cameraSettingsURL,
            label: "camera"
        ),
    ]
}

func requestMediaPermission(kind: String) throws -> [String: Any] {
    let normalizedKind = kind.lowercased()
    let mediaType: AVMediaType
    let settingsURL: String
    let label: String
    switch normalizedKind {
    case "microphone", "audio":
        mediaType = .audio
        settingsURL = microphoneSettingsURL
        label = "microphone"
    case "camera", "video":
        mediaType = .video
        settingsURL = cameraSettingsURL
        label = "camera"
    default:
        throw HostError.invalidArgument("Media permission kind must be microphone or camera.")
    }

    let current = AVCaptureDevice.authorizationStatus(for: mediaType)
    if current == .notDetermined {
        let semaphore = DispatchSemaphore(value: 0)
        AVCaptureDevice.requestAccess(for: mediaType) { _ in
            semaphore.signal()
        }
        if semaphore.wait(timeout: .now() + 60) == .timedOut {
            throw HostError.operationFailed("macOS \(label) permission request timed out.")
        }
    }
    return mediaPermissionStatus(mediaType, settingsURL: settingsURL, label: label)
}
