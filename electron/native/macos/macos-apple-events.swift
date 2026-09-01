import AppKit
import ApplicationServices
import Foundation

private let appleEventsSettingsURL =
    "x-apple.systempreferences:com.apple.preference.security?Privacy_Automation"

private let noErrStatus: Int32 = 0
private let errAEEventNotPermittedStatus: Int32 = -1743
private let errAEEventWouldRequireUserConsentStatus: Int32 = -1744

func openAppleEventsSettings() throws -> [String: Any] {
    guard let url = URL(string: appleEventsSettingsURL), NSWorkspace.shared.open(url) else {
        throw HostError.operationFailed("Unable to open macOS Automation privacy settings.")
    }
    return ["opened": true, "settingsUrl": appleEventsSettingsURL]
}

func readAppleEventsTargets() -> [String: Any] {
    let targets = NSWorkspace.shared.runningApplications.compactMap { application -> [String: Any]? in
        guard let bundleIdentifier = application.bundleIdentifier,
              !bundleIdentifier.isEmpty else {
            return nil
        }
        return [
            "bundleId": bundleIdentifier,
            "name": application.localizedName ?? bundleIdentifier,
            "processId": application.processIdentifier,
            "active": application.isActive,
            "hidden": application.isHidden,
            "terminated": application.isTerminated,
        ]
    }
    .sorted { left, right in
        let leftName = left["name"] as? String ?? ""
        let rightName = right["name"] as? String ?? ""
        return leftName.localizedCaseInsensitiveCompare(rightName) == .orderedAscending
    }
    return ["targets": targets]
}

private func appleEventsTargetBundleID(_ params: [String: Any]) throws -> String {
    let targetBundleID = try string(params["targetBundleId"], "targetBundleId")
    guard targetBundleID.count <= 255,
          targetBundleID.range(of: #"^[A-Za-z0-9][A-Za-z0-9.-]*$"#, options: .regularExpression) != nil else {
        throw HostError.invalidArgument("targetBundleId must be a valid bundle identifier.")
    }
    return targetBundleID
}

private func appleEventsStatus(
    targetBundleID: String,
    statusCode: Int32,
    askedUser: Bool
) -> [String: Any] {
    let status: String
    let reason: String
    switch statusCode {
    case noErrStatus:
        status = "ready"
        reason = "Apple Events automation permission is granted for the target application."
    case errAEEventNotPermittedStatus:
        status = "not_granted"
        reason = "Apple Events automation permission is denied for the target application."
    case errAEEventWouldRequireUserConsentStatus:
        status = "not_granted"
        reason = askedUser
            ? "Apple Events automation permission was not granted."
            : "Apple Events automation permission requires user consent."
    case Int32(procNotFound):
        status = "unavailable"
        reason = "The target application is not running."
    default:
        status = "unavailable"
        reason = "Apple Events permission query returned an unexpected system status."
    }
    return [
        "status": status,
        "reason": reason,
        "settingsUrl": appleEventsSettingsURL,
        "targetBundleId": targetBundleID,
        "eventClass": "*",
        "eventId": "*",
        "askedUser": askedUser,
        "statusCode": statusCode,
        "requiresUserConsent": statusCode == errAEEventWouldRequireUserConsentStatus,
    ]
}

func readAppleEventsPermission(params: [String: Any], askUser: Bool) throws -> [String: Any] {
    let targetBundleID = try appleEventsTargetBundleID(params)
    guard NSRunningApplication.runningApplications(withBundleIdentifier: targetBundleID).first != nil else {
        return appleEventsStatus(
            targetBundleID: targetBundleID,
            statusCode: Int32(procNotFound),
            askedUser: askUser
        )
    }

    var targetDescriptor = AEDesc()
    let bundleIDData = Data(targetBundleID.utf8)
    let createStatus: OSErr = bundleIDData.withUnsafeBytes { bytes in
        AECreateDesc(
            DescType(typeApplicationBundleID),
            bytes.baseAddress,
            bundleIDData.count,
            &targetDescriptor
        )
    }
    guard createStatus == noErrStatus else {
        throw HostError.operationFailed(
            "Apple Events target descriptor could not be created (status \(createStatus))."
        )
    }
    defer {
        AEDisposeDesc(&targetDescriptor)
    }

    let permissionStatus = AEDeterminePermissionToAutomateTarget(
        &targetDescriptor,
        AEEventClass(typeWildCard),
        AEEventID(typeWildCard),
        askUser
    )
    return appleEventsStatus(
        targetBundleID: targetBundleID,
        statusCode: permissionStatus,
        askedUser: askUser
    )
}

func readAppleEventsPermission(params: [String: Any]) throws -> [String: Any] {
    try readAppleEventsPermission(params: params, askUser: false)
}

func requestAppleEventsPermission(params: [String: Any]) throws -> [String: Any] {
    try readAppleEventsPermission(params: params, askUser: true)
}
