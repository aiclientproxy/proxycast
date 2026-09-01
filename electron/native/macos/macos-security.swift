import Foundation
import LocalAuthentication
import Security

var activeBookmarks: [String: URL] = [:]

func deviceKeyTag(identifier: String) throws -> Data {
    let normalized = try string(identifier, "identifier")
    guard normalized.range(of: "^[A-Za-z0-9._-]{1,96}$", options: .regularExpression) != nil else {
        throw HostError.invalidArgument("Device key identifier contains unsupported characters.")
    }
    return Data("com.limecloud.lime.device-key.\(normalized)".utf8)
}

func findSecureEnclavePrivateKey(identifier: String) throws -> SecKey {
    let tag = try deviceKeyTag(identifier: identifier)
    let query: [CFString: Any] = [
        kSecClass: kSecClassKey,
        kSecAttrApplicationTag: tag,
        kSecAttrKeyType: kSecAttrKeyTypeECSECPrimeRandom,
        kSecAttrTokenID: kSecAttrTokenIDSecureEnclave,
        kSecReturnRef: true,
    ]
    var item: CFTypeRef?
    let status = SecItemCopyMatching(query as CFDictionary, &item)
    guard status == errSecSuccess, let item else {
        throw HostError.unavailable("Secure Enclave device key is not available.")
    }
    return item as! SecKey
}

func createDeviceKey(identifier: String) throws -> [String: Any] {
    let tag = try deviceKeyTag(identifier: identifier)
    let attributes: [CFString: Any] = [
        kSecAttrKeyType: kSecAttrKeyTypeECSECPrimeRandom,
        kSecAttrKeySizeInBits: 256,
        kSecAttrTokenID: kSecAttrTokenIDSecureEnclave,
        kSecPrivateKeyAttrs: [
            kSecAttrIsPermanent: true,
            kSecAttrApplicationTag: tag,
        ],
    ]
    var error: Unmanaged<CFError>?
    guard let privateKey = SecKeyCreateRandomKey(attributes as CFDictionary, &error),
          let publicKey = SecKeyCopyPublicKey(privateKey),
          let publicData = SecKeyCopyExternalRepresentation(publicKey, &error) as Data?
    else {
        throw HostError.unavailable("Secure Enclave device key could not be created.")
    }
    return [
        "identifier": identifier,
        "created": true,
        "publicKey": publicData.base64EncodedString(),
    ]
}

func readDeviceKey(identifier: String) throws -> [String: Any] {
    let key = try findSecureEnclavePrivateKey(identifier: identifier)
    guard let publicKey = SecKeyCopyPublicKey(key) else {
        throw HostError.unavailable("Secure Enclave public key is not available.")
    }
    var error: Unmanaged<CFError>?
    guard let publicData = SecKeyCopyExternalRepresentation(publicKey, &error) as Data? else {
        throw HostError.unavailable("Secure Enclave public key could not be exported.")
    }
    return [
        "identifier": identifier,
        "exists": true,
        "publicKey": publicData.base64EncodedString(),
    ]
}

func signWithDeviceKey(identifier: String, message: String) throws -> [String: Any] {
    let key = try findSecureEnclavePrivateKey(identifier: identifier)
    guard let data = Data(base64Encoded: message) else {
        throw HostError.invalidArgument("Device key message is not valid base64.")
    }
    var error: Unmanaged<CFError>?
    guard let signature = SecKeyCreateSignature(
        key,
        .ecdsaSignatureMessageX962SHA256,
        data as CFData,
        &error
    ) as Data? else {
        throw HostError.operationFailed("Secure Enclave device key signing failed.")
    }
    return [
        "identifier": identifier,
        "signature": signature.base64EncodedString(),
    ]
}

func deleteDeviceKey(identifier: String) throws -> [String: Any] {
    let tag = try deviceKeyTag(identifier: identifier)
    let query: [CFString: Any] = [
        kSecClass: kSecClassKey,
        kSecAttrApplicationTag: tag,
        kSecAttrKeyType: kSecAttrKeyTypeECSECPrimeRandom,
        kSecAttrTokenID: kSecAttrTokenIDSecureEnclave,
    ]
    let status = SecItemDelete(query as CFDictionary)
    guard status == errSecSuccess || status == errSecItemNotFound else {
        throw HostError.operationFailed("Secure Enclave device key could not be deleted.")
    }
    return ["identifier": identifier, "deleted": true]
}

func readLocalAuthentication() -> [String: Any] {
    let context = LAContext()
    var error: NSError?
    let available = context.canEvaluatePolicy(.deviceOwnerAuthentication, error: &error)
    if available {
        return [
            "status": "ready",
            "policy": "deviceOwnerAuthentication",
            "reason": "Local Authentication is available on this Mac.",
        ]
    }
    var result: [String: Any] = [
        "status": "unavailable",
        "policy": "deviceOwnerAuthentication",
        "reason": error?.localizedDescription ?? "Local Authentication is unavailable.",
    ]
    if let errorCode = error?.code {
        result["errorCode"] = errorCode
    }
    return result
}

func requestLocalAuthentication(reason: String) throws -> [String: Any] {
    let prompt = try string(reason, "reason")
    let context = LAContext()
    var canEvaluateError: NSError?
    guard context.canEvaluatePolicy(.deviceOwnerAuthentication, error: &canEvaluateError) else {
        throw HostError.unavailable(
            canEvaluateError?.localizedDescription ?? "Local Authentication is unavailable."
        )
    }
    let semaphore = DispatchSemaphore(value: 0)
    var success = false
    var evaluationError: NSError?
    context.evaluatePolicy(.deviceOwnerAuthentication, localizedReason: prompt) { granted, error in
        success = granted
        evaluationError = error as NSError?
        semaphore.signal()
    }
    if semaphore.wait(timeout: .now() + 60) == .timedOut {
        throw HostError.operationFailed("Local Authentication request timed out.")
    }
    if success {
        return [
            "status": "ready",
            "policy": "deviceOwnerAuthentication",
            "authenticated": true,
        ]
    }
    var result: [String: Any] = [
        "status": "not_granted",
        "policy": "deviceOwnerAuthentication",
        "authenticated": false,
        "reason": evaluationError?.localizedDescription ?? "Local Authentication was not granted.",
    ]
    if let errorCode = evaluationError?.code {
        result["errorCode"] = errorCode
    }
    return result
}

func createBookmark(path: String) throws -> [String: Any] {
    let fileURL = URL(fileURLWithPath: path).standardizedFileURL
    let data = try fileURL.bookmarkData(
        options: .withSecurityScope,
        includingResourceValuesForKeys: nil,
        relativeTo: nil
    )
    return [
        "path": fileURL.path,
        "bookmark": data.base64EncodedString(),
    ]
}

func resolveBookmark(bookmark: String) throws -> (URL, [String: Any]) {
    guard let data = Data(base64Encoded: bookmark) else {
        throw HostError.invalidArgument("Bookmark data is not valid base64.")
    }
    var isStale = false
    let fileURL = try URL(
        resolvingBookmarkData: data,
        options: [.withSecurityScope, .withoutUI],
        relativeTo: nil,
        bookmarkDataIsStale: &isStale
    )
    return (
        fileURL,
        [
            "path": fileURL.path,
            "isStale": isStale,
        ]
    )
}

func startBookmark(bookmark: String) throws -> [String: Any] {
    let (fileURL, details) = try resolveBookmark(bookmark: bookmark)
    guard fileURL.startAccessingSecurityScopedResource() else {
        throw HostError.notGranted("The security-scoped bookmark could not be opened.")
    }
    let token = UUID().uuidString
    activeBookmarks[token] = fileURL
    return details.merging(["token": token, "started": true]) { _, next in next }
}

func stopBookmark(token: String) throws -> [String: Any] {
    guard let fileURL = activeBookmarks.removeValue(forKey: token) else {
        throw HostError.invalidArgument("Unknown security-scoped bookmark token.")
    }
    fileURL.stopAccessingSecurityScopedResource()
    return ["token": token, "stopped": true]
}

func stopAllSecurityResources() {
    for (_, fileURL) in activeBookmarks {
        fileURL.stopAccessingSecurityScopedResource()
    }
    activeBookmarks.removeAll()
}
