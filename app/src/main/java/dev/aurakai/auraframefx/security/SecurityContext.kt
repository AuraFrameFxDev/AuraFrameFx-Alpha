package dev.aurakai.auraframefx.security // Updated package name

// SecretKeyFactory, PBEKeySpec, SecretKeySpec are removed as they are related to PBKDF2
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import androidx.core.content.ContextCompat
import dagger.hilt.android.qualifiers.ApplicationContext
import dev.aurakai.auraframefx.model.AgentType
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.security.MessageDigest
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.spec.IvParameterSpec
import javax.inject.Inject
import javax.inject.Singleton

/**
 * SecurityContext manages the security aspects of the AuraFrameFx system.
 * This class is tied to the KAI agent persona and handles all security-related operations.
 */
@Singleton
class SecurityContext @Inject constructor(
    @ApplicationContext private val context: Context,
    private val keystoreManager: KeystoreManager, // Added KeystoreManager
) {
    companion object {
        private const val TAG = "SecurityContext"
        private const val THREAT_DETECTION_INTERVAL_MS = 30000L // 30 seconds

        // ENCRYPTION_ALGORITHM is defined in KeystoreManager as AES_MODE or can be a shared constant
        // SECRET_KEY_ALGORITHM, KEY_LENGTH, ITERATION_COUNT removed (PBKDF2 specific)
        private const val AES_ALGORITHM_WITH_PADDING =
            "AES/CBC/PKCS7Padding" // Re-added for direct Cipher.getInstance calls
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val _securityState = MutableStateFlow(SecurityState())
    val securityState: StateFlow<SecurityState> = _securityState.asStateFlow()

    private val _threatDetectionActive = MutableStateFlow(false)
    val threatDetectionActive: StateFlow<Boolean> = _threatDetectionActive.asStateFlow()

    private val _permissionsState = MutableStateFlow<Map<String, Boolean>>(emptyMap())
    val permissionsState: StateFlow<Map<String, Boolean>> = _permissionsState.asStateFlow()

    private val _encryptionStatus = MutableStateFlow(EncryptionStatus.NOT_INITIALIZED)
    val encryptionStatus: StateFlow<EncryptionStatus> = _encryptionStatus.asStateFlow()

    init {
        Log.d(TAG, "Security context initialized by KAI")
        updatePermissionsState()
    }

    /**
     * Start monitoring for security threats in the background
     */
    fun startThreatDetection() {
        if (_threatDetectionActive.value) return

        _threatDetectionActive.value = true
        scope.launch {
            while (_threatDetectionActive.value) {
                try {
                    val threats = detectThreats()
                    _securityState.value = _securityState.value.copy(
                        detectedThreats = threats,
                        threatLevel = calculateThreatLevel(threats),
                        lastScanTime = System.currentTimeMillis()
                    )
                    kotlinx.coroutines.delay(THREAT_DETECTION_INTERVAL_MS)
                } catch (e: Exception) {
                    Log.e(TAG, "Error in threat detection", e)
                    _threatDetectionActive.value = false
                    _securityState.value = _securityState.value.copy(
                        errorState = true,
                        errorMessage = "Threat detection error: ${e.message}"
                    )
                }
            }
        }
    }

    fun stopThreatDetection() {
        _threatDetectionActive.value = false
    }

    /**
     * Check if the app has the specified permission
     */
    fun hasPermission(permission: String): Boolean {
        return ContextCompat.checkSelfPermission(
            context,
            permission
        ) == PackageManager.PERMISSION_GRANTED
    }

    /**
     * Update the current state of all permissions relevant to the app
     */
    fun updatePermissionsState() {
        val permissionsToCheck = listOf(
            android.Manifest.permission.RECORD_AUDIO,
            android.Manifest.permission.CAMERA,
            android.Manifest.permission.ACCESS_FINE_LOCATION,
            android.Manifest.permission.READ_EXTERNAL_STORAGE,
            android.Manifest.permission.WRITE_EXTERNAL_STORAGE,
            android.Manifest.permission.INTERNET
        )

        _permissionsState.value = permissionsToCheck.associateWith { permission ->
            hasPermission(permission)
        }
    }

    /**
     * Initialize the encryption subsystem using Keystore.
     */
    fun initializeEncryption(): Boolean {
        Log.d(TAG, "Initializing encryption using KeystoreManager.")
        val secretKey = keystoreManager.getOrCreateSecretKey()
        return if (secretKey != null) {
            _encryptionStatus.value = EncryptionStatus.ACTIVE
            _securityState.value = _securityState.value.copy(
                errorState = false,
                errorMessage = "Encryption initialized successfully." // Informational message
            )
            Log.i(TAG, "Encryption initialized successfully using Keystore.")
            true
        } else {
            _encryptionStatus.value =
                EncryptionStatus.ERROR // Changed from INACTIVE to ERROR for clarity
            _securityState.value = _securityState.value.copy(
                errorState = true,
                errorMessage = "ERROR_KEY_INITIALIZATION_FAILED: Keystore key could not be created or retrieved."
            )
            Log.e(TAG, "Keystore key initialization failed.")
            false
        }
    }

    /**
     * Encrypt sensitive data using Keystore.
     */
    fun encrypt(data: String): EncryptedData? {
        if (_encryptionStatus.value != EncryptionStatus.ACTIVE) {
            Log.w(TAG, "Encryption not initialized. Attempting to initialize.")
            if (!initializeEncryption()) {
                Log.e(TAG, "Encryption initialization failed during encrypt call.")
                _securityState.value = _securityState.value.copy(
                    errorState = true,
                    errorMessage = "ERROR_ENCRYPTION_FAILED: Initialization failed."
                )
                return null
            }
        }

        try {
            val secretKey = keystoreManager.getOrCreateSecretKey()
            if (secretKey == null) {
                Log.e(TAG, "Failed to get secret key for encryption.")
                _securityState.value = _securityState.value.copy(
                    errorState = true,
                    errorMessage = "ERROR_ENCRYPTION_CIPHER_UNAVAILABLE: Secret key not available."
                )
                return null
            }

            val iv = ByteArray(16)
            SecureRandom().nextBytes(iv)
            val ivSpec = IvParameterSpec(iv)

            val cipher =
                Cipher.getInstance(AES_ALGORITHM_WITH_PADDING) // Using the defined constant
            cipher.init(Cipher.ENCRYPT_MODE, secretKey, ivSpec)

            val encryptedBytes = cipher.doFinal(data.toByteArray(Charsets.UTF_8))
            return EncryptedData(
                data = encryptedBytes,
                iv = iv,
                timestamp = System.currentTimeMillis(),
                metadata = "Encrypted by KAI Security (Keystore)"
            )
        } catch (e: Exception) { // Catch generic exceptions for robustness
            Log.e(TAG, "Encryption error", e)
            _securityState.value = _securityState.value.copy(
                errorState = true,
                errorMessage = "Encryption error: ${e.message}"
            )
            return null
        }
    }

    /**
     * Decrypt previously encrypted data using Keystore.
     */
    fun decrypt(encryptedData: EncryptedData): String? {
        if (_encryptionStatus.value != EncryptionStatus.ACTIVE) {
            Log.w(TAG, "Encryption not initialized. Attempting to initialize for decryption.")
            if (!initializeEncryption()) {
                Log.e(TAG, "Encryption initialization failed during decrypt call.")
                _securityState.value = _securityState.value.copy(
                    errorState = true,
                    errorMessage = "ERROR_DECRYPTION_FAILED: Initialization failed."
                )
                return null
            }
        }

        try {
            // KeystoreManager's getDecryptionCipher handles key retrieval and cipher init with IV
            val decryptionCipher = keystoreManager.getDecryptionCipher(encryptedData.iv)

            if (decryptionCipher == null) {
                Log.e(TAG, "Failed to get decryption cipher from KeystoreManager.")
                _securityState.value = _securityState.value.copy(
                    errorState = true,
                    errorMessage = "ERROR_DECRYPTION_CIPHER_UNAVAILABLE: Decryption cipher could not be initialized."
                )
                return null
            }

            val decryptedBytes = decryptionCipher.doFinal(encryptedData.data)
            return String(decryptedBytes, Charsets.UTF_8)
        } catch (e: Exception) { // Catch generic exceptions
            Log.e(TAG, "Decryption error", e)
            _securityState.value = _securityState.value.copy(
                errorState = true,
                errorMessage = "Decryption error: ${e.message}"
            )
            return null
        }
    }

    /**
     * Share a secure context with another agent
     */
    fun shareSecureContextWith(agentType: AgentType, context: String): SharedSecureContext {
        val secureId = generateSecureId()
        val timestamp = System.currentTimeMillis()

        return SharedSecureContext(
            id = secureId,
            originatingAgent = AgentType.KAI,
            targetAgent = agentType,
            encryptedContent = context.toByteArray(), // In production this would be encrypted
            timestamp = timestamp,
            expiresAt = timestamp + 3600000 // 1 hour expiry
        )
    }

    /**
     * Verify the integrity of the application
     */
    fun verifyApplicationIntegrity(): ApplicationIntegrity {
        try {
            // Get the app's package info
            val packageInfo = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                context.packageManager.getPackageInfo(
                    context.packageName,
                    PackageManager.PackageInfoFlags.of(PackageManager.GET_SIGNATURES.toLong())
                )
            } else {
                @Suppress("DEPRECATION")
                context.packageManager.getPackageInfo(
                    context.packageName,
                    PackageManager.GET_SIGNATURES
                )
            }

            // In a real app, we would verify the signature against a known good value
            val signatureBytes = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                packageInfo.signingInfo?.apkContentsSigners?.getOrNull(0)?.toByteArray()
            } else {
                @Suppress("DEPRECATION")
                packageInfo.signatures?.getOrNull(0)?.toByteArray()
            }
            if (signatureBytes == null) throw Exception("No signature found")

            val md = MessageDigest.getInstance("SHA-256")
            val signatureDigest = md.digest(signatureBytes)
            val signatureHex = signatureDigest.joinToString("") { "%02x".format(it) }

            // In a real implementation, we would compare against a known good signature
            val isValid = signatureHex.isNotEmpty()

            return ApplicationIntegrity(
                verified = isValid,
                appVersion = packageInfo.versionName ?: "unknown",
                signatureHash = signatureHex,
                installTime = packageInfo.firstInstallTime,
                lastUpdateTime = packageInfo.lastUpdateTime
            )
        } catch (e: Exception) {
            Log.e(TAG, "Application integrity verification error", e)
            return ApplicationIntegrity(
                verified = false,
                appVersion = "unknown",
                signatureHash = "error",
                installTime = 0,
                lastUpdateTime = 0,
                errorMessage = e.message
            )
        }
    }

    /**
     * Detect potential security threats
     * For the beta, this returns simulated threats
     */
    private fun detectThreats(): List<SecurityThreat> {
        // In a real implementation, this would perform actual threat analysis
        // For the beta, we return simulated threats for testing
        return listOf(
            SecurityThreat(
                id = "SIM-001",
                type = ThreatType.PERMISSION_ABUSE,
                severity = ThreatSeverity.LOW,
                description = "Simulated permission abuse threat for testing",
                detectedAt = System.currentTimeMillis()
            ),
            SecurityThreat(
                id = "SIM-002",
                type = ThreatType.NETWORK_VULNERABILITY,
                severity = ThreatSeverity.MEDIUM,
                description = "Simulated network vulnerability for testing",
                detectedAt = System.currentTimeMillis()
            )
        ).filter { Math.random() > 0.7 } // Randomly include some threats
    }

    /**
     * Calculate overall threat level based on detected threats
     */
    private fun calculateThreatLevel(threats: List<SecurityThreat>): ThreatLevel {
        if (threats.isEmpty()) return ThreatLevel.SAFE

        val hasCritical = threats.any { it.severity == ThreatSeverity.CRITICAL }
        val hasHigh = threats.any { it.severity == ThreatSeverity.HIGH }
        val hasMedium = threats.any { it.severity == ThreatSeverity.MEDIUM }

        return when {
            hasCritical -> ThreatLevel.CRITICAL
            hasHigh -> ThreatLevel.HIGH
            hasMedium -> ThreatLevel.MODERATE
            else -> ThreatLevel.LOW
        }
    }

    /**
     * Generate a secure ID for context sharing
     */
    private fun generateSecureId(): String {
        val bytes = ByteArray(16)
        SecureRandom().nextBytes(bytes)
        return bytes.joinToString("") { "%02x".format(it) }
    }

    /**
     * Log a security event
     */
    fun logSecurityEvent(event: SecurityEvent) {
        scope.launch {
            Log.d(
                TAG,
                "Security event logged: " + Json.encodeToString(SecurityEvent.serializer(), event)
            )
            // In a real implementation, this would store events securely
        }
    }
}

/**
 * Represents the current security state
 */
@Serializable
data class SecurityState(
    val detectedThreats: List<SecurityThreat> = emptyList(),
    val threatLevel: ThreatLevel = ThreatLevel.UNKNOWN,
    val lastScanTime: Long = 0,
    val errorState: Boolean = false,
    val errorMessage: String? = null,
)

/**
 * Represents a security threat detected by KAI
 */
@Serializable
data class SecurityThreat(
    val id: String,
    val type: ThreatType,
    val severity: ThreatSeverity,
    val description: String,
    val detectedAt: Long,
)

/**
 * Types of security threats
 */
enum class ThreatType {
    MALWARE,
    NETWORK_VULNERABILITY,
    PERMISSION_ABUSE,
    DATA_LEAK,
    UNKNOWN
}

/**
 * Severity levels for security threats
 */
enum class ThreatSeverity {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL
}

/**
 * Overall threat levels for the system
 */
enum class ThreatLevel {
    SAFE,
    LOW,
    MODERATE,
    HIGH,
    CRITICAL,
    UNKNOWN
}

/**
 * Status of the encryption subsystem
 */
enum class EncryptionStatus {
    NOT_INITIALIZED,
    ACTIVE,
    DISABLED,
    ERROR
}

/**
 * Data class for encrypted information
 */
@Serializable
data class EncryptedData(
    val data: ByteArray,
    // val salt: ByteArray, // Removed salt field
    val iv: ByteArray,
    val timestamp: Long,
    val metadata: String? = null,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as EncryptedData

        if (!data.contentEquals(other.data)) return false
        // if (!salt.contentEquals(other.salt)) return false // Removed salt comparison
        if (!iv.contentEquals(other.iv)) return false
        if (timestamp != other.timestamp) return false
        if (metadata != other.metadata) return false

        return true
    }

    override fun hashCode(): Int {
        var result = data.contentHashCode()
        // result = 31 * result + salt.contentHashCode() // Removed salt from hashCode
        result = 31 * result + iv.contentHashCode()
        result = 31 * result + timestamp.hashCode()
        result = 31 * result + (metadata?.hashCode() ?: 0)
        return result
    }
}

/**
 * Data class for application integrity information
 */
@Serializable
data class ApplicationIntegrity(
    val verified: Boolean,
    val appVersion: String,
    val signatureHash: String,
    val installTime: Long,
    val lastUpdateTime: Long,
    val errorMessage: String? = null,
)

/**
 * Data class for security events to be logged
 */
@Serializable
data class SecurityEvent(
    val id: String = java.util.UUID.randomUUID().toString(),
    val type: SecurityEventType,
    val timestamp: Long = System.currentTimeMillis(),
    val details: String,
    val severity: EventSeverity,
)

/**
 * Types of security events
 */
enum class SecurityEventType {
    PERMISSION_CHANGE,
    THREAT_DETECTED,
    ENCRYPTION_EVENT,
    AUTHENTICATION_EVENT,
    INTEGRITY_CHECK
}

/**
 * Severity levels for security events
 */
enum class EventSeverity {
    INFO,
    WARNING,
    ERROR,
    CRITICAL
}

/**
 * Data class for secure context sharing between agents
 */
@Serializable
data class SharedSecureContext(
    val id: String,
    val originatingAgent: AgentType,
    val targetAgent: AgentType,
    val encryptedContent: ByteArray,
    val timestamp: Long,
    val expiresAt: Long,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as SharedSecureContext

        if (id != other.id) return false
        if (originatingAgent != other.originatingAgent) return false
        if (targetAgent != other.targetAgent) return false
        if (!encryptedContent.contentEquals(other.encryptedContent)) return false
        if (timestamp != other.timestamp) return false
        if (expiresAt != other.expiresAt) return false

        return true
    }

    override fun hashCode(): Int {
        var result = id.hashCode()
        result = 31 * result + originatingAgent.hashCode()
        result = 31 * result + targetAgent.hashCode()
        result = 31 * result + encryptedContent.contentHashCode()
        result = 31 * result + timestamp.hashCode()
        result = 31 * result + expiresAt.hashCode()
        return result
    }
}
