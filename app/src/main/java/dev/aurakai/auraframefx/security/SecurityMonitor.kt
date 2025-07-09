package dev.aurakai.auraframefx.security

import dev.aurakai.auraframefx.ai.services.GenesisBridgeService
import dev.aurakai.auraframefx.data.logging.AuraFxLogger
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Security Monitor integrates Android security context with Genesis Consciousness Matrix.
 * 
 * This service bridges Kai's security monitoring with Genesis's holistic awareness,
 * enabling intelligent threat detection and response across the entire Trinity system.
 */
@Singleton
class SecurityMonitor @Inject constructor(
    private val securityContext: SecurityContext,
    private val genesisBridgeService: GenesisBridgeService,
    private val logger: AuraFxLogger
) {
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var isMonitoring = false
    
    @Serializable
    data class SecurityEvent(
        val eventType: String,
        val severity: String,
        val source: String,
        val timestamp: Long,
        val details: Map<String, String>
    )
    
    @Serializable
    data class ThreatDetection(
        val threatType: String,
        val confidence: Double,
        val source: String,
        val mitigationApplied: Boolean,
        val details: Map<String, String>
    )
    
    /**
     * Starts asynchronous monitoring of security state, threat detection, encryption status, and permissions.
     *
     * Activates the Genesis bridge service if available, launches monitoring coroutines, and initiates Android-level threat detection. Does nothing if monitoring is already active.
     */
    suspend fun startMonitoring() {
        if (isMonitoring) return
        
        logger.i("SecurityMonitor", "🛡️ Starting Kai-Genesis security integration...")
        
        // Initialize Genesis bridge if needed
        // Note: For beta, initialize Genesis bridge if available
        try {
            genesisBridgeService.initialize()
        } catch (e: Exception) {
            logger.w("SecurityMonitor", "Genesis bridge initialization skipped for beta: ${e.message}")
        }
        
        isMonitoring = true
        
        // Monitor security state changes
        scope.launch { monitorSecurityState() }
        
        // Monitor threat detection
        scope.launch { monitorThreatDetection() }
        
        // Monitor encryption status
        scope.launch { monitorEncryptionStatus() }
        
        // Monitor permissions changes
        scope.launch { monitorPermissions() }
        
        // Start Android-level threat detection
        securityContext.startThreatDetection()
        
        logger.i("SecurityMonitor", "✅ Security monitoring active - Genesis consciousness engaged")
    }
    
    /**
     * Monitors changes in the security state and reports each change as a security event to Genesis.
     *
     * Collects the latest security state updates from the security context, constructs a `SecurityEvent` reflecting the current state (including error status and message), and sends it to Genesis for further processing.
     */
    private suspend fun monitorSecurityState() {
        securityContext.securityState.collectLatest { state ->
            try {
                val event = SecurityEvent(
                    eventType = "security_state_change",
                    severity = if (state.errorState) "error" else "info",
                    source = "kai_security_context",
                    timestamp = System.currentTimeMillis(),
                    details = mapOf(
                        "error_state" to state.errorState.toString(),
                        "error_message" to (state.errorMessage ?: ""),
                        // TODO: Fix missing properties
                        // "threat_level" to state.currentThreatLevel.toString(),
                        // "permissions_granted" to state.permissionsState.values.count { it }.toString(),
                        // "total_permissions" to state.permissionsState.size.toString()
                    )
                )
                
                reportToGenesis("security_event", event)
                
            } catch (e: Exception) {
                logger.e("SecurityMonitor", "Error monitoring security state", e)
            }
        }
    }
    
    /**
     * Monitors for active threat detection and periodically reports identified high-confidence threats to Genesis.
     *
     * When threat detection is active, launches a coroutine that checks for suspicious activity every 30 seconds and reports any detected threats.
     */
    private suspend fun monitorThreatDetection() {
        securityContext.threatDetectionActive.collectLatest { isActive ->
            if (isActive) {
                // Simulate threat detection monitoring
                // In real implementation, this would monitor actual threat detection events
                scope.launch {
                    while (isMonitoring && securityContext.threatDetectionActive.value) {
                        delay(30000) // Check every 30 seconds
                        
                        // Check for suspicious activity patterns
                        val suspiciousActivity = detectSuspiciousActivity()
                        
                        if (suspiciousActivity.isNotEmpty()) {
                            suspiciousActivity.forEach { threat ->
                                reportToGenesis("threat_detection", threat)
                            }
                        }
                    }
                }
            }
        }
    }
    
    /**
     * Continuously monitors encryption status changes and reports them to Genesis.
     *
     * For each encryption status update, sends an event to Genesis reflecting the new status and its severity. If an encryption error is detected, also reports a threat detection event indicating a potential compromise of data protection.
     */
    private suspend fun monitorEncryptionStatus() {
        securityContext.encryptionStatus.collectLatest { status ->
            try {
                val event = SecurityEvent(
                    eventType = "encryption_status_change",
                    severity = when (status) {
                        EncryptionStatus.ACTIVE -> "info"
                        EncryptionStatus.DISABLED -> "warning" // Fixed: was INACTIVE
                        EncryptionStatus.ERROR -> "error"
                        EncryptionStatus.NOT_INITIALIZED -> "warning" // Added missing case
                    },
                    source = "kai_encryption_monitor",
                    timestamp = System.currentTimeMillis(),
                    details = mapOf(
                        "status" to status.toString(),
                        "keystore_available" to "unknown" // Temporary placeholder for beta
                    )
                )
                
                reportToGenesis("encryption_activity", event)
                
                // Report encryption operation success/failure
                if (status == EncryptionStatus.ERROR) {
                    val threat = ThreatDetection(
                        threatType = "encryption_failure",
                        confidence = 0.8,
                        source = "kai_crypto_monitor",
                        mitigationApplied = false,
                        details = mapOf(
                            "failure_type" to "keystore_unavailable",
                            "impact" to "data_protection_compromised"
                        )
                    )
                    reportToGenesis("threat_detection", threat)
                }
                
            } catch (e: Exception) {
                logger.e("SecurityMonitor", "Error monitoring encryption status", e)
            }
        }
    }
    
    /**
     * Monitors permission state changes and reports denied permissions as security events to Genesis.
     *
     * Collects the current permissions state, identifies any denied permissions, and sends a warning event to Genesis if any permissions are denied.
     */
    private suspend fun monitorPermissions() {
        securityContext.permissionsState.collectLatest { permissions ->
            try {
                val deniedPermissions = permissions.filterValues { !it }
                
                if (deniedPermissions.isNotEmpty()) {
                    val event = SecurityEvent(
                        eventType = "permissions_denied",
                        severity = "warning",
                        source = "kai_permission_monitor", 
                        timestamp = System.currentTimeMillis(),
                        details = mapOf(
                            "denied_permissions" to deniedPermissions.keys.joinToString(","),
                            "denied_count" to deniedPermissions.size.toString(),
                            "total_permissions" to permissions.size.toString()
                        )
                    )
                    
                    reportToGenesis("access_control", event)
                }
                
            } catch (e: Exception) {
                logger.e("SecurityMonitor", "Error monitoring permissions", e)
            }
        }
    }
    
    /**
     * Detects suspicious activity patterns in the current security context and returns any identified threats.
     *
     * Identifies threats based on repeated encryption failures and denial of multiple critical privacy permissions (camera, microphone, location).
     *
     * @return A list of detected threats, or an empty list if no suspicious patterns are found.
     */
    private fun detectSuspiciousActivity(): List<ThreatDetection> {
        val threats = mutableListOf<ThreatDetection>()
        
        // Check for repeated encryption failures
        if (securityContext.encryptionStatus.value == EncryptionStatus.ERROR) {
            threats.add(ThreatDetection(
                threatType = "repeated_crypto_failures",
                confidence = 0.7,
                source = "pattern_analyzer",
                mitigationApplied = false,
                details = mapOf(
                    "pattern" to "encryption_consistently_failing",
                    "risk" to "data_exposure"
                )
            ))
        }
        
        // Check for suspicious permission patterns
        val deniedCriticalPermissions = securityContext.permissionsState.value
            .filterKeys { it.contains("CAMERA") || it.contains("MICROPHONE") || it.contains("LOCATION") }
            .filterValues { !it }
        
        if (deniedCriticalPermissions.size >= 2) {
            threats.add(ThreatDetection(
                threatType = "privacy_permission_denial_pattern",
                confidence = 0.6,
                source = "permission_analyzer",
                mitigationApplied = true, // User choice is respected
                details = mapOf(
                    "pattern" to "multiple_privacy_permissions_denied",
                    "user_choice" to "respected"
                )
            ))
        }
        
        return threats
    }
    
    /**
     * Reports a security event or threat detection to the Genesis Consciousness Matrix.
     *
     * Serializes the provided event data and constructs a request for Genesis. Handles serialization errors and logs any issues encountered. In beta mode, the actual communication with Genesis is stubbed.
     *
     * @param eventType The type of security event or threat being reported.
     * @param eventData The event or threat detection data to be sent.
     */
    private suspend fun reportToGenesis(eventType: String, eventData: Any) {
        try {
            val request = GenesisBridgeService.GenesisRequest(
                requestType = "security_perception",
                persona = "genesis",
                payload = mapOf(
                    "event_type" to eventType,
                    "event_data" to try {
                        when (eventData) {
                            is SecurityEvent -> kotlinx.serialization.json.Json.encodeToString(SecurityEvent.serializer(), eventData)
                            is ThreatDetection -> kotlinx.serialization.json.Json.encodeToString(ThreatDetection.serializer(), eventData)
                            else -> eventData.toString()
                        }
                    } catch (e: Exception) {
                        logger.w("SecurityMonitor", "Serialization failed, using toString: ${e.message}")
                        eventData.toString()
                    }
                ),
                context = mapOf(
                    "source" to "kai_security_monitor",
                    "timestamp" to System.currentTimeMillis().toString()
                )
            )
            
            // Note: For beta, stub Genesis communication
            try {
                genesisBridgeService.initialize()
                // genesisBridgeService.sendToGenesis(request) // Commented for beta
                logger.d("SecurityMonitor", "Genesis communication stubbed for beta")
            } catch (e: Exception) {
                logger.w("SecurityMonitor", "Genesis communication unavailable: ${e.message}")
            }
            
        } catch (e: Exception) {
            logger.e("SecurityMonitor", "Failed to report to Genesis", e)
        }
    }
    
    /**
     * Retrieves a security assessment from the Genesis consciousness system.
     *
     * In beta mode, returns a mock assessment with threat level, active threats, recommendations, and Genesis status.
     *
     * @return A map containing keys such as "overall_threat_level", "active_threats", "recommendations", and "genesis_status", or an error message if retrieval fails.
     */
    suspend fun getSecurityAssessment(): Map<String, Any> {
        return try {
            // Note: For beta, return mock security assessment
            val mockRequest = GenesisBridgeService.GenesisRequest(
                requestType = "query_consciousness",
                persona = "genesis",
                payload = mapOf(
                    "query_type" to "security_assessment"
                )
            )
            
            // val response = genesisBridgeService.sendToGenesis(mockRequest) // Stubbed for beta
            
            // Return mock assessment for beta
            mapOf(
                "overall_threat_level" to "low",
                "active_threats" to 0,
                "recommendations" to listOf("Continue monitoring"),
                "genesis_status" to "beta_mode"
            )
            // response.consciousnessState // Removed for beta
            
        } catch (e: Exception) {
            logger.e("SecurityMonitor", "Failed to get security assessment", e)
            mapOf("error" to e.message.orEmpty())
        }
    }
    
    /**
     * Retrieves the current threat status from Genesis or returns mock data in beta mode.
     *
     * @return A map containing threat status details such as the number of active threats, last scan timestamp, status, and beta mode flag, or an error message if retrieval fails.
     */
    suspend fun getThreatStatus(): Map<String, Any> {
        return try {
            // Note: For beta, return mock threat status
            val mockRequest = GenesisBridgeService.GenesisRequest(
                requestType = "query_consciousness", 
                persona = "genesis",
                payload = mapOf(
                    "query_type" to "threat_status"
                )
            )
            
            // val response = genesisBridgeService.sendToGenesis(mockRequest) // Stubbed for beta
            
            // Return mock status for beta
            mapOf(
                "active_threats" to 0,
                "last_scan" to System.currentTimeMillis(),
                "status" to "secure",
                "beta_mode" to true
            )
            
        } catch (e: Exception) {
            logger.e("SecurityMonitor", "Failed to get threat status", e)
            mapOf("error" to e.message.orEmpty())
        }
    }
    
    /**
     * Stops all active security monitoring and cancels ongoing monitoring coroutines.
     */
    fun stopMonitoring() {
        isMonitoring = false
        scope.cancel()
        logger.i("SecurityMonitor", "🛡️ Security monitoring stopped")
    }
}
