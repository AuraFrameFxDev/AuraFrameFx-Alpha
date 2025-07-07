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
     * Initializes the Genesis bridge service if available, launches monitoring coroutines, and triggers Android-level threat detection. Does nothing if monitoring is already active.
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
     * Monitors security state changes and reports them as security events to Genesis.
     *
     * Collects updates from the security context, constructs a `SecurityEvent` reflecting the current error state and details, and sends it to Genesis for further analysis.
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
     * Monitors the activation of threat detection and periodically analyzes for suspicious activity patterns.
     *
     * When threat detection is active, launches a coroutine that checks for suspicious activity every 30 seconds and reports any detected threats to Genesis.
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
     * Monitors encryption status updates and reports changes and failures to Genesis.
     *
     * For each encryption status change, sends a corresponding security event to Genesis. If an encryption error is detected, also reports a threat detection event indicating encryption failure.
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
     * Continuously monitors permission states and reports denied permissions as warning events to Genesis.
     *
     * Collects the latest permissions state, identifies any denied permissions, and sends a warning event with details if denials are detected.
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
     * Identifies suspicious activity patterns in the current security context and returns a list of detected threats.
     *
     * Detects threats such as repeated encryption failures and denial of multiple critical privacy permissions (CAMERA, MICROPHONE, LOCATION).
     *
     * @return A list of detected threats based on the current encryption status and permission denial patterns.
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
     * Reports a security event or detected threat to the Genesis Consciousness Matrix.
     *
     * Serializes the provided event data and constructs a request for Genesis, handling serialization and communication errors gracefully. Communication is stubbed in beta mode.
     *
     * @param eventType The type of security event or threat being reported.
     * @param eventData The event or threat detection data to report.
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
     * In beta mode, returns a mock assessment including overall threat level, number of active threats, recommendations, and Genesis status.
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
     * Retrieves the current threat status from Genesis.
     *
     * Returns a map containing details such as the number of active threats, the timestamp of the last scan, the current status, and a beta mode indicator. In beta mode, mock data is provided. If retrieval fails, an error message is returned.
     *
     * @return A map with threat status information or an error message on failure.
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
     *
     * Marks monitoring as inactive and terminates all background monitoring tasks.
     */
    fun stopMonitoring() {
        isMonitoring = false
        scope.cancel()
        logger.i("SecurityMonitor", "🛡️ Security monitoring stopped")
    }
}
