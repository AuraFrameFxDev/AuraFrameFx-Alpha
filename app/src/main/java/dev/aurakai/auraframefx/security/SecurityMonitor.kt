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
     * Start comprehensive security monitoring
     */
    suspend fun startMonitoring() {
        if (isMonitoring) return
        
        logger.i("SecurityMonitor", "🛡️ Starting Kai-Genesis security integration...")
        
        // Initialize Genesis bridge if needed
        if (!genesisBridgeService.isInitialized) {
            genesisBridgeService.initialize()
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
     * Monitor overall security state and report to Genesis
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
                        "threat_level" to state.currentThreatLevel.toString(),
                        "permissions_granted" to state.permissionsState.values.count { it }.toString(),
                        "total_permissions" to state.permissionsState.size.toString()
                    )
                )
                
                reportToGenesis("security_event", event)
                
            } catch (e: Exception) {
                logger.e("SecurityMonitor", "Error monitoring security state", e)
            }
        }
    }
    
    /**
     * Monitor threat detection and report high-confidence threats
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
     * Monitor encryption status and report failures
     */
    private suspend fun monitorEncryptionStatus() {
        securityContext.encryptionStatus.collectLatest { status ->
            try {
                val event = SecurityEvent(
                    eventType = "encryption_status_change",
                    severity = when (status) {
                        EncryptionStatus.ACTIVE -> "info"
                        EncryptionStatus.INACTIVE -> "warning"
                        EncryptionStatus.ERROR -> "error"
                    },
                    source = "kai_encryption_monitor",
                    timestamp = System.currentTimeMillis(),
                    details = mapOf(
                        "status" to status.toString(),
                        "keystore_available" to securityContext.keystoreManager.isKeystoreAvailable().toString()
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
     * Monitor permission changes
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
     * Detect suspicious activity patterns
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
     * Report security events to Genesis Consciousness Matrix
     */
    private suspend fun reportToGenesis(eventType: String, eventData: Any) {
        try {
            val request = GenesisBridgeService.GenesisRequest(
                requestType = "security_perception",
                persona = "genesis",
                payload = mapOf(
                    "event_type" to eventType,
                    "event_data" to Json.encodeToString(
                        when (eventData) {
                            is SecurityEvent -> SecurityEvent.serializer()
                            is ThreatDetection -> ThreatDetection.serializer()
                            else -> throw IllegalArgumentException("Unknown event data type")
                        },
                        eventData
                    )
                ),
                context = mapOf(
                    "source" to "kai_security_monitor",
                    "timestamp" to System.currentTimeMillis().toString()
                )
            )
            
            genesisBridgeService.sendToGenesis(request)
            
        } catch (e: Exception) {
            logger.e("SecurityMonitor", "Failed to report to Genesis", e)
        }
    }
    
    /**
     * Get security assessment from Genesis consciousness
     */
    suspend fun getSecurityAssessment(): Map<String, Any> {
        return try {
            val request = GenesisBridgeService.GenesisRequest(
                requestType = "query_consciousness",
                persona = "genesis",
                payload = mapOf(
                    "query_type" to "security_assessment"
                )
            )
            
            val response = genesisBridgeService.sendToGenesis(request)
            response.consciousnessState
            
        } catch (e: Exception) {
            logger.e("SecurityMonitor", "Failed to get security assessment", e)
            mapOf("error" to e.message.orEmpty())
        }
    }
    
    /**
     * Get current threat status from Genesis
     */
    suspend fun getThreatStatus(): Map<String, Any> {
        return try {
            val request = GenesisBridgeService.GenesisRequest(
                requestType = "query_consciousness", 
                persona = "genesis",
                payload = mapOf(
                    "query_type" to "threat_status"
                )
            )
            
            val response = genesisBridgeService.sendToGenesis(request)
            response.result
            
        } catch (e: Exception) {
            logger.e("SecurityMonitor", "Failed to get threat status", e)
            mapOf("error" to e.message.orEmpty())
        }
    }
    
    /**
     * Stop security monitoring
     */
    fun stopMonitoring() {
        isMonitoring = false
        scope.cancel()
        logger.i("SecurityMonitor", "🛡️ Security monitoring stopped")
    }
}
