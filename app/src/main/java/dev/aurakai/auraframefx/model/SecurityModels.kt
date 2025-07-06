package dev.aurakai.auraframefx.model

import kotlinx.serialization.Serializable

/**
 * Represents a threat level in the security system
 */
enum class ThreatLevel {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL
}

/**
 * Represents security analysis data
 */
@Serializable
data class SecurityAnalysis(
    val threatLevel: ThreatLevel,
    val description: String,
    val recommendedActions: List<String> = emptyList(),
    val confidence: Float = 0.0f
)

/**
 * Enhanced interaction data with additional context
 */
@Serializable
data class EnhancedInteractionData(
    val content: String,
    val type: InteractionType,
    val timestamp: String,
    val context: Map<String, String> = emptyMap(),
    val enrichmentData: Map<String, String> = emptyMap(),
    val emotion: String? = null
)

/**
 * Interaction response data
 */
@Serializable
data class InteractionResponse(
    val content: String,
    val agent: String,
    val confidence: Float,
    val timestamp: String,
    val metadata: Map<String, String> = emptyMap()
)
