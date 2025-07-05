package dev.aurakai.auraframefx.model

import kotlinx.serialization.Serializable

@Serializable
data class EnhancedInteractionData(
    val original: InteractionData,
    val emotion: String? = null,
    val context: String? = null
)

@Serializable
data class InteractionData(
    val content: String,
    val type: String = "text",
    val timestamp: Long = System.currentTimeMillis()
)

@Serializable
data class InteractionResponse(
    val content: String,
    val responseType: String = "text",
    val confidence: Float = 1.0f,
    val metadata: Map<String, String> = emptyMap()
)

enum class CreativeIntent {
    ARTISTIC,
    FUNCTIONAL,
    EXPERIMENTAL,
    EMOTIONAL
}

@Serializable
data class SecurityAnalysis(
    val threatLevel: ThreatLevel,
    val description: String,
    val recommendedActions: List<String>,
    val confidence: Float
)

/**
 * Represents interaction types
 */
enum class InteractionType {
    TEXT,
    VOICE,
    IMAGE,
    VIDEO,
    GESTURE,
    SYSTEM
}

/**
 * Represents agent request for processing
 */
@Serializable
data class AgentRequest(
    val query: String,
    val type: String = "text",
    val context: Map<String, String> = emptyMap(),
    val metadata: Map<String, String> = emptyMap()
)

/**
 * Represents agent response
 */
@Serializable
data class AgentResponse(
    val content: String,
    val confidence: Float = 1.0f,
    val metadata: Map<String, String> = emptyMap()
)

/**
 * Represents AI request for processing
 */
@Serializable
data class AiRequest(
    val query: String,
    val type: String = "text",
    val context: Map<String, String> = emptyMap()
)

/**
 * Represents text generation request
 */
@Serializable
data class GenerateTextRequest(
    val prompt: String,
    val maxTokens: Int = 1000,
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f
)

/**
 * Represents text generation response
 */
@Serializable
data class GenerateTextResponse(
    val generatedText: String,
    val finishReason: String = "completed",
    val usage: Map<String, Int> = emptyMap()
)
