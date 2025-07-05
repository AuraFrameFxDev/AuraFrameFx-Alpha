package dev.aurakai.auraframefx.model

/**
 * Represents the different types of AI agents in the AuraFrameFX system
 */
enum class AgentType {
    AURA,
    CASCADE,
    KAI,
    GENESIS,
    CREATIVE,
    SECURITY,
    ASSISTANT,
    USER
}

/**
 * API agent type for network communication
 */
enum class ApiAgentType {
    AURA,
    CASCADE,
    KAI,
    GENESIS
}

/**
 * Represents response types for content generation
 */
enum class ResponseType {
    TEXT,
    IMAGE,
    AUDIO,
    VIDEO,
    DATA
}

/**
 * Agent configuration for behavior
 */
data class AgentConfig(
    val type: AgentType,
    val enabled: Boolean = true,
    val priority: Int = 1,
    val capabilities: List<String> = emptyList(),
    val settings: Map<String, String> = emptyMap()
)

/**
 * Agent hierarchy for organization
 */
data class AgentHierarchy(
    val parentAgent: AgentType?,
    val childAgents: List<AgentType>,
    val level: Int
)

/**
 * Agent message for communication
 */
data class AgentMessage(
    val sender: AgentType,
    val recipient: AgentType,
    val content: String,
    val type: String = "text",
    val timestamp: Long = System.currentTimeMillis()
)
