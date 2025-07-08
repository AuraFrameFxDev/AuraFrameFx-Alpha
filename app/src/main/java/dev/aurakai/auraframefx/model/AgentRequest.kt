package dev.aurakai.auraframefx.model

import kotlinx.serialization.Serializable

@Serializable
data class AgentRequest(
    val query: String,
    val type: String, // Based on usage in TaskExecutionManager
    val context: Map<String, String>? = null,
    val metadata: Map<String, String>? = null,
    val agentType: AgentType? = null // Adding for consistency with AiRequest, optional
)
