package dev.aurakai.auraframefx.model

import kotlinx.serialization.Serializable

@Serializable
data class AiRequest(
    val query: String,
    val context: Map<String, String>? = null, // Changed from contextId to context: Map
    val data: Map<String, String>? = null,
    val agentType: AgentType? = null
)
