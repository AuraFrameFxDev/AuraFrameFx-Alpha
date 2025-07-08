package dev.aurakai.auraframefx.model

import kotlinx.serialization.Serializable

@Serializable
data class AiRequest(
    val query: String,
    val contextId: String? = null, // Context identifier or type
    val data: Map<String, String>? = null, // Changed to Map<String, String> for consistency with other models
    val agentType: AgentType? = null // Adding this as it seemed to be expected in AIPipelineProcessor
)
