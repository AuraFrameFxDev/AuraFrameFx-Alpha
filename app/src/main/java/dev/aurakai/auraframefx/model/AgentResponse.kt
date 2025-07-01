package dev.aurakai.auraframefx.model

import kotlinx.serialization.Serializable

@Serializable
data class AgentResponse(
    val content: String,
    val isSuccess: Boolean,
    val error: String? = null
)
