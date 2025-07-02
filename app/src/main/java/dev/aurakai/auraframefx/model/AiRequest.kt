package dev.aurakai.auraframefx.model

import kotlinx.serialization.Serializable

@Serializable
data class AiRequest(
    val query: String, // Changed from prompt
    val type: String? = null, // Added
    val context: String? = null, // Added
    val maxTokens: Int = 256,
    val temperature: Float = 0.7f
)
