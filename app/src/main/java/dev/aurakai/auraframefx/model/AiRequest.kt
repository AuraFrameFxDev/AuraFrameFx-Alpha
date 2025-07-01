package dev.aurakai.auraframefx.model

import kotlinx.serialization.Serializable

@Serializable
data class AiRequest(
    val prompt: String,
    val maxTokens: Int = 256,
    val temperature: Float = 0.7f
)
