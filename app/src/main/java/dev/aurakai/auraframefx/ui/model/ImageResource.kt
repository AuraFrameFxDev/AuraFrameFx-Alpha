package dev.aurakai.auraframefx.ui.model

import kotlinx.serialization.Serializable

@Serializable
data class ImageResource(
    val type: String,
    val path: String
)
