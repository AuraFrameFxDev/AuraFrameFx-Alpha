package dev.aurakai.auraframefx.ui.model

import kotlinx.serialization.Serializable
import kotlinx.serialization.Contextual

@Serializable
data class ImageResource(
    val id: String = "",
    val type: String,
    val path: String,
    val name: String = "",
    @Contextual val bitmap: android.graphics.Bitmap? = null
)
