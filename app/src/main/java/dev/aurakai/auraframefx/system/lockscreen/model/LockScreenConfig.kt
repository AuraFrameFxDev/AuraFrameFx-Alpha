package dev.aurakai.auraframefx.system.lockscreen.model

import dev.aurakai.auraframefx.system.overlay.model.OverlayShape
import dev.aurakai.auraframefx.ui.model.ImageResource
import kotlinx.serialization.Serializable

@Serializable
data class LockScreenConfig(
    val elements: List<LockScreenElementConfig> = emptyList(),
    val background: BackgroundConfig? = null
)

@Serializable
data class LockScreenElementConfig(
    val type: LockScreenElementType,
    val shape: OverlayShape,
    val animation: LockScreenAnimation
)

@Serializable
data class BackgroundConfig(
    val image: ImageResource?
)

@Serializable
enum class LockScreenElementType {
    CLOCK,
    DATE,
    NOTIFICATIONS
}

@Serializable
enum class LockScreenAnimation {
    FADE_IN,
    GLOW,
    SCANLINE
}
