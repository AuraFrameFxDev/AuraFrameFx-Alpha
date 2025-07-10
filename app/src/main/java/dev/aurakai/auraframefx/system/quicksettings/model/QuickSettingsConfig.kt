package dev.aurakai.auraframefx.system.quicksettings.model

import android.graphics.Color
import dev.aurakai.auraframefx.system.overlay.model.OverlayShape
import dev.aurakai.auraframefx.ui.model.ImageResource
import kotlinx.serialization.Serializable

@Serializable
data class QuickSettingsConfig(
    val tiles: List<QuickSettingsTileConfig> = emptyList(),
    val background: QuickSettingsBackground? = null
)

@Serializable
data class QuickSettingsBackground(
    val color: String? = null,
    val alpha: Float = 1.0f,
    val padding: Int = 0,
    val heightDp: Int? = null
)

@Serializable
data class QuickSettingsTileConfig(
    val id: String,
    val label: String,
    val shape: OverlayShape,
    val animation: QuickSettingsAnimation,
    val enabled: Boolean = true,
    val enableClicks: Boolean = true
)

@Serializable
enum class QuickSettingsAnimation {
    FADE,
    SLIDE,
    PULSE
}
