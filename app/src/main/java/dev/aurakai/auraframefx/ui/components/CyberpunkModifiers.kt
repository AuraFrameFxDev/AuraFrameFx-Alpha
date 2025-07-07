package dev.aurakai.auraframefx.ui.components

import androidx.compose.foundation.border
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import dev.aurakai.auraframefx.ui.theme.NeonBlue
import dev.aurakai.auraframefx.ui.theme.NeonPurple
import dev.aurakai.auraframefx.ui.theme.NeonTeal

/**
 * Cyberpunk-themed modifier extensions for creating digital effects
 */

/**
     * Applies a neon blue edge glow effect to the modifier, combining a soft shadow and a semi-transparent border with rounded corners.
     *
     * Use this to give UI components a cyberpunk-inspired illuminated appearance.
     */
    fun Modifier.cyberEdgeGlow() = this
    .shadow(
        elevation = 8.dp,
        shape = RoundedCornerShape(4.dp),
        ambientColor = NeonBlue,
        spotColor = NeonBlue
    )
    .border(
        width = 1.dp,
        color = NeonBlue.copy(alpha = 0.6f),
        shape = RoundedCornerShape(4.dp)
    )

/**
     * Applies a digital glitch visual effect to the modifier, featuring a neon purple shadow and border with rounded corners.
     *
     * This effect creates a cyberpunk-inspired appearance suitable for highlighting UI components.
     */
    fun Modifier.digitalGlitchEffect() = this
    .shadow(
        elevation = 4.dp,
        shape = RoundedCornerShape(2.dp),
        ambientColor = NeonPurple,
        spotColor = NeonPurple
    )
    .border(
        width = 2.dp,
        color = NeonPurple.copy(alpha = 0.8f),
        shape = RoundedCornerShape(2.dp)
    )

/**
     * Applies a neon teal pixelated effect to the modifier, combining a soft shadow and a semi-transparent border with minimal rounded corners.
     *
     * Use this to give UI components a digital, pixel-inspired appearance with a cyberpunk aesthetic.
     */
    fun Modifier.digitalPixelEffect() = this
    .shadow(
        elevation = 6.dp,
        shape = RoundedCornerShape(1.dp),
        ambientColor = NeonTeal,
        spotColor = NeonTeal
    )
    .border(
        width = 1.dp,
        color = NeonTeal.copy(alpha = 0.7f),
        shape = RoundedCornerShape(1.dp)
    )

enum class CornerStyle {
    ROUNDED,
    SHARP,
    HEXAGON
}

enum class BackgroundStyle {
    SOLID,
    GRADIENT,
    GLITCH,
    MATRIX
}
