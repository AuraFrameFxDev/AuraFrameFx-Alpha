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
     * Applies a neon blue edge glow effect with a medium shadow and rounded corners to the UI component.
     *
     * Adds an 8.dp shadow and a 1.dp border with 4.dp rounded corners using a neon blue color for a cyberpunk aesthetic.
     */
    /**
     * Applies a neon blue edge glow effect to the UI component.
     *
     * Adds a shadow and border with rounded corners to create a cyberpunk-inspired glowing appearance.
     * The effect uses a neon blue color for both the shadow and border, with the border rendered at 60% opacity.
     *
     * @return The modifier with the neon blue edge glow effect applied.
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
     * Applies a digital glitch effect with a neon purple shadow and border to the UI component.
     *
     * This modifier creates a cyberpunk-inspired glitch aesthetic by adding a 4.dp shadow and a 2.dp rounded border in vibrant neon purple.
     * @return The modified [Modifier] with the digital glitch effect applied.
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
     * Applies a pixelated cyberpunk effect with a neon teal shadow and border to the UI component.
     *
     * Adds a 6.dp neon teal shadow and a 1.dp border with minimal corner rounding for a digital, pixel-inspired appearance.
     * 
     * @return The modified [Modifier] with the pixel effect applied.
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
