package dev.aurakai.auraframefx.ui.components

import androidx.compose.foundation.border
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.unit.dp
import dev.aurakai.auraframefx.ui.theme.NeonBlue
import dev.aurakai.auraframefx.ui.theme.NeonPurple
import dev.aurakai.auraframefx.ui.theme.NeonTeal

/**
 * Cyberpunk-themed modifier extensions for creating digital effects
 */

/**
 * Applies a cyberpunk-themed neon blue edge glow effect to the modifier.
 *
 * Adds a shadow with 8.dp elevation and a rounded 4.dp corner, along with a semi-transparent neon blue border.
 * Use to give UI elements a distinctive cyber edge highlight.
 */
/**
 * Applies a neon blue edge glow effect with a shadow and border to the modifier.
 *
 * Adds an 8.dp elevation shadow and a 1.dp border with rounded corners, using a neon blue color scheme for a cyberpunk appearance.
 */
/**
 * Applies a neon blue edge glow effect with a soft shadow and semi-transparent border to the UI element.
 *
 * @return The modified [Modifier] with the cyber edge glow styling applied.
 */
/**
 * Applies a neon blue edge glow effect to the UI element.
 *
 * Adds an 8.dp elevation shadow and a 1.dp border with 60% opacity neon blue color, both using a rounded corner shape of 4.dp.
 *
 * @return The modified [Modifier] with the cyber edge glow effect applied.
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
 * Applies a neon purple digital glitch effect to the modifier.
 *
 * This effect adds a 4.dp elevation shadow and a 2.dp border with 2.dp rounded corners, both styled with neon purple. The border uses 80% opacity to enhance the cyberpunk glitch aesthetic.
 *
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
 * Applies a pixelated cyberpunk effect with neon teal shadow and border to the modifier.
 *
 * Adds a 6.dp elevation shadow and a 1.dp border with 70% opacity neon teal color, both using slightly rounded corners for a digital pixel aesthetic.
 *
 * @return The modifier with the pixelated neon teal effect applied.
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
