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
     * Applies a cyberpunk-themed neon blue edge glow effect to the modifier.
     *
     * Adds a shadow with 8.dp elevation and a rounded 4.dp corner, along with a semi-transparent neon blue border.
     * Use to give UI elements a distinctive cyber edge highlight.
     */
<<<<<<< HEAD
    /**
     * Applies a neon blue edge glow effect to the component.
     *
     * Adds a shadow and a semi-transparent neon blue border with rounded corners for a cyberpunk-inspired appearance.
     */
=======
>>>>>>> pr458merge
    /**
     * Applies a cyberpunk-themed neon blue edge glow effect to the modifier.
     *
     * Adds a shadow with 8.dp elevation and rounded corners, along with a semi-transparent neon blue border for a futuristic visual style.
     */
    /**
     * Applies a neon blue edge glow effect to the component, creating a cyberpunk-inspired visual style.
     *
     * Adds an 8.dp shadow and a semi-transparent neon blue border with rounded corners for a futuristic appearance.
     */
    /**
     * Applies a neon blue edge glow effect to the modifier, creating a cyberpunk-inspired visual style.
     *
     * Adds an 8.dp elevation shadow and a semi-transparent neon blue border with rounded corners for a luminous, futuristic appearance.
     * 
     * @return The modifier with the neon blue edge glow effect applied.
     */
    /**
     * Applies a neon blue edge glow effect to the modifier, giving UI components a cyberpunk-inspired appearance.
     *
     * The effect includes a shadow with 8.dp elevation and rounded corners (4.dp radius), using neon blue for both ambient and spot colors. A 1.dp border with semi-transparent neon blue (alpha 0.6) and matching rounded corners is also added.
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
     * Applies a neon purple glitch effect to the modifier, evoking a cyberpunk digital aesthetic.
     *
     * Adds a 4.dp elevation shadow and a 2.dp semi-transparent neon purple border with slightly rounded corners for a distinctive glitch visual style.
     *
     * @return The modified [Modifier] with the glitch effect applied.
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
     * Applies a pixelated neon teal effect with a cyberpunk aesthetic to the modifier.
     *
     * Adds a 6.dp neon teal shadow and a 1.dp semi-transparent border with slightly rounded corners, creating a digital pixelated visual style for Compose UI components.
     *
     * @return The modified [Modifier] with the pixelated neon teal effect applied.
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
