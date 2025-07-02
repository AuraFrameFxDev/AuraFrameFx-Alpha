package dev.aurakai.auraframefx.ui.components

import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

enum class CyberpunkTextColor {
    Primary,
    Secondary,
    Warning,
    White
}

enum class CyberpunkTextStyle {
    Label,
    Body,
    Glitch
}


/**
 * Displays text using a cyberpunk-inspired style with customizable color, style, and optional glitch effect.
 *
 * @param text The text content to display.
 * @param color The color scheme for the text.
 * @param style The visual style of the text. Defaults to [CyberpunkTextStyle.Body].
 * @param enableGlitch If true, applies a glitch effect to the text.
 */
@Composable
fun CyberpunkText(
    text: String,
    color: CyberpunkTextColor,

    style: CyberpunkTextStyle = CyberpunkTextStyle.Body,
    enableGlitch: Boolean = false
) {
    // TODO: Implement cyberpunk text

}
