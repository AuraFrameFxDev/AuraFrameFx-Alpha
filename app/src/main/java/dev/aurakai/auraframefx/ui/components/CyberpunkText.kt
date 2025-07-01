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
 * Displays text with a customizable cyberpunk-inspired style.
 *
 * @param text The content to display.
 * @param color The color scheme to apply to the text.
 * @param style The visual style of the text. Defaults to [CyberpunkTextStyle.Body].
 * @param enableGlitch Whether to apply a glitch effect to the text.
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
