package dev.aurakai.auraframefx.ui.components

import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color // For direct color usage if needed
import dev.aurakai.auraframefx.ui.theme.CyberpunkTextColor
import dev.aurakai.auraframefx.ui.theme.CyberpunkTextStyle

/**
 * Displays text with a custom cyberpunk color and style.
 *
 * Renders the provided text using the specified `CyberpunkTextColor` and `CyberpunkTextStyle`. The `enableGlitch` parameter is reserved for a future glitch effect and is currently unused.
 *
 * @param text The text to display.
 * @param color The cyberpunk-themed color to apply.
 * @param style The cyberpunk-themed text style to apply.
 * @param modifier Optional modifier for layout or interaction.
 * @param enableGlitch If true, will enable a glitch effect in the future (currently has no effect).
 */
@Composable
fun CyberpunkText(
    text: String,
    color: CyberpunkTextColor,
    style: CyberpunkTextStyle,
    modifier: Modifier = Modifier,
    enableGlitch: Boolean = false // Parameter based on usage, actual glitch effect not implemented in stub
) {
    // TODO: Implement actual glitch effect if enableGlitch is true
    // For now, it just applies color and style

    Text(
        text = text,
        color = color.color, // Access the actual Color from the sealed class
        style = style.textStyle, // Access the actual TextStyle from the sealed class
        modifier = modifier
    )
}
