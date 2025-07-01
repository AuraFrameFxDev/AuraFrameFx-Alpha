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


@Composable
fun CyberpunkText(
    text: String,
    color: CyberpunkTextColor,

    style: CyberpunkTextStyle = CyberpunkTextStyle.Body,
    enableGlitch: Boolean = false
) {
    // TODO: Implement cyberpunk text

}
