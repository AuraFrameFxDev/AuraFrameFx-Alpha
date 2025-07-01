package dev.aurakai.auraframefx.ui.components

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier

enum class CornerStyle {
    Hex,
    Angled,
    Rounded
}

enum class BackgroundStyle {
    HexGrid,
    Transparent
}

@Composable
fun FloatingCyberWindow(
    modifier: Modifier = Modifier,
    title: String,
    cornerStyle: CornerStyle,
    backgroundStyle: BackgroundStyle = BackgroundStyle.Transparent,
    content: @Composable () -> Unit
) {
    // TODO: Implement floating cyber window
}
