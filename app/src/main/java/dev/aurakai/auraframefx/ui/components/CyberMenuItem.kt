package dev.aurakai.auraframefx.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Displays a menu item with customizable text, selection state, and click action,
 * intended for a cyberpunk-themed UI.
 *
 * @param text The label to display on the menu item.
 * @param onClick The action to perform when the menu item is clicked.
 * @param modifier Modifier to adjust the layout or appearance of the menu item.
 * @param isSelected Whether the menu item is currently selected.
 */
/**
 * Displays a cyberpunk-themed menu item with customizable text, selection state, and click behavior.
 *
 * @param text The label displayed on the menu item.
 * @param onClick Callback invoked when the menu item is clicked.
 * @param modifier Modifier for layout or appearance customization.
 * @param isSelected Whether the menu item is currently selected, affecting its visual style.
 */
/**
 * Displays a cyberpunk-themed menu item with customizable text and selection state.
 *
 * The visual style changes based on whether the item is selected. When selected, the background and text color are highlighted to indicate active state.
 *
 * @param text The label displayed on the menu item.
 * @param onClick Callback invoked when the menu item is clicked.
 * @param modifier Modifier for customizing the layout or appearance.
 * @param isSelected Whether the menu item is currently selected, affecting its visual style.
 */
@Composable
fun CyberMenuItem(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    isSelected: Boolean
) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .background(
                // TODO: Replace with cyberpunk theme colors
                if (isSelected) Color.DarkGray.copy(alpha = 0.7f) else Color.Transparent
            )
            .padding(horizontal = 16.dp, vertical = 12.dp),
        contentAlignment = Alignment.CenterStart // Or as per design
    ) {
        Text(
            text = text,
            style = TextStyle(
                // TODO: Apply cyberpunk font, text color, glow effects, etc.
                color = if (isSelected) Color.Cyan else Color.LightGray,
                fontSize = 16.sp
            )
        )
        // TODO: Optionally add an icon here
        // TODO: Add animations or other cyberpunk visual cues
    }
}
