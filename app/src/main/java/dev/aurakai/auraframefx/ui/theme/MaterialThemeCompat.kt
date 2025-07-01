package dev.aurakai.auraframefx.ui.theme

import android.app.Activity
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

// import android.app.Activity // Already imported by being moved up
/**
 * MaterialThemeCompat serves as a compatibility layer between the XML theme and Compose Material3.
 * It ensures consistent theme application across both systems while respecting Android's theme
 * characteristics.
 */

/**
 * Applies the AuraFrameFX Material3 theme with proper system integration
 */
@Composable
fun AuraFrameFXThemeCompat(
    content: @Composable () -> Unit,
) {
    // Wrap the existing theme implementation
    AuraFrameFXTheme {
        // Get the current view to apply system UI colors
        val view = LocalView.current
        if (!view.isInEditMode) {
            SideEffect {
                val window = (view.context as Activity).window
                // Use MaterialTheme color values for system bars
                window.statusBarColor = MaterialTheme.colorScheme.background.toArgb()
                // Ensure proper contrast for status bar icons
                WindowCompat.getInsetsController(window, view)
                    .isAppearanceLightStatusBars =
                    MaterialTheme.colorScheme.background.luminance() > 0.5
            }
        }

        content()
    }
}
