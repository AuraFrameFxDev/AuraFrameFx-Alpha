package dev.aurakai.auraframefx.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val DarkColorScheme = darkColorScheme(
    primary = NeonTeal,
    secondary = NeonPurple,
    tertiary = NeonBlue,
    background = DarkBackground,
    surface = Surface,
    onPrimary = OnPrimary,
    onSecondary = OnSecondary,
    onTertiary = OnPrimary,
    onBackground = OnSurface,
    onSurface = OnSurface,
    error = ErrorColor,
    onError = OnPrimary
)

private val LightColorScheme = lightColorScheme(
    primary = LightPrimary,
    secondary = LightSecondary,
    tertiary = LightTertiary,
    background = LightBackground,
    surface = LightSurface,
    onPrimary = LightOnPrimary,
    onSecondary = LightOnSecondary,
    onTertiary = LightOnTertiary,
    onBackground = LightOnBackground,
    onSurface = LightOnSurface,
    error = LightError,
    onError = LightOnError
)

/**
 * Applies the AuraFrameFX Material3 theme to the given composable content.
 *
 * Chooses between light, dark, or dynamic color schemes based on system settings and parameters. On supported devices, dynamic color adapts the theme to the user's wallpaper. Also updates the status bar color and icon appearance to match the selected theme.
 *
 * @param darkTheme If true, uses the dark color scheme; otherwise, uses the light color scheme. Defaults to the system dark theme setting.
 * @param dynamicColor If true and supported (Android 12+), enables dynamic color theming based on the user's wallpaper. Defaults to true.
 * @param content The composable content to display within the themed context.
 */
@Composable
fun AuraFrameFXTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    // Dynamic color is available on Android 12+
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.primary.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = AppTypography,
        content = content
    )
}
