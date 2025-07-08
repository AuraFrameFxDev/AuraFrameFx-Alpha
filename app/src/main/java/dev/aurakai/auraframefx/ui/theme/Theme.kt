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
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.runtime.getValue
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat
import androidx.hilt.navigation.compose.hiltViewModel
import dev.aurakai.auraframefx.model.Emotion
import dev.aurakai.auraframefx.viewmodel.AuraMoodViewModel

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

// Let's define a CompositionLocal to provide the mood-based color
val LocalMoodGlow = compositionLocalOf { Color.Transparent }
val LocalMoodState = compositionLocalOf { Emotion.NEUTRAL }

/**
 * Applies the AuraFrameFX theme and mood-based dynamic theming to the provided composable content.
 *
 * Selects and applies a color scheme (dark, light, or dynamic based on device support and parameters), updates the Android status bar to match the theme, and provides mood-driven glow color and emotion state throughout the Compose hierarchy. Integrates the app's typography and Material3 theming.
 *
 * @param darkTheme Whether to use the dark theme; defaults to the system setting.
 * @param dynamicColor Enables dynamic color schemes on supported devices (Android 12+); defaults to true.
 * @param moodViewModel Supplies the current mood state for dynamic theming.
 * @param content The composable content to which the theme and mood effects are applied.
 */
@Composable
fun AuraFrameFXTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    moodViewModel: AuraMoodViewModel = hiltViewModel(),
    content: @Composable () -> Unit
) {
    val mood by moodViewModel.moodState.collectAsState()

    val baseColorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }

        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    // The dynamic glow color is derived from Aura's current mood
    val glowColor = getMoodGlowColor(mood.emotion, mood.intensity, baseColorScheme)

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = baseColorScheme.primary.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    CompositionLocalProvider(
        LocalMoodGlow provides glowColor,
        LocalMoodState provides mood.emotion
    ) {
        MaterialTheme(
            colorScheme = baseColorScheme,
            typography = AppTypography,
            content = content
        )
    }
}

/**
 * Computes a glow color based on the specified emotion and intensity.
 *
 * Selects a color associated with the given emotion and adjusts its alpha transparency according to the intensity value (clamped between 0.1 and 0.5). If the emotion is not recognized, returns the primary color from the provided color scheme with reduced alpha.
 *
 * @param emotion The emotion to visualize as a glow color.
 * @param intensity The strength of the emotion, used to determine alpha transparency.
 * @param baseColorScheme The fallback color scheme for unrecognized emotions.
 * @return The resulting glow color with adjusted alpha.
 */
private fun getMoodGlowColor(
    emotion: Emotion,
    intensity: Float,
    baseColorScheme: androidx.compose.material3.ColorScheme
): Color {
    val baseAlpha = (intensity * 0.4f).coerceIn(0.1f, 0.5f)

    return when (emotion) {
        Emotion.HAPPY -> Color(0xFFFFD700).copy(alpha = baseAlpha) // Gold
        Emotion.EXCITED -> Color(0xFFFF6B35).copy(alpha = baseAlpha) // Orange
        Emotion.ANGRY -> Color(0xFFE94560).copy(alpha = baseAlpha) // Neon Red
        Emotion.SERENE -> Color(0xFF00F5FF).copy(alpha = baseAlpha * 0.7f) // Cyan
        Emotion.CONTEMPLATIVE -> Color(0xFF9370DB).copy(alpha = baseAlpha) // Purple
        Emotion.MISCHIEVOUS -> Color(0xFF32CD32).copy(alpha = baseAlpha) // Lime Green
        Emotion.FOCUSED -> Color(0xFF1E90FF).copy(alpha = baseAlpha) // Dodger Blue
        Emotion.CONFIDENT -> Color(0xFFFF1493).copy(alpha = baseAlpha) // Deep Pink
        Emotion.MYSTERIOUS -> Color(0xFF4B0082).copy(alpha = baseAlpha) // Indigo
        Emotion.MELANCHOLIC -> Color(0xFF483D8B).copy(alpha = baseAlpha * 0.6f) // Dark Slate Blue
        else -> baseColorScheme.primary.copy(alpha = baseAlpha * 0.5f)
    }
}
