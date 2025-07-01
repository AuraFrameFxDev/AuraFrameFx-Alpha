package dev.aurakai.auraframefx.ui.theme

import androidx.compose.ui.graphics.Color

// Primary Colors - Enhanced Neon Palette
val NeonTeal = Color(0xFF00FFCC) // Brighter teal for accents
val NeonPurple = Color(0xFFE000FF) // Slightly softer purple for readability
val NeonBlue = Color(0xFF00FFFF) // Bright cyan for highlights
val NeonPink = Color(0xFFFF00FF) // Bright pink for secondary accents
val NeonGreen = Color(0xFF00FF00) // Bright neon green
val NeonRed = Color(0xFFFF0000)   // Bright neon red
val NeonYellow = Color(0xFFFFFF00) // Bright yellow

// Alias for backwards compatibility
val NeonCyan = NeonBlue 

// Background Colors - Deep Cyberpunk Noir
val DarkBackground = Color(0xFF000000) // Pure black for depth
val Surface = Color(0xFF1A1A1A) // Very dark grey for surfaces
val SurfaceVariant = Color(0xFF2D2D2D) // Slightly lighter for variants
val CardBackground = Color(0xFF333333) // Card background color

// Text Colors - Neon Glow
val OnSurface = Color(0xFF00FFCC) // Neon teal for main text
val OnSurfaceVariant = Color(0xFF00FFFF) // Neon cyan for variant text
val OnPrimary = Color(0xFF000000) // Black for text on neon backgrounds
val OnSecondary = Color(0xFF1A1A1A) // Dark grey for secondary text

// Accent Colors - Glowing Effects
val Accent1 = NeonTeal 
val Accent2 = NeonPurple
val Accent3 = NeonBlue
val Accent4 = NeonPink

// System Colors - Enhanced for Cyberpunk
val ErrorColor = NeonRed 
val WarningColor = Color(0xFFFFA500) // Orange for warnings
val SuccessColor = NeonGreen

// Light Theme Colors - For fallback
val LightPrimary = NeonTeal
val LightOnPrimary = Color(0xFF000000)
val LightSecondary = NeonPurple
val LightOnSecondary = Color(0xFF000000)
val LightTertiary = NeonBlue
val LightOnTertiary = Color(0xFF000000)

val LightBackground = Color(0xFF1A1A1A)
val LightOnBackground = NeonTeal
val LightSurface = Color(0xFF2D2D2D)
val LightOnSurface = NeonTeal

val LightError = NeonRed
val LightOnError = Color(0xFF000000)

// Special Effects Colors
val GlowOverlay = Color(0x1A00FFCC) // Semi-transparent teal glow
val PulseOverlay = Color(0x1AE000FF) // Semi-transparent purple pulse
val HoverOverlay = Color(0x1A00FFFF) // Semi-transparent cyan hover
val PressOverlay = Color(0x1AFF00FF) // Semi-transparent pink press
