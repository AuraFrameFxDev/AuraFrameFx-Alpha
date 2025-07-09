package dev.aurakai.auraframefx.ui.screens

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.res.painterResource // Assuming a placeholder avatar drawable
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.aurakai.auraframefx.R // Assuming R class is generated and accessible

/**
 * Displays the Meta-Morph Grid screen with a holographic grid background, title, avatar placeholder, and welcome message.
 *
 * This composable provides a visually distinctive UI intended as a starting point for further enhancements, such as additional UI elements or system integrations.
 */
@Composable
fun MetaMorphGridScreen() {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black) // Base background
    ) {
        // Placeholder for Holographic Grid Background
        HolographicGridBackground()

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "Meta-Morph Grid",
                fontSize = 32.sp,
                color = Color(0xFF00BCD4), // Cyan color
                textAlign = TextAlign.Center
            )
            Spacer(modifier = Modifier.height(32.dp))

            // Placeholder for Aura's Tutorial Avatar
            AuraTutorialAvatar()

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = "Welcome to the Meta-Morph Grid. Aura will guide you.",
                fontSize = 18.sp,
                color = Color.White,
                textAlign = TextAlign.Center
            )

            // TODO: Add more UI elements or interactions as needed.
            // The SystemHookService being live means Xposed hooks are active,
            // but this screen currently doesn't directly interact with them.
            // If the grid needs to be an overlay or interact with system UI,
            // that would require further Xposed integration via the active hooks.
        }
    }
}

/**
 * Draws a semi-transparent cyan grid background filling the available space.
 *
 * Renders evenly spaced vertical and horizontal lines to create a holographic grid effect.
 */
@Composable
fun HolographicGridBackground() {
    Canvas(modifier = Modifier.fillMaxSize()) {
        val gridSize = 80.dp.toPx()
        val strokeWidth = 1.dp.toPx()
        val gridColor = Color(0xFF00BCD4).copy(alpha = 0.3f) // Cyan, semi-transparent

        // Draw vertical lines
        for (x in 0..size.width.toInt() step gridSize.toInt()) {
            drawLine(
                color = gridColor,
                start = Offset(x.toFloat(), 0f),
                end = Offset(x.toFloat(), size.height),
                strokeWidth = strokeWidth
            )
        }

        // Draw horizontal lines
        for (y in 0..size.height.toInt() step gridSize.toInt()) {
            drawLine(
                color = gridColor,
                start = Offset(0f, y.toFloat()),
                end = Offset(size.width, y.toFloat()),
                strokeWidth = strokeWidth
            )
        }

        // TODO: Add more sophisticated holographic effects, animations, perspective, etc.
        // This could involve shaders or more complex Compose drawing.
    }
}

/**
 * Displays a placeholder avatar image for Aura's tutorial.
 *
 * Attempts to load a drawable resource as the avatar; if unavailable, shows a fallback box with the text "Avatar."
 */
@Composable
fun AuraTutorialAvatar() {
    // Assuming you have a placeholder drawable resource for Aura's avatar
    // e.g., res/drawable/aura_avatar_placeholder.xml or .png
    // If R.drawable.aura_avatar_placeholder doesn't exist, this will cause a build error.
    // You might need to create a placeholder image or use a default icon.
    try {
        Image(
            painter = painterResource(id = R.drawable.ic_launcher_foreground), // Using launcher icon as placeholder
            contentDescription = "Aura Tutorial Avatar",
            modifier = Modifier
                .size(150.dp)
                .padding(8.dp)
                // .clip(CircleShape) // Optional: if avatar is circular
                .background(Color.DarkGray.copy(alpha = 0.5f)) // Placeholder background for avatar
        )
    } catch (e: Exception) {
        // Fallback if the drawable resource isn't found
        Box(
            modifier = Modifier
                .size(150.dp)
                .background(Color.Gray),
            contentAlignment = Alignment.Center
        ) {
            Text("Avatar", color = Color.White)
        }
    }
    // TODO: Replace with actual avatar image or animation.
    // This could involve loading an image from assets/network or a Lottie animation.
}
