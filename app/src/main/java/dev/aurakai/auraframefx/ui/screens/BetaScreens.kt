package dev.aurakai.auraframefx.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController

/**
 * Displays the home screen with app titles, status messages, and navigation buttons.
 *
 * Presents the main interface for the app, including navigation to the AI chat screen and a placeholder for settings.
 */
/**
 * Displays the main home screen with app title, status messages, and navigation buttons centered vertically.
 *
 * The screen includes the app name, a subtitle, a beta status message, and two buttons: one to navigate to the AI Chat screen and another for settings (navigation not yet implemented).
 */
/**
 * Displays the main home screen with the app title, subtitle, beta status message, and navigation buttons.
 *
 * Vertically centers the content, including the app title ("AuraFrameFX Alpha"), subtitle ("Trinity AI System"), and a beta status message. Provides two buttons: one navigates to the AI Chat screen, and the other is a placeholder for future settings navigation.
 */
/**
 * Displays the main home screen with app title, subtitle, beta status, and navigation buttons.
 *
 * The screen centers its content vertically and horizontally, showing the app's name, a subtitle,
 * and a beta status message. It provides a button to navigate to the AI Chat screen and a placeholder
 * button for Settings.
 *
 * @param navController Used to handle navigation actions from the home screen.
 */
/**
 * Displays the main home screen with centered app title, subtitle, beta status, and navigation buttons.
 *
 * The screen presents the app's name ("AuraFrameFX Alpha"), a subtitle ("Trinity AI System"), and a beta status message.
 * Two buttons are provided: one navigates to the AI Chat screen, and the other is a placeholder for future settings functionality.
 *
 * @param navController Used to handle navigation actions from the home screen.
 */
/**
 * Displays the main home screen with app title, subtitle, beta status, and navigation buttons.
 *
 * The screen centers its content both vertically and horizontally, presenting the app's name,
 * a subtitle, and a beta status message. It provides a button to navigate to the AI Chat screen
 * and a placeholder button for future settings navigation.
 *
 * @param navController Used to handle navigation actions from the home screen.
 */
@Composable
fun HomeScreen(navController: NavHostController) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "AuraFrameFX Alpha",
            style = MaterialTheme.typography.headlineLarge
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = "Trinity AI System",
            style = MaterialTheme.typography.headlineMedium
        )
        Spacer(modifier = Modifier.height(32.dp))
        Text(
            text = "Beta Build - Core AI Systems Active",
            style = MaterialTheme.typography.bodyLarge
        )
        Spacer(modifier = Modifier.height(16.dp))
        Button(
            onClick = { navController.navigate("ai_chat") }
        ) {
            Text("AI Chat (Beta)")
        }
        Spacer(modifier = Modifier.height(16.dp))
        Button(
            onClick = { /* TODO: Add navigation */ }
        ) {
            Text("Settings")
        }
    }
}

/**
 * Displays the AI chat interface screen with a centered headline and status message.
 *
 * Shows a large headline and a status message, both centered vertically and horizontally within the screen.
 */
@Composable
fun BetaAiChatScreen() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "AI Chat Interface",
            style = MaterialTheme.typography.headlineLarge
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = "Trinity AI agents ready",
            style = MaterialTheme.typography.bodyLarge
        )
    }
}
