package dev.aurakai.auraframefx.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
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
 * Displays the home screen with app title, status messages, and navigation buttons in a centered column layout.
 *
 * Presents the app name, a subtitle, and a status message, along with buttons to navigate to the AI chat screen or (in the future) the settings screen.
 */
/**
 * Displays the home screen with a centered column layout featuring the app title, subtitle, status message, and navigation buttons.
 *
 * Presents two buttons: one navigates to the AI chat screen, and the other is a placeholder for settings.
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
 * Composable that displays the AI chat interface with a centered headline and status message.
 *
 * Presents a large headline ("AI Chat Interface") and a readiness status ("Trinity AI agents ready") in a vertically and horizontally centered column layout.
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
