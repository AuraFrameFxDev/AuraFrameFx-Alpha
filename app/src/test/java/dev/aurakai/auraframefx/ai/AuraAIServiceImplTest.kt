// File: app/src/main/java/dev/aurakai/auraframefx/ui/navigation/NavDestination.kt
package dev.aurakai.auraframefx.ui.navigation

import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Message
import androidx.compose.material.icons.filled.Folder

sealed class NavDestination(val route: String, val icon: ImageVector) {
    object Home : NavDestination("home", Icons.Filled.Home)
    object Messages : NavDestination("messages", Icons.Filled.Message)
    object Files : NavDestination("files", Icons.Filled.Folder)
}

// File: app/src/main/java/dev/aurakai/auraframefx/ui/screens/ConferenceRoomScreen.kt
package dev.aurakai.auraframefx.ui.screens

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.weight
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier

@Composable
fun ConferenceRoomScreen() {
    Column(modifier = Modifier.fillMaxSize()) {
        Box(modifier = Modifier.weight(1f)) {
            // Participant list or video grid
        }
        Row(modifier = Modifier.weight(1f)) {
            // Chat area
        }
    }
}

// File: app/src/main/java/dev/aurakai/auraframefx/ui/screens/HomeScreen.kt
package dev.aurakai.auraframefx.ui.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.material.Button
import androidx.compose.material.Text
import androidx.compose.runtime.Composable

@Composable
fun HomeScreen(onClick: () -> Unit) {
    Column {
        Button(onClick = onClick) {
            Text(text = "Home")
        }
        // Other home screen content...
    }
}