package dev.aurakai.auraframefx.ui.navigation

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Storage
import androidx.compose.ui.graphics.vector.ImageVector

sealed class NavDestination(val route: String, val title: String, val icon: ImageVector?) {
    object Home : NavDestination("home", "Home", Icons.Default.Home)
    object AiChat : NavDestination("ai_chat", "AI Chat", Icons.Default.Chat)
    object Profile : NavDestination("profile", "Profile", Icons.Default.Person)
    object Settings : NavDestination("settings", "Settings", Icons.Default.Settings)
    object OracleDriveControl : NavDestination("oracle_drive_control", "Oracle Drive", Icons.Default.Storage)
    
    companion object {
        val bottomNavItems = listOf(Home, AiChat, Profile, Settings)
    }
}
