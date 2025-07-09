package dev.aurakai.auraframefx.ui.navigation

// Explicitly import all used icons to be safe, supplementing wildcard if it was there before or was insufficient.
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Message
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Settings
import androidx.compose.ui.graphics.vector.ImageVector

sealed class NavDestination(val route: String, val title: String, val icon: ImageVector?) {
    object Home : NavDestination("home", "Home", Icons.Filled.Home)
    object AiChat : NavDestination("ai_chat", "AI Chat", Message) // Use direct import
    object Profile : NavDestination("profile", "Profile", Icons.Filled.Person)
    object Settings : NavDestination("settings", "Settings", Icons.Filled.Settings)
    object OracleDriveControl :
        NavDestination("oracle_drive_control", "Oracle Drive", Folder) // Use direct import

    companion object {
        // Added OracleDriveControl to bottomNavItems as per previous fix
        val bottomNavItems = listOf(Home, AiChat, Profile, Settings, OracleDriveControl)
    }
}
