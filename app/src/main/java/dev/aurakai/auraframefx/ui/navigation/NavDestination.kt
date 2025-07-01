package dev.aurakai.auraframefx.ui.navigation

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Settings
import androidx.compose.ui.graphics.vector.ImageVector
import dev.aurakai.auraframefx.ui.theme.AppStrings

/**
 * Navigation destinations for the app
 */
sealed class NavDestination(
    val route: String,
    val title: String,
    val icon: ImageVector? = null,
) {
    object Home : NavDestination(
        route = "home",
        title = AppStrings.NAV_HOME,
        icon = Icons.Default.Home
    )

    object AiChat : NavDestination(
        route = "ai_chat",
        title = AppStrings.NAV_AI_CHAT,
        icon = Icons.Default.Chat
    )

    object Profile : NavDestination(
        route = "profile",
        title = AppStrings.NAV_PROFILE,
        icon = Icons.Default.Person
    )

    object Settings : NavDestination(
        route = "settings",
        title = AppStrings.NAV_SETTINGS,
        icon = Icons.Default.Settings
    )

    object OracleDriveControl : NavDestination(
        route = "oracle_drive_control",
        title = "OracleDrive Control", // You may want to localize this
        icon = null // Optionally add an icon
    )

    // Add more destinations as needed

    companion object {
        val bottomNavItems = listOf(Home, AiChat, Profile, Settings)
        val all = listOf(Home, AiChat, Profile, Settings, OracleDriveControl)
    }
}
