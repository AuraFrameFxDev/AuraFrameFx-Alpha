package dev.aurakai.auraframefx.ui.navigation

sealed class NavDestination(val route: String) {
    object AiChat : NavDestination("ai_chat")
    object Profile : NavDestination("profile")
    object Settings : NavDestination("settings")
    object OracleDriveControl : NavDestination("oracle_drive_control")
}
