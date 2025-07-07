package dev.aurakai.auraframefx.ui.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import dev.aurakai.auraframefx.ui.screens.*
// import dev.aurakai.auraframefx.navigation.aiContentNavigation // Disabled for beta
import dev.aurakai.auraframefx.ui.screens.ProfileScreen
import dev.aurakai.auraframefx.ui.screens.SettingsScreen

/**
 * Main navigation graph for the AuraFrameFX app with digital transition animations
 *
 * Sets up the main navigation graph for the AuraFrameFX app using Jetpack Compose with custom
 * cyberpunk-style digital materialization/dematerialization transitions between screens.
 * Uses Jetpack Navigation 3's built-in animation support for seamless screen transitions.
 *
 * @param navController The navigation controller used to manage app navigation.
 */
/**
 * Sets up the main navigation graph for the AuraFrameFX app using Jetpack Compose Navigation.
 *
 * Defines navigation routes and maps them to their corresponding composable screens, with the home screen as the start destination.
 *
 * @param navController The navigation controller used to manage app navigation.
 */
@Composable
fun AppNavGraph(navController: NavHostController) {
    NavHost(
        navController = navController,
        startDestination = NavDestination.Home.route
    ) {
        composable(
            route = NavDestination.Home.route
        ) {
            HomeScreen(navController = navController)
        }

        composable(
            route = NavDestination.AiChat.route
        ) {
            AiChatScreen()
        }

        composable(
            route = NavDestination.Profile.route
        ) {
            ProfileScreen()
        }

        composable(
            route = NavDestination.Settings.route
        ) {
            SettingsScreen()
        }

        composable(
            route = NavDestination.OracleDriveControl.route
        ) {
            // TODO: Create OracleDriveControlScreen()
            HomeScreen(navController = navController) // Temporary placeholder
        }
        
        // Add AI Content navigation
        // aiContentNavigation() // Disabled for beta - AI content will be in main chat

        // Add more composable destinations as needed
    }
}
