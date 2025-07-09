package dev.aurakai.auraframefx.ui.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import dev.aurakai.auraframefx.ui.transitions.DigitalTransitions
import dev.aurakai.auraframefx.ui.screens.AiChatScreen
import dev.aurakai.auraframefx.ui.screens.HomeScreen
import dev.aurakai.auraframefx.ui.screens.oracledrive.OracleDriveControlScreen
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
 * Defines the available composable destinations and their routes, including Home, AI Chat, Profile, Settings, and Oracle Drive Control (currently a placeholder).
 *
 * @param navController The navigation controller used to manage app navigation.
 */
/**
 * Sets up the main navigation graph for the AuraFrameFX app using Jetpack Compose Navigation.
 *
 * Defines the available composable destinations and their associated routes, enabling navigation between screens such as Home, AI Chat, Profile, Settings, and Oracle Drive Control.
 *
 * @param navController The navigation controller used to manage navigation between composable destinations.
 */
/**
 * Sets up the main navigation graph for the AuraFrameFX app using the provided navigation controller.
 *
 * Configures navigation routes for Home, AI Chat, Profile, Settings, and Oracle Drive Control screens.
 *
 * @param navController The navigation controller used to manage navigation between screens.
 */
/**
 * Sets up the main navigation graph for the application, mapping navigation routes to their corresponding composable screens.
 *
 * @param navController The navigation controller used to manage navigation between screens.
 */
/**
 * Sets up the main navigation graph for the AuraFrameFX app using Jetpack Compose Navigation.
 *
 * Defines composable destinations for Home, AI Chat, Profile, Settings, and Oracle Drive Control screens, enabling navigation between them via the provided navigation controller.
 *
 * @param navController The navigation controller used to manage navigation between screens.
 */
/**
 * Sets up the main navigation graph for the AuraFrameFX app using Jetpack Compose Navigation.
 *
 * Configures navigation between Home, AI Chat, Profile, Settings, and Oracle Drive Control screens using the provided navigation controller.
 *
 * @param navController The navigation controller that manages app navigation.
 */
/**
 * Sets up the main navigation graph for the AuraFrameFX app using Jetpack Compose Navigation.
 *
 * Configures navigation routes for the Home, AI Chat, Profile, Settings, and Oracle Drive Control screens, mapping each route to its corresponding composable screen.
 */
/**
 * Sets up the main navigation graph for the AuraFrameFX app using Jetpack Compose Navigation.
 *
 * Defines composable destinations for Home, AI Chat, Profile, Settings, and Oracle Drive Control screens, enabling navigation between these primary app sections.
 *
 * @param navController The navigation controller used to manage navigation between screens.
 */
/**
 * Sets up the main navigation graph for the AuraFrameFX app using Jetpack Compose Navigation.
 *
 * Maps primary app routes to their corresponding composable screens, enabling navigation between Home, AI Chat, Profile, Settings, and Oracle Drive Control sections.
 *
 * @param navController The navigation controller used to manage app navigation.
 */
/**
 * Sets up the main navigation graph for the AuraFrameFX app using Jetpack Compose Navigation.
 *
 * Configures navigation routes for Home, AI Chat, Profile, Settings, and Oracle Drive Control screens, enabling navigation between them using the provided navigation controller.
 *
 * @param navController The navigation controller that manages navigation between app screens.
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
            // Fixed: Use actual OracleDriveControlScreen instead of placeholder
            dev.aurakai.auraframefx.ui.screens.oracledrive.OracleDriveControlScreen()
        }
        
        // Add AI Content navigation
        // aiContentNavigation() // Disabled for beta - AI content will be in main chat

        // Add more composable destinations as needed
    }
}
