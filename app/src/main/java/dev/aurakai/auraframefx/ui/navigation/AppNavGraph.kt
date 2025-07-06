package dev.aurakai.auraframefx.ui.navigation

import androidx.compose.animation.AnimatedContentTransitionScope
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import dev.aurakai.auraframefx.navigation.aiContentNavigation
import dev.aurakai.auraframefx.ui.transitions.DigitalTransitions.EnterDigitalMaterialization
import dev.aurakai.auraframefx.ui.transitions.DigitalTransitions.ExitDigitalDematerialization
import dev.aurakai.auraframefx.ui.screens.AiChatScreen
import dev.aurakai.auraframefx.ui.screens.HomeScreen
import dev.aurakai.auraframefx.ui.screens.OracleDriveControlScreen
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
@Composable
fun AppNavGraph(navController: NavHostController) {
    NavHost(
        navController = navController,
        startDestination = NavDestination.Home.route
    ) {
        composable(
            route = NavDestination.Home.route,
            enterTransition = {
                slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Left) +
                        fadeIn(animationSpec = tween(300)) +
                        EnterDigitalMaterialization
            },
            exitTransition = {
                slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Left) +
                        fadeOut(animationSpec = tween(300)) +
                        ExitDigitalDematerialization
            }
        ) {
            HomeScreen(navController = navController)
        }

        composable(
            route = NavDestination.AiChat.route,
            enterTransition = {
                slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Left) +
                        fadeIn(animationSpec = tween(300)) +
                        EnterDigitalMaterialization
            },
            exitTransition = {
                slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Right) +
                        fadeOut(animationSpec = tween(300)) +
                        ExitDigitalDematerialization
            }
        ) {
            AiChatScreen()
        }

        composable(
            route = NavDestination.Profile.route,
            enterTransition = {
                slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Up) +
                        fadeIn(animationSpec = tween(300)) +
                        EnterDigitalMaterialization
            },
            exitTransition = {
                slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Down) +
                        fadeOut(animationSpec = tween(300)) +
                        ExitDigitalDematerialization
            }
        ) {
            ProfileScreen()
        }

        composable(
            route = NavDestination.Settings.route,
            enterTransition = {
                slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Down) +
                        fadeIn(animationSpec = tween(300)) +
                        EnterDigitalMaterialization
            },
            exitTransition = {
                slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Up) +
                        fadeOut(animationSpec = tween(300)) +
                        ExitDigitalDematerialization
            }
        ) {
            SettingsScreen()
        }

        composable(
            route = NavDestination.OracleDriveControl.route,
            enterTransition = {
                slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Left) +
                        fadeIn(animationSpec = tween(300)) +
                        EnterDigitalMaterialization
            },
            exitTransition = {
                slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Right) +
                        fadeOut(animationSpec = tween(300)) +
                        ExitDigitalDematerialization
            }
        ) {
            OracleDriveControlScreen()
        }
        
        // Add AI Content navigation
        aiContentNavigation()

        // Add more composable destinations as needed
    }
}
