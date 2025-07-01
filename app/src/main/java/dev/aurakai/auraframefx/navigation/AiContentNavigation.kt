package dev.aurakai.auraframefx.navigation

import androidx.navigation.NavController
import androidx.navigation.NavGraphBuilder
import androidx.navigation.compose.composable
import dev.aurakai.auraframefx.ui.aicontent.AiContentScreen

/**
 * Navigation constants for AI content features
 */
object AiContentDestinations {
    const val AI_CONTENT_ROUTE = "ai_content"
}

/**
 * Extension function to add AI content related navigation to NavGraphBuilder
 */
fun NavGraphBuilder.aiContentNavigation() {
    composable(route = AiContentDestinations.AI_CONTENT_ROUTE) {
        AiContentScreen()
    }
}

/**
 * Extension function for NavController to navigate to the AI content screen
 */
fun NavController.navigateToAiContent() {
    navigate(AiContentDestinations.AI_CONTENT_ROUTE)
}
