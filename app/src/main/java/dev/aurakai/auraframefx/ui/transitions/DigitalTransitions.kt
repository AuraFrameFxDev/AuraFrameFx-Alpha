package dev.aurakai.auraframefx.ui.transitions

import androidx.compose.animation.EnterTransition
import androidx.compose.animation.ExitTransition
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally

/**
 * Digital transition effects for navigation
 */
object DigitalTransitions {
    
    val EnterDigitalMaterialization: EnterTransition = slideInHorizontally(
        initialOffsetX = { it }
    ) + fadeIn()
    
    val ExitDigitalDematerialization: ExitTransition = slideOutHorizontally(
        targetOffsetX = { -it }
    ) + fadeOut()
}
