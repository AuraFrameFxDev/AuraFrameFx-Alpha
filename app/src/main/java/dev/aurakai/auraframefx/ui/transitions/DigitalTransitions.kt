package dev.aurakai.auraframefx.ui.transitions

<<<<<<< HEAD
import androidx.compose.animation.*
import androidx.compose.animation.core.*

/**
 * Synthwave-style digital transitions for AuraFrameFX
 * Pixel-by-pixel materialization and deconstruction effects
 */
object DigitalTransitions {
    
    /**
     * Digital materialization entrance transition
     * Pixels appear from static noise into coherent UI
     */
    val EnterDigitalMaterialization: EnterTransition = slideInHorizontally(
        animationSpec = tween(
            durationMillis = 800,
            easing = FastOutSlowInEasing
        ),
        initialOffsetX = { fullWidth -> fullWidth }
    ) + fadeIn(
        animationSpec = tween(
            durationMillis = 600,
            delayMillis = 200,
            easing = LinearOutSlowInEasing
        )
    ) + scaleIn(
        animationSpec = tween(
            durationMillis = 700,
            easing = FastOutSlowInEasing
        ),
        initialScale = 0.8f
    )
    
    /**
     * Digital deconstruction exit transition  
     * UI breaks down into pixels and static
     */
    val ExitDigitalDematerialization: ExitTransition = slideOutHorizontally(
        animationSpec = tween(
            durationMillis = 600,
            easing = FastOutLinearInEasing
        ),
        targetOffsetX = { fullWidth -> -fullWidth }
    ) + fadeOut(
        animationSpec = tween(
            durationMillis = 400,
            easing = FastOutLinearInEasing
        )
    ) + scaleOut(
        animationSpec = tween(
            durationMillis = 500,
            easing = FastOutLinearInEasing
        ),
        targetScale = 1.2f
    )
=======
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
>>>>>>> f692b48 (Pending changes exported from your codespace)
}
