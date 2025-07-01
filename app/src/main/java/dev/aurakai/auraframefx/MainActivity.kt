package dev.aurakai.auraframefx

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.navigation.compose.rememberNavController
import dev.aurakai.auraframefx.ui.animation.digitalPixelEffect
import dev.aurakai.auraframefx.ui.components.BottomNavigationBar
import dev.aurakai.auraframefx.ui.navigation.AppNavGraph
import dev.aurakai.auraframefx.ui.theme.AuraFrameFXThemeCompat

// Using Jetpack Navigation 3 with built-in animation support

class MainActivity : ComponentActivity() {
    /**
     * Initializes the activity and sets the Compose UI content to the main screen within the app's theme.
     * Uses digital transition animations for a cyberpunk aesthetic.
     *
     * @param savedInstanceState The saved state of the activity, or null if none exists.
     */
    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            AuraFrameFXThemeCompat {
                MainScreen()
            }
        }
    }

    /**
     * Called when the activity is about to be destroyed.
     *
     * Override this method to perform cleanup operations before the activity is removed from memory.
     */
    override fun onDestroy() {
        super.onDestroy()
        // Perform any cleanup here if needed
    }
}

/**
 * Displays the main screen layout with a bottom navigation bar and navigation graph.
 *
 * Sets up the app's primary UI structure using a Scaffold, integrating navigation and content padding.
 * Applies cyberpunk-style digital transition effects between screens.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen() {
    // Use Jetpack Navigation 3's nav controller for digital transitions
    val navController = rememberNavController()

    // State to control digital effects
    var showDigitalEffects by remember { mutableStateOf(true) }

    Scaffold(
        bottomBar = { BottomNavigationBar(navController = navController) }
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                // Apply our custom digital effects
                .then(
                    if (showDigitalEffects) {
                        Modifier
                            .digitalPixelEffect(visible = true)
                            .digitalScanlineEffect(visible = true)
                    } else {
                        Modifier
                    }
                )
        ) {
            AppNavGraph(navController = navController)
        }
    }
}

/**
 * Displays a preview of the main screen composable within the app's theme for design-time visualization.
 */
@Preview(showBackground = true)
@Composable
fun MainScreenPreview() {
    AuraFrameFXThemeCompat {
        MainScreen()
    }
}
