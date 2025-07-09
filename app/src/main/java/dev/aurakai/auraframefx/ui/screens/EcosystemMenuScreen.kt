package dev.aurakai.auraframefx.ui.screens

// import androidx.compose.ui.tooling.preview.Preview

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import dev.aurakai.auraframefx.system.homescreen.HomeScreenTransitionType
import dev.aurakai.auraframefx.ui.components.HologramTransition
import dev.aurakai.auraframefx.ui.components.DigitalTransitionRow

/**
 * Displays the Ecosystem Menu screen with an optional hologram transition and transition type selector.
 *
 * @param transitionType The type of home screen transition to display in the selector.
 * @param showHologram Whether to show the hologram transition effect.
 */
@Composable
fun EcosystemMenuScreen(
    transitionType: HomeScreenTransitionType = HomeScreenTransitionType.DIGITAL_DECONSTRUCT,
    showHologram: Boolean = true,
) {
    HologramTransition(visible = showHologram) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(text = "Ecosystem Menu Screen (Placeholder)")
                DigitalTransitionRow(
                    currentType = transitionType,
                    onTypeSelected = {}
                )
            }
        }
    }
}

// @Preview(showBackground = true)
// @Composable
// fun EcosystemMenuScreenPreview() { // Renamed
//     EcosystemMenuScreen()
// }
