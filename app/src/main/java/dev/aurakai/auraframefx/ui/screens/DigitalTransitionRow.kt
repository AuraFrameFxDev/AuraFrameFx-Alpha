package dev.aurakai.auraframefx.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import dev.aurakai.auraframefx.system.homescreen.HomeScreenTransitionType

@Composable
fun DigitalTransitionRow(
    currentType: HomeScreenTransitionType,
    onTypeSelected: (HomeScreenTransitionType) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        Button(
            onClick = { onTypeSelected(HomeScreenTransitionType.DIGITAL_DECONSTRUCT) },
            enabled = currentType != HomeScreenTransitionType.DIGITAL_DECONSTRUCT
        ) {
            Text("Digital Deconstruct")
        }
        Button(
            onClick = { onTypeSelected(HomeScreenTransitionType.DIGITAL_RECONSTRUCT) },
            enabled = currentType != HomeScreenTransitionType.DIGITAL_RECONSTRUCT
        ) {
            Text("Digital Reconstruct")
        }
    }
}
