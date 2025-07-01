package dev.aurakai.auraframefx.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color // Corrected Color import
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
// dev.aurakai.auraframefx.system.homescreen.* // Wildcard removed, will use specific imports
import dev.aurakai.auraframefx.system.homescreen.model.HomeScreenTransitionType // Added import
import dev.aurakai.auraframefx.viewmodel.HomeScreenTransitionViewModel // Added import
import java.util.Locale

// Assuming DigitalTransitionRow is in the same package or needs specific import if error persists.
// For now, will rely on it being in dev.aurakai.auraframefx.ui.screens

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreenTransitionScreen(
    viewModel: HomeScreenTransitionViewModel = hiltViewModel(),
) {
    val currentConfig by viewModel.currentConfig.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Home Screen Transitions") },
                navigationIcon = {
                    IconButton(onClick = { /* TODO: Navigate back */ }) {
                        Icon(Icons.Default.ArrowBack, "Back")
                    }
                }
            )
        },
        floatingActionButton = {
            FloatingActionButton(
                onClick = { viewModel.resetToDefault() }
            ) {
                Icon(Icons.Default.Restore, "Reset")
            }
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Transition Type Section
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xFF00FFCC).copy(alpha = 0.1f)
                )
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp)
                ) {
                    Text(
                        text = "Transition Type",
                        style = MaterialTheme.typography.titleMedium
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    TransitionTypePicker(
                        currentType = currentConfig?.type ?: HomeScreenTransitionType.GLOBE_ROTATE,
                        onTypeSelected = { type -> viewModel.updateTransitionType(type) }
                    )
                }
            }

            // Duration Section
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xFF00FFCC).copy(alpha = 0.1f)
                )
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp)
                ) {
                    Text(
                        text = "Duration",
                        style = MaterialTheme.typography.titleMedium
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    DurationSlider(
                        currentDuration = currentConfig?.duration ?: 500,
                        onDurationChanged = { duration ->
                            viewModel.updateTransitionDuration(
                                duration
                            )
                        }
                    )
                }
            }

            // Properties Section
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xFF00FFCC).copy(alpha = 0.1f)
                )
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp)
                ) {
                    Text(
                        text = "Properties",
                        style = MaterialTheme.typography.titleMedium
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    TransitionPropertiesEditor(
                        currentProperties = currentConfig?.properties ?: emptyMap(),
                        onPropertiesChanged = { properties ->
                            viewModel.updateTransitionProperties(
                                properties
                            )
                        }
                    )
                }
            }
        }
    }
}

@Composable
fun TransitionTypePicker(
    currentType: HomeScreenTransitionType,
    onTypeSelected: (HomeScreenTransitionType) -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        // Basic Transitions
        Text(
            text = "Basic Transitions",
            style = MaterialTheme.typography.titleSmall
        )
        BasicTransitionRow(
            currentType = currentType,
            onTypeSelected = onTypeSelected
        )

        // Card Stack Transitions
        Text(
            text = "Card Stack Transitions",
            style = MaterialTheme.typography.titleSmall
        )
        CardStackTransitionRow(
            currentType = currentType,
            onTypeSelected = onTypeSelected
        )

        // 3D Transitions
        Text(
            text = "3D Transitions",
            style = MaterialTheme.typography.titleSmall
        )
        ThreeDTransitionRow(
            currentType = currentType,
            onTypeSelected = onTypeSelected
        )

        // Globe Transitions
        Text(
            text = "Globe Transitions",
            style = MaterialTheme.typography.titleSmall
        )
        GlobeTransitionRow(
            currentType = currentType,
            onTypeSelected = onTypeSelected
        )

        // Fan Transitions
        Text(
            text = "Fan Transitions",
            style = MaterialTheme.typography.titleSmall
        )
        FanTransitionRow(
            currentType = currentType,
            onTypeSelected = onTypeSelected
        )

        // Spread Transitions
        Text(
            text = "Spread Transitions",
            style = MaterialTheme.typography.titleSmall
        )
        SpreadTransitionRow(
            currentType = currentType,
            onTypeSelected = onTypeSelected
        )
        // Digital/Deconstruct/Hologram Transitions
        Text(
            text = "Digital/Hologram Transitions",
            style = MaterialTheme.typography.titleSmall
        )
        DigitalTransitionRow(
            currentType = currentType,
            onTypeSelected = onTypeSelected
        )
        TransitionButton(
            label = "Hologram Forming",
            isSelected = currentType == HomeScreenTransitionType.HOLOGRAM_FORMING,
            onClick = { onTypeSelected(HomeScreenTransitionType.HOLOGRAM_FORMING) }
        )
    }
}

@Composable
fun DurationSlider(
    currentDuration: Int,
    onDurationChanged: (Int) -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = "Duration (ms)",
            style = MaterialTheme.typography.bodyMedium
        )
        Slider(
            value = currentDuration.toFloat(),
            onValueChange = { onDurationChanged(it.toInt()) },
            valueRange = 100f..2000f,
            steps = 19,
            modifier = Modifier.fillMaxWidth()
        )
        Text(
            text = "${currentDuration}ms",
            style = MaterialTheme.typography.bodySmall
        )
    }
}

@Composable
fun TransitionPropertiesEditor(
    currentProperties: Map<String, Any>,
    onPropertiesChanged: (Map<String, Any>) -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        // Angle
        PropertySlider(
            label = "Angle",
            value = currentProperties["angle"] as? Float ?: 0f,
            onValueChange = { angle ->
                onPropertiesChanged(currentProperties + ("angle" to angle))
            },
            valueRange = 0f..360f
        )

        // Scale
        PropertySlider(
            label = "Scale",
            value = currentProperties["scale"] as? Float ?: 1f,
            onValueChange = { scale ->
                onPropertiesChanged(currentProperties + ("scale" to scale))
            },
            valueRange = 0.1f..2f
        )

        // Offset
        PropertySlider(
            label = "Offset",
            value = currentProperties["offset"] as? Float ?: 0f,
            onValueChange = { offset ->
                onPropertiesChanged(currentProperties + ("offset" to offset))
            },
            valueRange = -100f..100f
        )

        // Amplitude
        PropertySlider(
            label = "Amplitude",
            value = currentProperties["amplitude"] as? Float ?: 0f,
            onValueChange = { amplitude ->
                onPropertiesChanged(currentProperties + ("amplitude" to amplitude))
            },
            valueRange = 0f..1f
        )

        // Frequency
        PropertySlider(
            label = "Frequency",
            value = currentProperties["frequency"] as? Float ?: 0f,
            onValueChange = { frequency ->
                onPropertiesChanged(currentProperties + ("frequency" to frequency))
            },
            valueRange = 0f..2f
        )

        // Blur
        PropertySlider(
            label = "Blur",
            value = currentProperties["blur"] as? Float ?: 0f,
            onValueChange = { blur ->
                onPropertiesChanged(currentProperties + ("blur" to blur))
            },
            valueRange = 0f..100f
        )

        // Spread
        PropertySlider(
            label = "Spread",
            value = currentProperties["spread"] as? Float ?: 0f,
            onValueChange = { spread ->
                onPropertiesChanged(currentProperties + ("spread" to spread))
            },
            valueRange = 0f..1f
        )
    }
}

@Composable
fun PropertySlider(
    label: String,
    value: Float,
    onValueChange: (Float) -> Unit,
    valueRange: ClosedFloatingPointRange<Float>,
) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium
        )
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = valueRange,
            modifier = Modifier.fillMaxWidth()
        )
        Text(
            text = String.format(Locale.ROOT, "%.2f", value),
            style = MaterialTheme.typography.bodySmall
        )
    }
}

@Composable
fun BasicTransitionRow(
    currentType: HomeScreenTransitionType,
    onTypeSelected: (HomeScreenTransitionType) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        TransitionButton(
            label = "Slide Left",
            isSelected = currentType == HomeScreenTransitionType.SLIDE_LEFT,
            onClick = { onTypeSelected(HomeScreenTransitionType.SLIDE_LEFT) }
        )
        TransitionButton(
            label = "Slide Right",
            isSelected = currentType == HomeScreenTransitionType.SLIDE_RIGHT,
            onClick = { onTypeSelected(HomeScreenTransitionType.SLIDE_RIGHT) }
        )
        TransitionButton(
            label = "Slide Up",
            isSelected = currentType == HomeScreenTransitionType.SLIDE_UP,
            onClick = { onTypeSelected(HomeScreenTransitionType.SLIDE_UP) }
        )
        TransitionButton(
            label = "Slide Down",
            isSelected = currentType == HomeScreenTransitionType.SLIDE_DOWN,
            onClick = { onTypeSelected(HomeScreenTransitionType.SLIDE_DOWN) }
        )
    }
}

@Composable
fun CardStackTransitionRow(
    currentType: HomeScreenTransitionType,
    onTypeSelected: (HomeScreenTransitionType) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        TransitionButton(
            label = "Stack Slide",
            isSelected = currentType == HomeScreenTransitionType.STACK_SLIDE,
            onClick = { onTypeSelected(HomeScreenTransitionType.STACK_SLIDE) }
        )
        TransitionButton(
            label = "Stack Fade",
            isSelected = currentType == HomeScreenTransitionType.STACK_FADE,
            onClick = { onTypeSelected(HomeScreenTransitionType.STACK_FADE) }
        )
        TransitionButton(
            label = "Stack Scale",
            isSelected = currentType == HomeScreenTransitionType.STACK_SCALE,
            onClick = { onTypeSelected(HomeScreenTransitionType.STACK_SCALE) }
        )
        TransitionButton(
            label = "Stack Rotate",
            isSelected = currentType == HomeScreenTransitionType.STACK_ROTATE,
            onClick = { onTypeSelected(HomeScreenTransitionType.STACK_ROTATE) }
        )
    }
}

@Composable
fun ThreeDTransitionRow(
    currentType: HomeScreenTransitionType,
    onTypeSelected: (HomeScreenTransitionType) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        TransitionButton(
            label = "3D Rotate",
            isSelected = currentType == HomeScreenTransitionType.STACK_ROTATE_3D,
            onClick = { onTypeSelected(HomeScreenTransitionType.STACK_ROTATE_3D) }
        )
        TransitionButton(
            label = "3D Scale",
            isSelected = currentType == HomeScreenTransitionType.STACK_SCALE_3D,
            onClick = { onTypeSelected(HomeScreenTransitionType.STACK_SCALE_3D) }
        )
        TransitionButton(
            label = "3D Slide",
            isSelected = currentType == HomeScreenTransitionType.STACK_SLIDE_3D,
            onClick = { onTypeSelected(HomeScreenTransitionType.STACK_SLIDE_3D) }
        )
        TransitionButton(
            label = "3D Wave",
            isSelected = currentType == HomeScreenTransitionType.STACK_WAVE_3D,
            onClick = { onTypeSelected(HomeScreenTransitionType.STACK_WAVE_3D) }
        )
    }
}

@Composable
fun GlobeTransitionRow(
    currentType: HomeScreenTransitionType,
    onTypeSelected: (HomeScreenTransitionType) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        TransitionButton(
            label = "Globe Rotate",
            isSelected = currentType == HomeScreenTransitionType.GLOBE_ROTATE,
            onClick = { onTypeSelected(HomeScreenTransitionType.GLOBE_ROTATE) }
        )
        TransitionButton(
            label = "Globe Scale",
            isSelected = currentType == HomeScreenTransitionType.GLOBE_SCALE,
            onClick = { onTypeSelected(HomeScreenTransitionType.GLOBE_SCALE) }
        )
        TransitionButton(
            label = "Globe Pulse",
            isSelected = currentType == HomeScreenTransitionType.GLOBE_PULSE,
            onClick = { onTypeSelected(HomeScreenTransitionType.GLOBE_PULSE) }
        )
        TransitionButton(
            label = "Globe Glow",
            isSelected = currentType == HomeScreenTransitionType.GLOBE_GLOW,
            onClick = { onTypeSelected(HomeScreenTransitionType.GLOBE_GLOW) }
        )
    }
}

@Composable
fun FanTransitionRow(
    currentType: HomeScreenTransitionType,
    onTypeSelected: (HomeScreenTransitionType) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        TransitionButton(
            label = "Fan In",
            isSelected = currentType == HomeScreenTransitionType.FAN_IN,
            onClick = { onTypeSelected(HomeScreenTransitionType.FAN_IN) }
        )
        TransitionButton(
            label = "Fan Out",
            isSelected = currentType == HomeScreenTransitionType.FAN_OUT,
            onClick = { onTypeSelected(HomeScreenTransitionType.FAN_OUT) }
        )
        TransitionButton(
            label = "Fan Rotate",
            isSelected = currentType == HomeScreenTransitionType.FAN_ROTATE,
            onClick = { onTypeSelected(HomeScreenTransitionType.FAN_ROTATE) }
        )
        TransitionButton(
            label = "Fan Scale",
            isSelected = currentType == HomeScreenTransitionType.FAN_SCALE,
            onClick = { onTypeSelected(HomeScreenTransitionType.FAN_SCALE) }
        )
    }
}

@Composable
fun SpreadTransitionRow(
    currentType: HomeScreenTransitionType,
    onTypeSelected: (HomeScreenTransitionType) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        TransitionButton(
            label = "Spread In",
            isSelected = currentType == HomeScreenTransitionType.SPREAD_IN,
            onClick = { onTypeSelected(HomeScreenTransitionType.SPREAD_IN) }
        )
        TransitionButton(
            label = "Spread Out",
            isSelected = currentType == HomeScreenTransitionType.SPREAD_OUT,
            onClick = { onTypeSelected(HomeScreenTransitionType.SPREAD_OUT) }
        )
        TransitionButton(
            label = "Spread Rotate",
            isSelected = currentType == HomeScreenTransitionType.SPREAD_ROTATE,
            onClick = { onTypeSelected(HomeScreenTransitionType.SPREAD_ROTATE) }
        )
        TransitionButton(
            label = "Spread Scale",
            isSelected = currentType == HomeScreenTransitionType.SPREAD_SCALE,
            onClick = { onTypeSelected(HomeScreenTransitionType.SPREAD_SCALE) }
        )
    }
}

@Composable
fun TransitionButton(
    label: String,
    isSelected: Boolean,
    onClick: () -> Unit,
) {
    Button(
        onClick = onClick,
        colors = ButtonDefaults.buttonColors(
            containerColor = if (isSelected) Color(0xFF00FFCC) else Color(0x3300FFCC)
        )
    ) {
        Text(label)
    }
}
