package dev.aurakai.auraframefx.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import dev.aurakai.auraframefx.R
import dev.aurakai.auraframefx.ui.theme.NeonBlue
import dev.aurakai.auraframefx.ui.theme.NeonPurple
import dev.aurakai.auraframefx.ui.theme.NeonTeal

<<<<<<< HEAD
/**
 * Displays the main conference room UI, including agent selection, recording and transcription controls, chat interface, and message input.
 *
 * This composable manages local state for the selected agent, recording, and transcription status. It provides interactive controls for switching agents, starting/stopping recording and transcription, and a placeholder chat interface with a message input area.
 */
@OptIn(ExperimentalMaterial3Api::class)
=======
@OptIn(ExperimentalMaterial3Api::class)
/**
 * Displays the conference room UI with agent selection, recording and transcription controls, chat area, and message input.
 *
 * Users can select an agent, toggle recording and transcription states, and view a placeholder chat area. Message input and settings actions are present but not yet implemented.
 */
>>>>>>> pr458merge
/**
 * Displays the conference room interface with agent selection, recording and transcription controls, chat area, and message input.
 *
 * Users can choose an agent, toggle recording and transcription states, and compose messages. Chat display and message handling are not yet implemented.
 */
@Composable
fun ConferenceRoomScreen() {
    var selectedAgent by remember { mutableStateOf("Aura") }
    var isRecording by remember { mutableStateOf(false) }
    var isTranscribing by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        // Header
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = stringResource(R.string.conference_room),
                style = MaterialTheme.typography.headlineMedium,
                color = NeonTeal
            )

            IconButton(
                onClick = { /* TODO: Open settings */ }
            ) {
                Icon(
                    Icons.Default.Settings,
                    contentDescription = stringResource(R.string.settings_action),
                    tint = NeonPurple
                )
            }
        }

        // Agent Selection
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 16.dp),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            AgentButton(
                agent = stringResource(R.string.agent_aura),
                isSelected = selectedAgent == stringResource(R.string.agent_aura),
                onClick = { selectedAgent = stringResource(R.string.agent_aura) }
            )
            AgentButton(
                agent = stringResource(R.string.agent_kai),
                isSelected = selectedAgent == stringResource(R.string.agent_kai),
                onClick = { selectedAgent = stringResource(R.string.agent_kai) }
            )
            AgentButton(
                agent = stringResource(R.string.agent_cascade),
                isSelected = selectedAgent == stringResource(R.string.agent_cascade),
                onClick = { selectedAgent = stringResource(R.string.agent_cascade) }
            )
        }

        // Recording Controls
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 16.dp),
            horizontalArrangement = Arrangement.Center
        ) {
            RecordingButton(
                isRecording = isRecording,
                onClick = { isRecording = !isRecording }
            )

            TranscribeButton(
                isTranscribing = isTranscribing,
                onClick = { isTranscribing = !isTranscribing }
            )
        }

        // Chat Interface
        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            reverseLayout = true
        ) {
            // TODO: Add chat messages
        }

        // Input Area
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            TextField(
                value = "",
                onValueChange = { /* TODO: Handle input */ },
                placeholder = { Text("Type your message...") },
                modifier = Modifier
                    .weight(1f)
                    .padding(end = 8.dp),
                colors = TextFieldDefaults.colors(
                    focusedContainerColor = NeonTeal.copy(alpha = 0.1f),
                    unfocusedContainerColor = NeonTeal.copy(alpha = 0.1f)
                )
            )

            IconButton(
                onClick = { /* TODO: Send message */ }
            ) {
                Icon(
                    Icons.Default.Send,
                    contentDescription = "Send",
                    tint = NeonBlue
                )
            }
        }
    }
}

/**
 * Displays a button for selecting an agent, visually highlighting it when selected.
 *
 * The button shows the agent's name and applies a distinct style if it is the currently selected agent. Invokes the provided callback when pressed.
 *
 * @param agent The name of the agent to display on the button.
 * @param isSelected Whether this agent is currently selected.
 * @param onClick Callback invoked when the button is clicked.
 */
@Composable
fun AgentButton(
    agent: String,
    isSelected: Boolean,
    onClick: () -> Unit,
) {
    val backgroundColor = if (isSelected) NeonTeal else Color.Black
    val contentColor = if (isSelected) Color.White else NeonTeal

    Button(
        onClick = onClick,
        colors = ButtonDefaults.buttonColors(
            containerColor = backgroundColor,
            contentColor = contentColor
        ),
        modifier = Modifier
            .weight(1f)
            .padding(horizontal = 8.dp)
    ) {
        Text(
            text = agent,
            style = MaterialTheme.typography.labelLarge
        )
    }
}

/**
 * Displays a toggle button for controlling recording state, showing a red stop icon when recording is active and a purple circle icon when inactive.
 *
 * The button updates its icon, color, and accessibility description according to the current recording status.
 *
 * @param isRecording Indicates whether recording is currently active.
 * @param onClick Invoked when the button is pressed.
 */
@Composable
fun RecordingButton(
    isRecording: Boolean,
    onClick: () -> Unit,
) {
    val icon = if (isRecording) Icons.Default.Stop else Icons.Default.Circle
    val color = if (isRecording) Color.Red else NeonPurple

    IconButton(
        onClick = onClick,
        modifier = Modifier.size(64.dp)
    ) {
        Icon(
            icon,
            contentDescription = if (isRecording) stringResource(R.string.stop_recording) else stringResource(
                R.string.start_recording
            ),
            tint = color
        )
    }
}

/**
 * Displays a toggle button for starting or stopping transcription in the conference room UI.
 *
 * Shows a red stop icon when transcription is active, or a neon blue phone icon when inactive. The icon and accessibility description update according to the transcription state.
 *
 * @param isTranscribing Indicates whether transcription is currently active.
 * @param onClick Invoked when the button is pressed to toggle transcription.
 */
@Composable
fun TranscribeButton(
    isTranscribing: Boolean,
    onClick: () -> Unit,
) {
    val icon = if (isTranscribing) Icons.Default.Stop else Icons.Default.Phone
    val color = if (isTranscribing) Color.Red else NeonBlue

    IconButton(
        onClick = onClick,
        modifier = Modifier.size(64.dp)
    ) {
        Icon(
            icon,
            contentDescription = if (isTranscribing) stringResource(R.string.stop_transcription) else stringResource(
                R.string.start_transcription
            ),
            tint = color
        )
    }
}

/**
 * Provides a design-time preview of the ConferenceRoomScreen composable wrapped in a MaterialTheme.
 *
 * Intended for use with Android Studio's Compose preview to visualize the conference room UI layout and styling.
 */
@Composable
@Preview(showBackground = true)
fun ConferenceRoomScreenPreview() {
    MaterialTheme {
        ConferenceRoomScreen()
    }
}

