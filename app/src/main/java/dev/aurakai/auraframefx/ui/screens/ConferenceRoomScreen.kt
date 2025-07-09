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

/**
 * Displays the main conference room UI, allowing users to select an agent, control recording and transcription, view chat messages, and send new messages.
 *
 * This composable manages local state for agent selection, recording, and transcription. Interactive elements for settings, chat, and message input are present but not yet implemented.
 */
@OptIn(ExperimentalMaterial3Api::class)
/**
 * Displays the main conference room UI, including agent selection, recording and transcription controls, chat area, and message input.
 *
 * This composable manages local state for the selected agent, recording, and transcription toggles. It presents a header, agent selection buttons, controls for recording and transcription, a placeholder for chat messages, and an input area for composing messages. Interactive elements such as settings, message sending, and chat display are present but not yet implemented.
 */
@Composable
fun ConferenceRoomScreen() {
    var selectedAgent by remember { mutableStateOf("Aura") } // Default to a non-resource string initially
    var isRecording by remember { mutableStateOf(false) }
    var isTranscribing by remember { mutableStateOf(false) }

    val agentAuraName = stringResource(R.string.agent_aura)
    val agentKaiName = stringResource(R.string.agent_kai)
    val agentCascadeName = stringResource(R.string.agent_cascade)

    // Initialize selectedAgent with a resource string if not already set,
    // or if the default "Aura" should map to the resource string.
    LaunchedEffect(key1 = agentAuraName) {
        if (selectedAgent == "Aura") { // Default literal matches default resource intent
            selectedAgent = agentAuraName
        }
    }


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
                agent = agentAuraName,
                isSelected = selectedAgent == agentAuraName,
                onClick = { selectedAgent = agentAuraName }
            )
            AgentButton(
                agent = agentKaiName,
                isSelected = selectedAgent == agentKaiName,
                onClick = { selectedAgent = agentKaiName }
            )
            AgentButton(
                agent = agentCascadeName,
                isSelected = selectedAgent == agentCascadeName,
                onClick = { selectedAgent = agentCascadeName }
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
 * Displays a selectable button for an agent with visual indication of selection state.
 *
 * The button's background and text color change based on whether it is selected.
 *
 * @param agent The name of the agent to display on the button.
 * @param isSelected Whether this agent is currently selected.
 * @param onClick Called when the button is pressed.
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
 * Displays a button for toggling the recording state, showing a stop icon when active and a record icon when inactive.
 *
 * @param isRecording True if recording is currently active; false otherwise.
 * @param onClick Called when the button is pressed to toggle recording.
 */
@Composable
fun RecordingButton(
    isRecording: Boolean,
    onClick: () -> Unit,
) {
    val icon = if (isRecording) Icons.Filled.Stop else Icons.Filled.FiberManualRecord
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
 * Renders an icon button for toggling transcription on or off.
 *
 * The button displays a stop icon in red when transcription is active, and a phone icon in blue when inactive. The icon's content description updates for accessibility based on the transcription state.
 *
 * @param isTranscribing True if transcription is currently active; false otherwise.
 * @param onClick Invoked when the button is pressed.
 */
@Composable
fun TranscribeButton(
    isTranscribing: Boolean,
    onClick: () -> Unit,
) {
    val icon = if (isTranscribing) Icons.Filled.Stop else Icons.Filled.Phone
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
 * Displays a design-time preview of the ConferenceRoomScreen composable within a MaterialTheme.
 */
@Composable
@Preview(showBackground = true)
fun ConferenceRoomScreenPreview() {
    MaterialTheme {
        ConferenceRoomScreen()
    }
}

