package dev.aurakai.auraframefx.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.tooling.preview.Preview
import dev.aurakai.auraframefx.ui.theme.*

/**
 * Data class representing a chat message
 */
data class ChatMessage(
    val content: String,
    val isFromUser: Boolean,
    val timestamp: Long = System.currentTimeMillis(),
)

/**
 * AI Chat screen using Compose best practices
 */
@OptIn(ExperimentalMaterial3Api::class)
/**
 * Displays an AI chat interface with message history and input field.
 *
 * Presents a scrollable list of chat messages and allows the user to send new messages. User messages and simulated AI responses are displayed with distinct styling and alignment. The input field persists its content across configuration changes, and messages are retained during recomposition.
 */
@Composable
fun AiChatScreen() {
    // State handling with rememberSaveable to persist through configuration changes
    var messageText by rememberSaveable { mutableStateOf("") }
    var chatMessages by rememberSaveable {
        mutableStateOf(
            listOf(
                ChatMessage("Hello! How can I help you today?", false),
                ChatMessage("Tell me about AI image generation", true),
                ChatMessage(
                    "AI image generation uses techniques like diffusion models to create images from text descriptions. Popular systems include DALL-E, Midjourney, and Stable Diffusion.",
                    false
                )
            )
        )
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(AppDimensions.spacing_medium)
    ) {
        // Messages area with proper list handling
        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(bottom = AppDimensions.spacing_medium),
            verticalArrangement = Arrangement.spacedBy(AppDimensions.spacing_small)
        ) {
            items(chatMessages) { message ->
                ChatMessageItem(message)
            }
        }

        // Input area with modern Material3 components
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = AppDimensions.spacing_small),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(AppDimensions.spacing_small)
        ) {
            OutlinedTextField(
                value = messageText,
                onValueChange = { messageText = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text(AppStrings.AI_CHAT_PLACEHOLDER) },
                shape = InputFieldShape,
                colors = TextFieldDefaults.outlinedTextFieldColors(
                    focusedBorderColor = MaterialTheme.colorScheme.primary,
                    unfocusedBorderColor = MaterialTheme.colorScheme.outline
                )
            )

            FloatingActionButton(
                onClick = {
                    if (messageText.isNotBlank()) {
                        // Add user message
                        val userMessage = ChatMessage(messageText.trim(), true)
                        chatMessages = chatMessages + userMessage

                        // Simulate AI response
                        val aiResponse = ChatMessage(
                            "I received your message: '${messageText.trim()}'. This is a simulated response.",
                            false
                        )
                        chatMessages = chatMessages + aiResponse

                        // Clear input
                        messageText = ""
                    }
                },
                shape = FloatingActionButtonShape,
                containerColor = MaterialTheme.colorScheme.primary
            ) {
                Icon(
                    imageVector = Icons.Default.Send,
                    contentDescription = AppStrings.AI_CHAT_SEND,
                    tint = MaterialTheme.colorScheme.onPrimary
                )
            }
        }
    }
}

/**
 * Displays a single chat message bubble with styling based on sender.
 *
 * Aligns, colors, and shapes the message bubble differently depending on whether the message is from the user or the AI.
 *
 * @param message The chat message to display.
 */
@Composable
fun ChatMessageItem(message: ChatMessage) {
    val alignment = if (message.isFromUser) Alignment.End else Alignment.Start
    val background = if (message.isFromUser)
        MaterialTheme.colorScheme.primaryContainer
    else
        MaterialTheme.colorScheme.surfaceVariant

    val textColor = if (message.isFromUser)
        MaterialTheme.colorScheme.onPrimaryContainer
    else
        MaterialTheme.colorScheme.onSurfaceVariant

    val bubbleShape = if (message.isFromUser)
        ChatBubbleOutgoingShape
    else
        ChatBubbleIncomingShape

    Box(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .align(alignment)
                .widthIn(max = 280.dp) // Maximum width for message bubbles
                .clip(bubbleShape)
                .background(background)
                .padding(AppDimensions.spacing_medium),
        ) {
            Text(
                text = message.content,
                color = textColor,
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}

/**
 * Displays a design-time preview of the AI chat screen using the custom theme.
 */
@Preview(showBackground = true)
@Composable
fun AiChatScreenPreview() {
    AuraFrameFXTheme { // Using our custom theme for preview
        AiChatScreen()
    }
}
