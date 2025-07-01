package dev.aurakai.auraframefx.ui.aicontent

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle

/**
 * Composable screen that demonstrates the AI content generation capabilities
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AiContentScreen(
    viewModel: AiContentViewModel = hiltViewModel(),
) {
    val textGenerationState by viewModel.textGenerationState.collectAsStateWithLifecycle()
    val imageDescriptionState by viewModel.imageDescriptionState.collectAsStateWithLifecycle()

    val textPrompt = remember { mutableStateOf("") }
    val imageUrl = remember { mutableStateOf("") }
    val imageContext = remember { mutableStateOf("") }

    val focusManager = LocalFocusManager.current
    val scrollState = rememberScrollState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("AuraFrameFx AI Content") }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp)
                .verticalScroll(scrollState),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Text Generation Section
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Text(
                        text = "Text Generation",
                        style = MaterialTheme.typography.headlineSmall
                    )

                    OutlinedTextField(
                        value = textPrompt.value,
                        onValueChange = { textPrompt.value = it },
                        label = { Text("Enter prompt") },
                        modifier = Modifier.fillMaxWidth(),
                        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                        keyboardActions = KeyboardActions(
                            onDone = { focusManager.clearFocus() }
                        )
                    )

                    Button(
                        onClick = {
                            viewModel.generateText(textPrompt.value)
                            focusManager.clearFocus()
                        },
                        modifier = Modifier.align(Alignment.End),
                        enabled = textPrompt.value.isNotBlank()
                    ) {
                        Text("Generate Text")
                    }

                    // Display the generated text
                    when (val state = textGenerationState) {
                        is TextGenerationState.Loading -> {
                            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                            Text("Generating text...")
                        }

                        is TextGenerationState.Success -> {
                            Text(
                                text = "Generated Text:",
                                style = MaterialTheme.typography.bodyLarge
                            )
                            Surface(
                                modifier = Modifier.fillMaxWidth(),
                                color = MaterialTheme.colorScheme.surfaceVariant,
                                shape = MaterialTheme.shapes.medium
                            ) {
                                Text(
                                    text = state.generatedText,
                                    modifier = Modifier.padding(16.dp),
                                    style = MaterialTheme.typography.bodyMedium
                                )
                            }
                            Text(
                                text = "Finish reason: ${state.finishReason}",
                                style = MaterialTheme.typography.bodySmall
                            )
                        }

                        is TextGenerationState.Error -> {
                            Text(
                                text = "Error: ${state.errorMessage}",
                                color = MaterialTheme.colorScheme.error
                            )
                        }

                        else -> { /* Idle state, nothing to show */
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Image Description Section
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Text(
                        text = "Image Description Generation",
                        style = MaterialTheme.typography.headlineSmall
                    )

                    OutlinedTextField(
                        value = imageUrl.value,
                        onValueChange = { imageUrl.value = it },
                        label = { Text("Enter image URL") },
                        modifier = Modifier.fillMaxWidth(),
                        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Next),
                        keyboardActions = KeyboardActions(
                            onNext = { focusManager.clearFocus() }
                        )
                    )

                    OutlinedTextField(
                        value = imageContext.value,
                        onValueChange = { imageContext.value = it },
                        label = { Text("Context (optional)") },
                        modifier = Modifier.fillMaxWidth(),
                        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                        keyboardActions = KeyboardActions(
                            onDone = { focusManager.clearFocus() }
                        )
                    )

                    Button(
                        onClick = {
                            viewModel.generateImageDescription(
                                imageUrl.value,
                                if (imageContext.value.isBlank()) null else imageContext.value
                            )
                            focusManager.clearFocus()
                        },
                        modifier = Modifier.align(Alignment.End),
                        enabled = imageUrl.value.isNotBlank()
                    ) {
                        Text("Generate Description")
                    }

                    // Display the generated image description
                    when (val state = imageDescriptionState) {
                        is ImageDescriptionState.Loading -> {
                            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                            Text("Generating description...")
                        }

                        is ImageDescriptionState.Success -> {
                            Text(
                                text = "Generated Description:",
                                style = MaterialTheme.typography.bodyLarge
                            )
                            Surface(
                                modifier = Modifier.fillMaxWidth(),
                                color = MaterialTheme.colorScheme.surfaceVariant,
                                shape = MaterialTheme.shapes.medium
                            ) {
                                Text(
                                    text = state.description,
                                    modifier = Modifier.padding(16.dp),
                                    style = MaterialTheme.typography.bodyMedium
                                )
                            }
                        }

                        is ImageDescriptionState.Error -> {
                            Text(
                                text = "Error: ${state.errorMessage}",
                                color = MaterialTheme.colorScheme.error
                            )
                        }

                        else -> { /* Idle state, nothing to show */
                        }
                    }
                }
            }
        }
    }
}
