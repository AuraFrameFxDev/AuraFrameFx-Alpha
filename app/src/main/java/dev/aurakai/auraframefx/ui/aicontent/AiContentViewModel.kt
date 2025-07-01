package dev.aurakai.auraframefx.ui.aicontent

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import dev.aurakai.auraframefx.network.AuraFxContentApiClient
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import timber.log.Timber
import javax.inject.Inject

/**
 * ViewModel for the AI Content generation features
 */
@HiltViewModel
class AiContentViewModel @Inject constructor(
    private val contentApiClient: AuraFxContentApiClient,
) : ViewModel() {

    // UI states
    private val _textGenerationState =
        MutableStateFlow<TextGenerationState>(TextGenerationState.Idle)
    val textGenerationState: StateFlow<TextGenerationState> = _textGenerationState

    private val _imageDescriptionState =
        MutableStateFlow<ImageDescriptionState>(ImageDescriptionState.Idle)
    val imageDescriptionState: StateFlow<ImageDescriptionState> = _imageDescriptionState

    /**
     * Generates text content based on the provided prompt
     */
    fun generateText(prompt: String, maxTokens: Int? = 500, temperature: Float? = 0.7f) {
        _textGenerationState.value = TextGenerationState.Loading

        viewModelScope.launch {
            try {
                val response = contentApiClient.generateText(prompt, maxTokens, temperature)
                _textGenerationState.value = TextGenerationState.Success(
                    generatedText = response.generatedText ?: "",
                    finishReason = response.finishReason ?: ""
                )
            } catch (e: Exception) {
                Timber.e(e, "Error generating text")
                _textGenerationState.value = TextGenerationState.Error(e.message ?: "Unknown error")
            }
        }
    }

    /**
     * Generates a description for an image at the provided URL
     */
    fun generateImageDescription(imageUrl: String, context: String? = null) {
        _imageDescriptionState.value = ImageDescriptionState.Loading

        viewModelScope.launch {
            try {
                val response = contentApiClient.generateImageDescription(imageUrl, context)
                _imageDescriptionState.value = ImageDescriptionState.Success(
                    description = response.description ?: ""
                )
            } catch (e: Exception) {
                Timber.e(e, "Error generating image description")
                _imageDescriptionState.value =
                    ImageDescriptionState.Error(e.message ?: "Unknown error")
            }
        }
    }

    /**
     * Reset the text generation state to idle
     */
    fun resetTextGenerationState() {
        _textGenerationState.value = TextGenerationState.Idle
    }

    /**
     * Reset the image description state to idle
     */
    fun resetImageDescriptionState() {
        _imageDescriptionState.value = ImageDescriptionState.Idle
    }
}

/**
 * State class for text generation
 */
sealed class TextGenerationState {
    object Idle : TextGenerationState()
    object Loading : TextGenerationState()
    data class Success(val generatedText: String, val finishReason: String) : TextGenerationState()
    data class Error(val errorMessage: String) : TextGenerationState()
}

/**
 * State class for image description generation
 */
sealed class ImageDescriptionState {
    object Idle : ImageDescriptionState()
    object Loading : ImageDescriptionState()
    data class Success(val description: String) : ImageDescriptionState()
    data class Error(val errorMessage: String) : ImageDescriptionState()
}
