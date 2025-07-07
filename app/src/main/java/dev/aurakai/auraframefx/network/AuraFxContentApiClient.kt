package dev.aurakai.auraframefx.network

import dev.aurakai.auraframefx.api.client.apis.AIContentApi
import dev.aurakai.auraframefx.api.client.models.GenerateImageDescriptionRequest
import dev.aurakai.auraframefx.api.client.models.GenerateTextRequest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Client wrapper for the AuraFrameFx AI Content API.
 * Provides clean methods to access text and image description generation capabilities.
 */
@Singleton
class AuraFxContentApiClient @Inject constructor(
    private val aiContentApi: AIContentApi,
) {
    /**
     * Asynchronously generates AI-powered text based on the provided prompt.
     *
     * @param prompt The prompt to guide text generation.
     * @param maxTokens Optional maximum number of tokens for the generated text; defaults to 500.
     * @param temperature Optional value controlling randomness of the output; defaults to 0.7.
     * @return The raw API response containing the generated text.
     */
    suspend fun generateText(
        prompt: String,
        maxTokens: Int? = null,
        temperature: Float? = null,
    ): Any = withContext(Dispatchers.IO) { // Temporary: Use Any instead of missing ResponseType

        aiContentApi.aiGenerateTextPost(
            GenerateTextRequest(
                prompt = prompt,
                maxTokens = maxTokens ?: 500,
                temperature = temperature ?: 0.7f
            )
        )
    }

    /**
     * Generates an AI-powered description for an image at the specified URL, optionally using additional context.
     *
     * @param imageUrl The URL of the image to describe.
     * @param context Optional text to provide context or influence the generated description.
     * @return The raw API response containing the generated image description.
     */
    suspend fun generateImageDescription(
        imageUrl: String,
        context: String? = null,
    ): Any = withContext(Dispatchers.IO) { // Temporary: Use Any instead of missing GenerateImageDescriptionResponse

        aiContentApi.aiGenerateImageDescriptionPost(
            GenerateImageDescriptionRequest(
                imageUrl = imageUrl,
                context = context
            )
        )
    }
}
