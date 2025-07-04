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
     * Generates text using the AI Content API based on the provided prompt.
     *
     * @param prompt The text prompt to guide generation.
     * @param maxTokens Optional maximum number of tokens for the output; defaults to 500 if not specified.
     * @param temperature Optional value controlling randomness; defaults to 0.7 if not specified.
     * @return The API response containing the generated text and completion reason.
     */
    suspend fun generateText(
        prompt: String,
        maxTokens: Int? = null,
        temperature: Float? = null,
    ): AIContentApi.ResponseType = withContext(Dispatchers.IO) {

        aiContentApi.aiGenerateTextPost(
            GenerateTextRequest(
                prompt = prompt,
                maxTokens = maxTokens ?: 500,
                temperature = temperature ?: 0.7f
            )
        )
    }

    /**
     * Generates a description for an image using AI, with optional context to influence the output.
     *
     * @param imageUrl The URL of the image to describe.
     * @param context Additional context to guide the description, or null for default behavior.
     * @return The API response containing the generated image description.
     */
    suspend fun generateImageDescription(
        imageUrl: String,
        context: String? = null,
    ): GenerateImageDescriptionResponse = withContext(Dispatchers.IO) {

        aiContentApi.aiGenerateImageDescriptionPost(
            GenerateImageDescriptionRequest(
                imageUrl = imageUrl,
                context = context
            )
        )
    }
}
