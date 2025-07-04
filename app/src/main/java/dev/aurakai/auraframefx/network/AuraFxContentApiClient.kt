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
     * Generates AI-based text from the specified prompt using the AI Content API.
     *
     * @param prompt The input text prompt for content generation.
     * @param maxTokens Maximum number of tokens for the generated text. Defaults to 500 if not provided.
     * @param temperature Controls the randomness of the generated text. Defaults to 0.7 if not provided.
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
     * Generates an AI-based description for the specified image URL, using optional context to influence the output.
     *
     * @param imageUrl The URL of the image to describe.
     * @param context Additional context to guide the description, if provided.
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
