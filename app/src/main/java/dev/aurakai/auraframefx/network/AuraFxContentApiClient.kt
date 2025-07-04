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
     * Generates AI text based on the provided prompt, with optional parameters to control output length and randomness.
     *
     * @param prompt The text prompt to generate a response for.
     * @param maxTokens Optional maximum number of tokens in the generated output. Defaults to 500 if not specified.
     * @param temperature Optional value to control the randomness of the output. Defaults to 0.7 if not specified.
     * @return The API response containing the generated text and the reason for completion.
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
     * Generates an AI-powered description for an image at the specified URL, optionally using additional context to influence the output.
     *
     * @param imageUrl The URL of the image to be described.
     * @param context Optional text providing extra context or guidance for the description.
     * @return The response containing the generated image description.
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
