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
     * Generates AI-based text from the provided prompt.
     *
     * @param prompt The text prompt to guide the AI-generated output.
     * @param maxTokens Optional maximum number of tokens for the generated text. Defaults to 500.
     * @param temperature Optional value controlling randomness in generation. Defaults to 0.7.
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
     * Generates an AI-generated description for the given image URL, optionally influenced by additional context.
     *
     * @param imageUrl The URL of the image to describe.
     * @param context Optional context to guide the description; if not provided, a general description is generated.
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
