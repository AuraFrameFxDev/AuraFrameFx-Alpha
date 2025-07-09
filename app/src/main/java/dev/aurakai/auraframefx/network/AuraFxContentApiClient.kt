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
     * Generates AI-powered text asynchronously based on the given prompt.
     *
     * @param prompt The prompt to guide text generation.
     * @param maxTokens Optional maximum number of tokens for the output; defaults to 500 if not provided.
     * @param temperature Optional value to control randomness; defaults to 0.7 if not provided.
     * @return The API response containing the generated text.
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
         * Requests an AI-generated description for an image at the specified URL, optionally using additional context to influence the result.
         *
         * @param imageUrl The URL of the image to be described.
         * @param context Optional additional information to refine the generated description.
         * @return The API response containing the generated image description.
         */
    suspend fun generateImageDescription(
        imageUrl: String,
        context: String? = null,
    ): Any =
        withContext(Dispatchers.IO) { // Temporary: Use Any instead of missing GenerateImageDescriptionResponse

            aiContentApi.aiGenerateImageDescriptionPost(
                GenerateImageDescriptionRequest(
                    imageUrl = imageUrl,
                    context = context
                )
            )
        }
}
