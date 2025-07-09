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
     * @param prompt The text prompt to guide the AI-generated output.
     * @param maxTokens The maximum number of tokens for the generated text. Defaults to 500 if not specified.
     * @param temperature Controls the randomness of the generated text. Defaults to 0.7 if not specified.
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
         * Asynchronously generates a description for an image at the given URL, optionally using additional context to influence the output.
         *
         * @param imageUrl The URL of the image to describe.
         * @param context Optional context to guide the description generation.
         * @return The generated image description from the API.
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
