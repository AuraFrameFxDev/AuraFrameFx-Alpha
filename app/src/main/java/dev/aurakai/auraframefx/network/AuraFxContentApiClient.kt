package dev.aurakai.auraframefx.network

import dev.aurakai.auraframefx.generated.api.auraframefxai.ContentApi
import dev.aurakai.auraframefx.model.GenerateImageDescriptionRequest
import dev.aurakai.auraframefx.model.GenerateTextRequest
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Client wrapper for the AuraFrameFx AI Content API.
 * Provides clean methods to access text and image description generation capabilities.
 */
@Singleton
class AuraFxContentApiClient @Inject constructor(
    private val contentApi: ContentApi,
) {
    /**
     * Generates text content based on a provided prompt.
     *
     * @param prompt The text prompt for content generation
     * @param maxTokens Maximum number of tokens for the generated text (optional)
     * @param temperature Controls the randomness of the output (optional)
     * @return The API response containing generated text and finish reason
     */
    suspend fun generateText(
        prompt: String,
        maxTokens: Int? = null,
        temperature: Float? = null,
    ) = contentApi.generateText(
        GenerateTextRequest(
            prompt = prompt,
            maxTokens = maxTokens ?: 500,
            temperature = temperature ?: 0.7f
        )
    )

    /**
     * Generates a descriptive caption for an image based on provided URL and optional context.
     *
     * @param imageUrl URL of the image to describe
     * @param context Additional context for the image description (optional)
     * @return The API response containing the image description
     */
    suspend fun generateImageDescription(
        imageUrl: String,
        context: String? = null,
    ) = contentApi.generateImageDescription(
        GenerateImageDescriptionRequest(
            imageUrl = imageUrl,
            context = context
        )
    )
}
