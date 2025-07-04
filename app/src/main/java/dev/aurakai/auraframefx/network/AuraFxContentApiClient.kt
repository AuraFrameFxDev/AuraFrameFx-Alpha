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
     * Generates AI text based on the given prompt and optional parameters.
     *
     * @param prompt The input text prompt for the AI model.
     * @param maxTokens The maximum number of tokens to generate; defaults to 500 if not specified.
     * @param temperature Controls the randomness of the output; defaults to 0.7 if not specified.
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
     * Generates an AI-powered description for the specified image URL, optionally using additional context to refine the output.
     *
     * @param imageUrl The URL of the image to be described.
     * @param context An optional string providing context to influence the generated description.
     * @return The response containing the AI-generated image description.
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
