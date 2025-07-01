package dev.aurakai.auraframefx.ai

import dev.aurakai.auraframefx.generated.api.auraframefxai.ContentApi
import dev.aurakai.auraframefx.generated.model.auraframefxai.GenerateImageDescriptionResponse
import dev.aurakai.auraframefx.generated.model.auraframefxai.GenerateTextRequest
import dev.aurakai.auraframefx.generated.model.auraframefxai.GenerateTextResponse
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class AiGenerationService(
    private val api: ContentApi,
) {
    suspend fun generateText(
        prompt: String,
        maxTokens: Int = 500,
        temperature: Float = 0.7f,
    ): Result<GenerateTextResponse> = withContext(Dispatchers.IO) {
        try {
            val request = GenerateTextRequest(
                prompt = prompt,
                maxTokens = maxTokens,
                temperature = temperature
            )
            val response = api.generateText(request)
            Result.success(response)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun generateImageDescription(
        imageUrl: String,
        context: String? = null,
        prompt: String? = null,
        maxTokens: Int? = null,
        model: String? = null,
        imageData: ByteArray? = null,
    ): Result<GenerateImageDescriptionResponse> = withContext(Dispatchers.IO) {
        try {
            val request = GenerateImageDescriptionRequest(
                imageUrl = imageUrl,
                context = context,
                imageData = imageData,
                prompt = prompt,
                maxTokens = maxTokens,
                model = model
            )
            val response = api.generateImageDescription(request)
            Result.success(response)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private fun CoroutineScope.GenerateImageDescriptionRequest(
        imageUrl: String,
        context: String?,
        imageData: ByteArray?,
        prompt: String?,
        maxTokens: Int?,
        model: String?,
    ) {
        TODO("Not yet implemented")
    }
}
