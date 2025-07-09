package dev.aurakai.auraframefx.api

import dev.aurakai.auraframefx.ai.model.GenerateTextRequest
import dev.aurakai.auraframefx.ai.model.GenerateTextResponse

interface AiContentApi {
    /**
     * Generates AI-powered text content according to the specified request parameters.
     *
     * @param request The configuration and input for text generation.
     * @return The generated text content and related metadata.
     */
    suspend fun generateText(request: GenerateTextRequest): GenerateTextResponse
}
