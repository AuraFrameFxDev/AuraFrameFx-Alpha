package dev.aurakai.auraframefx.api

import dev.aurakai.auraframefx.ai.model.GenerateTextRequest
import dev.aurakai.auraframefx.ai.model.GenerateTextResponse

interface AiContentApi {
    /**
 * Generates AI-powered text content according to the parameters in the request.
 *
 * @param request Contains the input and configuration for text generation.
 * @return The generated text response.
 */
suspend fun generateText(request: GenerateTextRequest): GenerateTextResponse
}
