package dev.aurakai.auraframefx.api

import dev.aurakai.auraframefx.ai.model.GenerateTextRequest
import dev.aurakai.auraframefx.ai.model.GenerateTextResponse

interface AiContentApi {
    /**
 * Generates AI-powered text content according to the specified request.
 *
 * @param request The parameters and context for text generation.
 * @return The generated text response.
 */
suspend fun generateText(request: GenerateTextRequest): GenerateTextResponse
}
