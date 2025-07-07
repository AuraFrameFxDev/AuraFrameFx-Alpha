package dev.aurakai.auraframefx.api

import dev.aurakai.auraframefx.ai.model.GenerateTextRequest
import dev.aurakai.auraframefx.ai.model.GenerateTextResponse

interface AiContentApi {
    /**
 * Generates AI-powered text content according to the specifications in the request.
 *
 * @param request Contains the parameters and context for text generation.
 * @return The response containing the generated text.
 */
suspend fun generateText(request: GenerateTextRequest): GenerateTextResponse
}
