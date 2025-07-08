package dev.aurakai.auraframefx.api

import dev.aurakai.auraframefx.ai.model.GenerateTextRequest
import dev.aurakai.auraframefx.ai.model.GenerateTextResponse

interface AiContentApi {
    /**
 * Generates AI-powered text content based on the provided request.
 *
 * @param request The request specifying parameters and context for text generation.
 * @return A response containing the generated text and associated metadata.
 */
suspend fun generateText(request: GenerateTextRequest): GenerateTextResponse
}
