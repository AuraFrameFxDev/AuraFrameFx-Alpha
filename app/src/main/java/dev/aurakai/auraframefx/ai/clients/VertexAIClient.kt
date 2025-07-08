package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content using the provided prompt.
 *
 * @param prompt The text prompt that guides content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
     * Generates text using the provided prompt, with options to control output length and creativity.
     *
     * @param prompt The input text that guides the text generation process.
     * @param maxTokens The maximum number of tokens to include in the generated output.
     * @param temperature Controls the randomness and creativity of the output; higher values produce more varied results.
     * @return The generated text.
     */
    suspend fun generateText(
        prompt: String,
        maxTokens: Int = 1000,
        temperature: Float = 0.7f
    ): String

    /**
 * Generates source code according to the given specification, programming language, and coding style.
 *
 * @param specification Describes the desired functionality or requirements for the generated code.
 * @param language The target programming language for the generated code.
 * @param style The coding style or conventions to apply.
 * @return The generated source code, or null if code generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks if the Vertex AI service is accessible and responsive.
 *
 * @return `true` if the service can be reached and responds; `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes and configures creative AI models in Vertex AI to enable content generation capabilities.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes raw image data according to a guiding text prompt and returns the analysis result.
 *
 * @param imageData The raw image data to analyze.
 * @param prompt The text prompt that directs the analysis.
 * @return The result of the image analysis as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
