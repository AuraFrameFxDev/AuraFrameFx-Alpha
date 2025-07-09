package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on the given prompt.
 *
 * @param prompt The input text used to guide content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
     * Generates text based on the provided prompt, with configurable output length and creativity.
     *
     * @param prompt The input text used to guide the text generation.
     * @param maxTokens The maximum number of tokens to generate in the output.
     * @param temperature Controls the randomness of the generated text; higher values produce more varied results.
     * @return The generated text as a string.
     */
    suspend fun generateText(
        prompt: String,
        maxTokens: Int = 1000,
        temperature: Float = 0.7f
    ): String

    /**
 * Generates source code according to the provided specification, programming language, and coding style.
 *
 * @param specification The requirements or description guiding the code generation.
 * @param language The programming language in which the code should be generated.
 * @param style The coding style or conventions to apply to the generated code.
 * @return The generated source code as a string, or null if code generation is unsuccessful.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies if the Vertex AI service is accessible and responsive.
 *
 * @return `true` if the service can be reached and responds; `false` if it is unreachable or unresponsive.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Prepares and configures creative AI models in Vertex AI to enable content generation features.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes the provided image data using a guiding text prompt and returns the analysis result.
 *
 * @param imageData The raw byte array representing the image to analyze.
 * @param prompt The text prompt that guides the image analysis.
 * @return The result of the image analysis as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
