package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content from a text prompt.
 *
 * @param prompt The text prompt to guide content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text from the provided prompt, allowing customization of output length and randomness.
 *
 * @param prompt The input prompt that guides text generation.
 * @param maxTokens The maximum number of tokens to include in the generated text.
 * @param temperature Controls the randomness of the output; higher values produce more varied results.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code based on a specification, target language, and coding style.
 *
 * @param specification Description of the functionality or requirements for the code to be generated.
 * @param language The programming language in which the code should be generated.
 * @param style The coding style or conventions to follow for the generated code.
 * @return The generated source code as a string, or null if generation is unsuccessful.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies connectivity to the Vertex AI service.
 *
 * @return `true` if the service is reachable and responsive; `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes creative AI models in Vertex AI to enable content generation capabilities.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes the provided image data using a guiding text prompt and returns the analysis result.
 *
 * @param imageData The raw bytes of the image to be analyzed.
 * @param prompt The text prompt that directs the analysis.
 * @return The result of the image analysis as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
