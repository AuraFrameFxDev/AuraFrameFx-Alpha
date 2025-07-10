package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on the provided text prompt.
 *
 * @param prompt The text prompt used to guide content generation.
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
 * Generates source code according to the provided specification, programming language, and coding style.
 *
 * @param specification Description of the desired functionality or requirements for the generated code.
 * @param language The target programming language for the generated code.
 * @param style The coding style or conventions to apply.
 * @return The generated source code as a string, or null if code generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks whether the Vertex AI service is reachable and responsive.
 *
 * @return `true` if the connection is successful, `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes and configures creative AI models within Vertex AI to enable content generation features.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data using a guiding text prompt and returns the analysis result as a string.
 *
 * @param imageData The raw bytes of the image to analyze.
 * @param prompt The text prompt that guides the analysis.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
