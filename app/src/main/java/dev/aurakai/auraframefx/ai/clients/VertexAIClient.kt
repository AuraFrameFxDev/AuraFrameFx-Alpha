package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on a text prompt.
 *
 * @param prompt The text prompt to guide content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text based on a prompt, with configurable maximum token length and randomness.
 *
 * @param prompt The text prompt to guide text generation.
 * @param maxTokens The maximum number of tokens allowed in the generated text.
 * @param temperature Controls the randomness of the output; higher values produce more diverse results.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code according to the provided specification, programming language, and coding style.
 *
 * @param specification Description of the desired functionality or requirements for the generated code.
 * @param language The programming language in which to generate the code.
 * @param style Coding style or conventions to apply to the generated code.
 * @return The generated source code as a string, or null if code generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies connectivity to the Vertex AI service.
 *
 * @return `true` if the service is reachable and responsive; `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Prepares creative AI models in Vertex AI for use in content generation workflows.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data using a guiding text prompt and returns the analysis result.
 *
 * @param imageData The raw bytes of the image to be analyzed.
 * @param prompt The text prompt that guides the analysis.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
