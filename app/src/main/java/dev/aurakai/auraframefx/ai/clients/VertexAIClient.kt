package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on the provided prompt.
 *
 * @param prompt The input text used to guide content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text based on the given prompt, with configurable maximum token length and output randomness.
 *
 * @param prompt The input prompt for text generation.
 * @param maxTokens The maximum number of tokens in the generated text.
 * @param temperature The degree of randomness in the output; higher values produce more diverse results.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code according to the provided specification, programming language, and coding style.
 *
 * @param specification Description of the desired functionality or requirements for the generated code.
 * @param language Target programming language for the generated code.
 * @param style Coding style or conventions to apply.
 * @return The generated source code, or null if code generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks if the Vertex AI service is reachable and responsive.
 *
 * @return `true` if the connection is successful, `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes and configures creative AI models within Vertex AI for content generation.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes the provided image data according to the given prompt and returns the analysis result.
 *
 * @param imageData The image data to analyze.
 * @param prompt The text prompt guiding the analysis.
 * @return The result of the image analysis.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
