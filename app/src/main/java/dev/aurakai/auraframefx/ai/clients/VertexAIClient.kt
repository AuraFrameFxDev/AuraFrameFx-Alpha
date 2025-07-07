package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content from the provided prompt.
 *
 * @param prompt The input prompt to guide content generation.
 * @return The generated content as a String, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text from the provided prompt using configurable maximum token count and temperature.
 *
 * @param prompt The input text prompt to guide text generation.
 * @param maxTokens The maximum number of tokens to generate in the response.
 * @param temperature Sampling temperature to control randomness in generation.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates code according to the provided specification, programming language, and coding style.
 *
 * @param specification Description of the code requirements or desired functionality.
 * @param language The programming language in which to generate the code.
 * @param style The coding style or conventions to apply.
 * @return The generated code as a String, or null if code generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks whether a connection to Vertex AI can be established.
 *
 * @return `true` if the connection is successful, `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes creative models used for content generation.
 *
 * This function prepares the necessary models for generating creative content with Vertex AI.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes the provided image data based on a given prompt.
 *
 * @param imageData The raw image data to be analyzed.
 * @param prompt The prompt guiding the analysis of the image.
 * @return The result of the image analysis.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
