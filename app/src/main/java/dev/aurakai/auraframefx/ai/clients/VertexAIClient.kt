package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on the given prompt.
 *
 * @param prompt The input prompt used to guide content generation.
 * @return The generated content as a string, or null if content generation is unsuccessful.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text based on the given prompt with optional control over output length and creativity.
 *
 * @param prompt The input prompt to guide text generation.
 * @param maxTokens Optional maximum number of tokens in the generated output.
 * @param temperature Optional value to adjust randomness and creativity in the generated text.
 * @return The generated text as a string.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates code based on the given specification, language, and style.
 *
 * @param specification Description of the desired code functionality or requirements.
 * @param language The programming language for the generated code.
 * @param style The coding style or conventions to follow.
 * @return The generated code as a string, or null if generation is unsuccessful.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies connectivity to Vertex AI.
 *
 * @return `true` if a connection can be established; `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Prepares creative models for content generation with Vertex AI.
 *
 * This function must be called before generating creative content to ensure the required models are initialized and ready for use.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes raw image data using a provided prompt and returns the analysis result.
 *
 * @param imageData The image data to analyze.
 * @param prompt The prompt that guides the image analysis.
 * @return The result of the image analysis as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
