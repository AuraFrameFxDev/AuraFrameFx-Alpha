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
 * @return The generated content as a String, or null if content generation is unsuccessful.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text based on the given prompt, with optional control over output length and randomness.
 *
 * @param prompt The input prompt to guide text generation.
 * @param maxTokens Optional maximum number of tokens in the generated text.
 * @param temperature Optional value to adjust the randomness of the generated text.
 * @return The generated text as a String.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates code based on a specification, target programming language, and coding style.
 *
 * @param specification Description of the desired code functionality or requirements.
 * @param language The programming language for the generated code.
 * @param style The coding style or conventions to follow.
 * @return The generated code as a String, or null if generation is unsuccessful.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies connectivity to Vertex AI.
 *
 * @return `true` if a connection can be established, `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Prepares creative models required for content generation with Vertex AI.
 *
 * This function must be called before generating creative content to ensure all necessary models are initialized.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data using a prompt to guide the interpretation.
 *
 * @param imageData The raw bytes of the image to analyze.
 * @param prompt The textual prompt that directs the analysis.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
