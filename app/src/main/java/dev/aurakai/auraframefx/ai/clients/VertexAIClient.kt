package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content using AI based on the given text prompt.
 *
 * @param prompt The text prompt that guides the content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text using AI based on the provided prompt, with options to control output length and creativity.
 *
 * @param prompt The input text that guides the AI's response.
 * @param maxTokens Maximum number of tokens allowed in the generated output.
 * @param temperature Controls the randomness of the output; higher values produce more varied results.
 * @return The AI-generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code based on a specification, target language, and coding style.
 *
 * @param specification Description of the desired code functionality.
 * @param language Programming language for the generated code.
 * @param style Coding style or conventions to apply.
 * @return The generated source code as a string, or null if generation is unsuccessful.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks if the Vertex AI service is reachable and operational.
 *
 * @return `true` if the service is available; `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes creative AI models in Vertex AI for content generation.
 *
 * Should be called before invoking content generation methods to ensure required models are ready for use.
 */
    suspend fun initializeCreativeModels()

    /**
 * Performs AI-driven analysis of image data based on a provided text prompt.
 *
 * @param imageData The raw bytes representing the image to analyze.
 * @param prompt The textual instruction or context guiding the analysis.
 * @return The result of the AI analysis as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
