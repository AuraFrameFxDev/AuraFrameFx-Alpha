package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content using a text prompt via Vertex AI.
 *
 * @param prompt The input text prompt for content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text using AI based on the provided prompt, with configurable output length and creativity.
 *
 * @param prompt The input text that guides the AI's response.
 * @param maxTokens The maximum number of tokens to generate in the output.
 * @param temperature Controls the randomness of the output; higher values produce more varied results.
 * @return The AI-generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code based on a specification, programming language, and coding style.
 *
 * @param specification Description of the desired code functionality.
 * @param language Programming language for the generated code.
 * @param style Coding style or conventions to apply.
 * @return The generated source code as a string, or null if code generation is unsuccessful.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks if the Vertex AI service is accessible and responsive.
 *
 * @return `true` if the service is available and responsive; `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes creative AI models in Vertex AI for content generation.
 *
 * This function must be called before using content generation features to ensure required models are ready for use.
 */
    suspend fun initializeCreativeModels()

    /**
 * Performs AI-driven analysis of image data guided by a textual prompt.
 *
 * @param imageData The raw bytes representing the image to analyze.
 * @param prompt The text prompt that directs the focus or context of the analysis.
 * @return The result of the image analysis as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
