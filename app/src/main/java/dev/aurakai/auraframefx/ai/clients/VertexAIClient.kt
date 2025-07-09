package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates AI-driven content based on a text prompt.
 *
 * @param prompt The text prompt used to guide content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Produces AI-generated text from a prompt, allowing customization of output length and randomness.
 *
 * @param prompt The input text to guide the generation.
 * @param maxTokens The maximum number of tokens in the generated output.
 * @param temperature The degree of randomness in the generated text; higher values yield more diverse results.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code based on a specification, programming language, and coding style.
 *
 * @param specification Description of the desired code functionality.
 * @param language The programming language in which to generate the code.
 * @param style The coding style or conventions to apply.
 * @return The generated source code as a string, or null if code generation is unsuccessful.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks if the Vertex AI service is available and responsive.
 *
 * @return `true` if the service is accessible; `false` otherwise.
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
 * @param prompt The text prompt that directs the focus of the analysis.
 * @return The result of the image analysis as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
