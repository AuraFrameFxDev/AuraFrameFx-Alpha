package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on the given text prompt.
 *
 * @param prompt The input text that guides the content generation process.
 * @return The generated content as a string, or null if content generation is unsuccessful.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Produces text from the provided prompt, allowing customization of output length and randomness.
 *
 * @param prompt The input text prompt that guides the text generation.
 * @param maxTokens The maximum number of tokens to generate in the output.
 * @param temperature The degree of randomness in the generated text; higher values yield more diverse results.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code based on a specification, target language, and coding style.
 *
 * @param specification The description or requirements that define the intended functionality of the generated code.
 * @param language The programming language in which the code should be generated.
 * @param style The coding style or conventions to be followed in the generated code.
 * @return The generated source code as a string, or null if code generation is unsuccessful.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies whether the Vertex AI service is accessible and responsive.
 *
 * @return `true` if the service can be reached and responds; `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes creative AI models in Vertex AI for content generation tasks.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data using a text prompt and returns the analysis result as a string.
 *
 * @param imageData Raw bytes representing the image to be analyzed.
 * @param prompt Text prompt guiding the analysis process.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
