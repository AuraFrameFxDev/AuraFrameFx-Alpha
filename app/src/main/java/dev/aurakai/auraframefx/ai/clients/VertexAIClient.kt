package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on the provided text prompt.
 *
 * @param prompt The text prompt used to generate content.
 * @return The generated content as a string, or null if content generation is unsuccessful.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text based on a prompt with configurable maximum token count and randomness.
 *
 * @param prompt The input text prompt for generation.
 * @param maxTokens The upper limit on the number of tokens in the generated output.
 * @param temperature Degree of randomness in the generated text; higher values yield more diverse results.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code according to a specification, programming language, and coding style.
 *
 * @param specification Description of the desired functionality or requirements for the generated code.
 * @param language The programming language for the generated code.
 * @param style The coding style or conventions to apply.
 * @return The generated source code as a string, or null if code generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks if the Vertex AI service is reachable and responsive.
 *
 * @return `true` if the service can be contacted successfully, `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes creative AI models in Vertex AI to enable content generation features.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data based on a guiding text prompt and returns the analysis result.
 *
 * @param imageData The raw bytes of the image to analyze.
 * @param prompt The text prompt that guides the analysis.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
