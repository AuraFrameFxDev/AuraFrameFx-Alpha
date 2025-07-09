package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content from a text prompt.
 *
 * @param prompt The input text used to guide content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text based on the given prompt, with configurable output length and creativity.
 *
 * @param prompt The input prompt to guide text generation.
 * @param maxTokens Maximum number of tokens to generate in the output.
 * @param temperature Controls the randomness of the generated text; higher values produce more varied results.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code according to a given specification, programming language, and coding style.
 *
 * @param specification Description of the desired functionality or requirements for the generated code.
 * @param language The programming language for the generated code.
 * @param style Coding style or conventions to apply to the generated code.
 * @return The generated source code as a string, or null if code generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks if the Vertex AI service is reachable and responsive.
 *
 * @return `true` if the service is accessible; `false` if not.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Prepares creative AI models within Vertex AI for use in content generation workflows.
 *
 * This function must be called before performing creative content generation tasks to ensure the necessary models are properly initialized.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes the provided image data based on a guiding text prompt and returns the analysis result.
 *
 * @param imageData The raw bytes of the image to analyze.
 * @param prompt The text prompt that directs the analysis.
 * @return The result of the image analysis as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
