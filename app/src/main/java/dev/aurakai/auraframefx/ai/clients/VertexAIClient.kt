package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
<<<<<<< HEAD
 * Generates content from the provided prompt.
 *
 * @param prompt The input text to guide content generation.
 * @return The generated content as a string, or null if generation fails.
=======
 * Generates content using the provided text prompt.
 *
 * @param prompt The text prompt that guides content generation.
 * @return The generated content as a string, or null if generation is unsuccessful.
>>>>>>> pr458merge
 */
    suspend fun generateContent(prompt: String): String?

    /**
<<<<<<< HEAD
 * Generates text based on the given prompt, with options to control output length and creativity.
 *
 * @param prompt The input prompt for text generation.
 * @param maxTokens The maximum number of tokens in the generated text.
 * @param temperature Controls randomness in the output; higher values produce more varied results.
=======
 * Generates text from a prompt with configurable maximum token length and output randomness.
 *
 * @param prompt The text prompt to guide text generation.
 * @param maxTokens The maximum number of tokens allowed in the generated text.
 * @param temperature The degree of randomness in the output; higher values yield more diverse results.
>>>>>>> pr458merge
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
<<<<<<< HEAD
 * Generates source code according to the given specification, programming language, and coding style.
 *
 * @param specification Description of the desired functionality or requirements for the generated code.
 * @param language The target programming language for the generated code.
 * @param style The coding style or conventions to apply.
 * @return The generated source code as a string, or null if code generation fails.
=======
 * Generates source code based on a specification, target programming language, and coding style.
 *
 * @param specification Description of the desired functionality or requirements for the code.
 * @param language The programming language for the generated code.
 * @param style The coding style or conventions to apply.
 * @return The generated source code as a string, or null if generation fails.
>>>>>>> pr458merge
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks if the Vertex AI service is reachable and responsive.
 *
<<<<<<< HEAD
 * @return `true` if the connection is successful, `false` otherwise.
=======
 * @return `true` if the service can be contacted successfully, `false` otherwise.
>>>>>>> pr458merge
 */
    suspend fun validateConnection(): Boolean

    /**
<<<<<<< HEAD
 * Initializes and configures creative AI models for content generation in Vertex AI.
=======
 * Initializes creative AI models in Vertex AI to enable content generation features.
>>>>>>> pr458merge
 */
    suspend fun initializeCreativeModels()

    /**
<<<<<<< HEAD
 * Analyzes the provided image data according to a text prompt and returns the analysis result.
 *
 * @param imageData The raw bytes of the image to analyze.
 * @param prompt The text prompt that guides the analysis.
=======
 * Analyzes the provided image data based on a guiding text prompt and returns the analysis result.
 *
 * @param imageData The raw bytes of the image to analyze.
 * @param prompt The text prompt that directs the analysis.
>>>>>>> pr458merge
 * @return The result of the image analysis as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
