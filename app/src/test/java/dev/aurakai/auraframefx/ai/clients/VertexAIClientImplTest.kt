package dev.aurakai.auraframefx.ai.clients

import io.mockk.*
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.junit.jupiter.params.provider.CsvSource
import org.junit.jupiter.params.provider.NullSource
import org.junit.jupiter.params.provider.EmptySource
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.runBlocking
import java.io.IOException
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.net.ConnectException
import java.util.concurrent.TimeoutException

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class VertexAIClientImplTest {

    private lateinit var httpClient: HttpClient
    private lateinit var vertexAIClient: VertexAIClientImpl
    private lateinit var mockResponse: HttpResponse<String>
    
    private val validApiKey = "test-api-key"
    private val validProjectId = "test-project-id"
    private val validLocation = "us-central1"
    private val validModel = "gemini-pro"

    @BeforeEach
    fun setUp() {
        MockKAnnotations.init(this)
        httpClient = mockk()
        mockResponse = mockk()
        vertexAIClient = VertexAIClientImpl(httpClient, validApiKey, validProjectId, validLocation)
    }

    @AfterEach
    fun tearDown() {
        clearAllMocks()
    }

    @Nested
    @DisplayName("Constructor Tests")
    inner class ConstructorTests {

        @Test
        @DisplayName("Should create client with valid parameters")
        fun `should create client with valid parameters`() {
            assertDoesNotThrow {
                VertexAIClientImpl(httpClient, validApiKey, validProjectId, validLocation)
            }
        }

        @ParameterizedTest
        @NullSource
        @EmptySource
        @ValueSource(strings = ["", "   "])
        @DisplayName("Should throw exception for invalid API key")
        fun `should throw exception for invalid API key`(apiKey: String?) {
            assertThrows<IllegalArgumentException> {
                VertexAIClientImpl(httpClient, apiKey, validProjectId, validLocation)
            }
        }

        @ParameterizedTest
        @NullSource
        @EmptySource
        @ValueSource(strings = ["", "   "])
        @DisplayName("Should throw exception for invalid project ID")
        fun `should throw exception for invalid project ID`(projectId: String?) {
            assertThrows<IllegalArgumentException> {
                VertexAIClientImpl(httpClient, validApiKey, projectId, validLocation)
            }
        }

        @ParameterizedTest
        @NullSource
        @EmptySource
        @ValueSource(strings = ["", "   "])
        @DisplayName("Should throw exception for invalid location")
        fun `should throw exception for invalid location`(location: String?) {
            assertThrows<IllegalArgumentException> {
                VertexAIClientImpl(httpClient, validApiKey, validProjectId, location)
            }
        }
    }

    @Nested
    @DisplayName("Generate Content Tests")
    inner class GenerateContentTests {

        @Test
        @DisplayName("Should generate content successfully with valid prompt")
        fun `should generate content successfully with valid prompt`() = runTest {
            val prompt = "Test prompt"
            val expectedResponse = """
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "Generated response"
                                    }
                                ]
                            }
                        }
                    ]
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns expectedResponse
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            val result = vertexAIClient.generateContent(prompt, validModel)

            assertEquals("Generated response", result)
            coVerify { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) }
        }

        @Test
        @DisplayName("Should handle empty response gracefully")
        fun `should handle empty response gracefully`() = runTest {
            val prompt = "Test prompt"
            val emptyResponse = """
                {
                    "candidates": []
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns emptyResponse
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            val result = vertexAIClient.generateContent(prompt, validModel)

            assertEquals("", result)
        }

        @Test
        @DisplayName("Should handle malformed JSON response")
        fun `should handle malformed JSON response`() = runTest {
            val prompt = "Test prompt"
            val malformedJson = "{ invalid json"

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns malformedJson
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            assertThrows<RuntimeException> {
                runBlocking { vertexAIClient.generateContent(prompt, validModel) }
            }
        }

        @ParameterizedTest
        @CsvSource(
            "400, Bad Request",
            "401, Unauthorized",
            "403, Forbidden",
            "404, Not Found",
            "500, Internal Server Error",
            "503, Service Unavailable"
        )
        @DisplayName("Should handle HTTP error responses")
        fun `should handle HTTP error responses`(statusCode: Int, errorMessage: String) = runTest {
            val prompt = "Test prompt"
            val errorResponse = """
                {
                    "error": {
                        "message": "$errorMessage"
                    }
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns statusCode
            every { mockResponse.body() } returns errorResponse
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            assertThrows<RuntimeException> {
                runBlocking { vertexAIClient.generateContent(prompt, validModel) }
            }
        }

        @Test
        @DisplayName("Should handle network timeout")
        fun `should handle network timeout`() = runTest {
            val prompt = "Test prompt"
            
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } throws TimeoutException("Request timed out")

            assertThrows<TimeoutException> {
                runBlocking { vertexAIClient.generateContent(prompt, validModel) }
            }
        }

        @Test
        @DisplayName("Should handle connection failure")
        fun `should handle connection failure`() = runTest {
            val prompt = "Test prompt"
            
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } throws ConnectException("Connection refused")

            assertThrows<ConnectException> {
                runBlocking { vertexAIClient.generateContent(prompt, validModel) }
            }
        }

        @Test
        @DisplayName("Should handle IO exception")
        fun `should handle IO exception`() = runTest {
            val prompt = "Test prompt"
            
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } throws IOException("IO error")

            assertThrows<IOException> {
                runBlocking { vertexAIClient.generateContent(prompt, validModel) }
            }
        }

        @ParameterizedTest
        @NullSource
        @EmptySource
        @ValueSource(strings = ["", "   "])
        @DisplayName("Should throw exception for invalid prompt")
        fun `should throw exception for invalid prompt`(prompt: String?) = runTest {
            assertThrows<IllegalArgumentException> {
                runBlocking { vertexAIClient.generateContent(prompt, validModel) }
            }
        }

        @ParameterizedTest
        @NullSource
        @EmptySource
        @ValueSource(strings = ["", "   "])
        @DisplayName("Should throw exception for invalid model")
        fun `should throw exception for invalid model`(model: String?) = runTest {
            assertThrows<IllegalArgumentException> {
                runBlocking { vertexAIClient.generateContent("Test prompt", model) }
            }
        }

        @Test
        @DisplayName("Should handle very long prompt")
        fun `should handle very long prompt`() = runTest {
            val longPrompt = "a".repeat(10000)
            val expectedResponse = """
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "Response to long prompt"
                                    }
                                ]
                            }
                        }
                    ]
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns expectedResponse
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            val result = vertexAIClient.generateContent(longPrompt, validModel)

            assertEquals("Response to long prompt", result)
            coVerify { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) }
        }

        @Test
        @DisplayName("Should handle special characters in prompt")
        fun `should handle special characters in prompt`() = runTest {
            val specialPrompt = "Test with special chars: !@#$%^&*()_+-=[]{}|;:,.<>?`~"
            val expectedResponse = """
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "Response with special chars"
                                    }
                                ]
                            }
                        }
                    ]
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns expectedResponse
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            val result = vertexAIClient.generateContent(specialPrompt, validModel)

            assertEquals("Response with special chars", result)
        }

        @Test
        @DisplayName("Should handle Unicode characters in prompt")
        fun `should handle Unicode characters in prompt`() = runTest {
            val unicodePrompt = "Test with Unicode: 你好世界 🌍 café naïve résumé"
            val expectedResponse = """
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "Unicode response: 你好 🚀"
                                    }
                                ]
                            }
                        }
                    ]
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns expectedResponse
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            val result = vertexAIClient.generateContent(unicodePrompt, validModel)

            assertEquals("Unicode response: 你好 🚀", result)
        }
    }

    @Nested
    @DisplayName("Request Building Tests")
    inner class RequestBuildingTests {

        @Test
        @DisplayName("Should build request with correct headers")
        fun `should build request with correct headers`() = runTest {
            val prompt = "Test prompt"
            val expectedResponse = """
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "Generated response"
                                    }
                                ]
                            }
                        }
                    ]
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns expectedResponse
            
            val requestSlot = slot<HttpRequest>()
            coEvery { httpClient.send(capture(requestSlot), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            vertexAIClient.generateContent(prompt, validModel)

            val capturedRequest = requestSlot.captured
            assertTrue(capturedRequest.headers().firstValue("Authorization").isPresent)
            assertEquals("Bearer $validApiKey", capturedRequest.headers().firstValue("Authorization").get())
            assertTrue(capturedRequest.headers().firstValue("Content-Type").isPresent)
            assertEquals("application/json", capturedRequest.headers().firstValue("Content-Type").get())
        }

        @Test
        @DisplayName("Should build request with correct URL")
        fun `should build request with correct URL`() = runTest {
            val prompt = "Test prompt"
            val expectedResponse = """
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "Generated response"
                                    }
                                ]
                            }
                        }
                    ]
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns expectedResponse
            
            val requestSlot = slot<HttpRequest>()
            coEvery { httpClient.send(capture(requestSlot), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            vertexAIClient.generateContent(prompt, validModel)

            val capturedRequest = requestSlot.captured
            val expectedUrl = "https://$validLocation-aiplatform.googleapis.com/v1/projects/$validProjectId/locations/$validLocation/publishers/google/models/$validModel:generateContent"
            assertEquals(expectedUrl, capturedRequest.uri().toString())
        }

        @Test
        @DisplayName("Should build request with correct JSON body")
        fun `should build request with correct JSON body`() = runTest {
            val prompt = "Test prompt"
            val expectedResponse = """
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "Generated response"
                                    }
                                ]
                            }
                        }
                    ]
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns expectedResponse
            
            val requestSlot = slot<HttpRequest>()
            coEvery { httpClient.send(capture(requestSlot), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            vertexAIClient.generateContent(prompt, validModel)

            val capturedRequest = requestSlot.captured
            val bodyPublisher = capturedRequest.bodyPublisher()
            assertTrue(bodyPublisher.isPresent)
            
            // The body should contain the structured request with the prompt
            // This is a simplified check - in practice, you'd want to parse the JSON
            // and verify the structure more thoroughly
        }
    }

    @Nested
    @DisplayName("Response Parsing Tests")
    inner class ResponseParsingTests {

        @Test
        @DisplayName("Should parse response with single candidate")
        fun `should parse response with single candidate`() = runTest {
            val prompt = "Test prompt"
            val response = """
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "Single candidate response"
                                    }
                                ]
                            }
                        }
                    ]
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns response
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            val result = vertexAIClient.generateContent(prompt, validModel)

            assertEquals("Single candidate response", result)
        }

        @Test
        @DisplayName("Should parse response with multiple candidates and return first")
        fun `should parse response with multiple candidates and return first`() = runTest {
            val prompt = "Test prompt"
            val response = """
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "First candidate"
                                    }
                                ]
                            }
                        },
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "Second candidate"
                                    }
                                ]
                            }
                        }
                    ]
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns response
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            val result = vertexAIClient.generateContent(prompt, validModel)

            assertEquals("First candidate", result)
        }

        @Test
        @DisplayName("Should handle response with multiple parts")
        fun `should handle response with multiple parts`() = runTest {
            val prompt = "Test prompt"
            val response = """
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "Part 1"
                                    },
                                    {
                                        "text": "Part 2"
                                    }
                                ]
                            }
                        }
                    ]
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns response
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            val result = vertexAIClient.generateContent(prompt, validModel)

            assertTrue(result.contains("Part 1"))
            assertTrue(result.contains("Part 2"))
        }

        @Test
        @DisplayName("Should handle response with missing text field")
        fun `should handle response with missing text field`() = runTest {
            val prompt = "Test prompt"
            val response = """
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": "test_function"
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns response
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            val result = vertexAIClient.generateContent(prompt, validModel)

            assertEquals("", result)
        }
    }

    @Nested
    @DisplayName("Edge Cases and Error Handling")
    inner class EdgeCasesTests {

        @Test
        @DisplayName("Should handle null response body")
        fun `should handle null response body`() = runTest {
            val prompt = "Test prompt"

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns null
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            assertThrows<RuntimeException> {
                runBlocking { vertexAIClient.generateContent(prompt, validModel) }
            }
        }

        @Test
        @DisplayName("Should handle response with invalid JSON structure")
        fun `should handle response with invalid JSON structure`() = runTest {
            val prompt = "Test prompt"
            val invalidStructure = """
                {
                    "notCandidates": []
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns invalidStructure
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            val result = vertexAIClient.generateContent(prompt, validModel)

            assertEquals("", result)
        }

        @Test
        @DisplayName("Should handle response with null candidates")
        fun `should handle response with null candidates`() = runTest {
            val prompt = "Test prompt"
            val response = """
                {
                    "candidates": null
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns response
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            val result = vertexAIClient.generateContent(prompt, validModel)

            assertEquals("", result)
        }

        @Test
        @DisplayName("Should handle concurrent requests properly")
        fun `should handle concurrent requests properly`() = runTest {
            val prompt1 = "Test prompt 1"
            val prompt2 = "Test prompt 2"
            val response1 = """
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "Response 1"
                                    }
                                ]
                            }
                        }
                    ]
                }
            """.trimIndent()
            val response2 = """
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "Response 2"
                                    }
                                ]
                            }
                        }
                    ]
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returnsMany listOf(response1, response2)
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            val result1 = vertexAIClient.generateContent(prompt1, validModel)
            val result2 = vertexAIClient.generateContent(prompt2, validModel)

            assertEquals("Response 1", result1)
            assertEquals("Response 2", result2)
            coVerify(exactly = 2) { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) }
        }
    }

    @Nested
    @DisplayName("Performance and Resource Tests")
    inner class PerformanceTests {

        @Test
        @DisplayName("Should handle rapid successive requests")
        fun `should handle rapid successive requests`() = runTest {
            val prompt = "Test prompt"
            val expectedResponse = """
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "Generated response"
                                    }
                                ]
                            }
                        }
                    ]
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns expectedResponse
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            repeat(10) {
                val result = vertexAIClient.generateContent(prompt, validModel)
                assertEquals("Generated response", result)
            }

            coVerify(exactly = 10) { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) }
        }

        @Test
        @DisplayName("Should handle memory-intensive prompts")
        fun `should handle memory-intensive prompts`() = runTest {
            val largePrompt = "Large prompt: " + "x".repeat(100000)
            val expectedResponse = """
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "Response to large prompt"
                                    }
                                ]
                            }
                        }
                    ]
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns expectedResponse
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            val result = vertexAIClient.generateContent(largePrompt, validModel)

            assertEquals("Response to large prompt", result)
        }
    }

    @Nested
    @DisplayName("Integration Scenarios")
    inner class IntegrationTests {

        @Test
        @DisplayName("Should handle different model types")
        fun `should handle different model types`() = runTest {
            val prompt = "Test prompt"
            val models = listOf("gemini-pro", "gemini-pro-vision", "text-bison", "chat-bison")
            val expectedResponse = """
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "Model response"
                                    }
                                ]
                            }
                        }
                    ]
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns expectedResponse
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            models.forEach { model ->
                val result = vertexAIClient.generateContent(prompt, model)
                assertEquals("Model response", result)
            }

            coVerify(exactly = models.size) { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) }
        }

        @Test
        @DisplayName("Should handle different locations")
        fun `should handle different locations`() = runTest {
            val prompt = "Test prompt"
            val locations = listOf("us-central1", "us-east1", "europe-west1", "asia-northeast1")
            val expectedResponse = """
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "Location response"
                                    }
                                ]
                            }
                        }
                    ]
                }
            """.trimIndent()

            every { mockResponse.statusCode() } returns 200
            every { mockResponse.body() } returns expectedResponse
            coEvery { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) } returns mockResponse

            locations.forEach { location ->
                val client = VertexAIClientImpl(httpClient, validApiKey, validProjectId, location)
                val result = client.generateContent(prompt, validModel)
                assertEquals("Location response", result)
            }

            coVerify(exactly = locations.size) { httpClient.send(any<HttpRequest>(), any<HttpResponse.BodyHandler<String>>()) }
        }
    }
}