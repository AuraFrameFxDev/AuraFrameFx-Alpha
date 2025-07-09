package dev.aurakai.auraframefx.ai.services

import io.mockk.*
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.junit.jupiter.params.provider.CsvSource
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import java.util.stream.Stream

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@DisplayName("GenesisBridgeService Tests")
class GenesisBridgeServiceTest {

    private lateinit var genesisBridgeService: GenesisBridgeService
    private val mockHttpClient = mockk<HttpClient>()
    private val mockLogger = mockk<Logger>()
    private val mockConfigService = mockk<ConfigService>()
    private val mockRetryPolicy = mockk<RetryPolicy>()

    @BeforeEach
    fun setUp() {
        clearAllMocks()
        genesisBridgeService = GenesisBridgeService(
            httpClient = mockHttpClient,
            logger = mockLogger,
            configService = mockConfigService,
            retryPolicy = mockRetryPolicy
        )
    }

    @Nested
    @DisplayName("Connection Tests")
    inner class ConnectionTests {

        @Test
        @DisplayName("Should successfully establish connection with valid credentials")
        fun `should successfully establish connection with valid credentials`() = runTest {
            // Given
            val validApiKey = "valid-api-key"
            val validEndpoint = "https://api.genesis.ai/v1"
            
            every { mockConfigService.getApiKey() } returns validApiKey
            every { mockConfigService.getEndpoint() } returns validEndpoint
            coEvery { mockHttpClient.get(any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"status":"connected","version":"1.0.0"}"""
            }

            // When
            val result = genesisBridgeService.connect()

            // Then
            assertTrue(result.isSuccess)
            verify { mockLogger.info("Successfully connected to Genesis AI") }
            coVerify { mockHttpClient.get("$validEndpoint/health") }
        }

        @Test
        @DisplayName("Should fail connection with invalid API key")
        fun `should fail connection with invalid API key`() = runTest {
            // Given
            val invalidApiKey = "invalid-api-key"
            val validEndpoint = "https://api.genesis.ai/v1"
            
            every { mockConfigService.getApiKey() } returns invalidApiKey
            every { mockConfigService.getEndpoint() } returns validEndpoint
            coEvery { mockHttpClient.get(any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns false
                every { statusCode } returns 401
                every { body } returns """{"error":"Unauthorized","message":"Invalid API key"}"""
            }

            // When
            val result = genesisBridgeService.connect()

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("Failed to connect to Genesis AI: Unauthorized") }
        }

        @Test
        @DisplayName("Should handle network timeout gracefully")
        fun `should handle network timeout gracefully`() = runTest {
            // Given
            every { mockConfigService.getApiKey() } returns "valid-key"
            every { mockConfigService.getEndpoint() } returns "https://api.genesis.ai/v1"
            coEvery { mockHttpClient.get(any()) } throws TimeoutException("Connection timeout")

            // When
            val result = genesisBridgeService.connect()

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("Connection timeout to Genesis AI") }
        }

        @Test
        @DisplayName("Should handle null or empty API key")
        fun `should handle null or empty API key`() = runTest {
            // Given
            every { mockConfigService.getApiKey() } returns null
            every { mockConfigService.getEndpoint() } returns "https://api.genesis.ai/v1"

            // When & Then
            assertThrows<IllegalArgumentException> {
                runBlocking { genesisBridgeService.connect() }
            }
        }

        @ParameterizedTest
        @ValueSource(strings = ["", "   ", "\t", "\n"])
        @DisplayName("Should reject empty or whitespace API keys")
        fun `should reject empty or whitespace API keys`(apiKey: String) = runTest {
            // Given
            every { mockConfigService.getApiKey() } returns apiKey
            every { mockConfigService.getEndpoint() } returns "https://api.genesis.ai/v1"

            // When & Then
            assertThrows<IllegalArgumentException> {
                runBlocking { genesisBridgeService.connect() }
            }
        }
    }

    @Nested
    @DisplayName("AI Model Interaction Tests")
    inner class AIModelInteractionTests {

        @Test
        @DisplayName("Should successfully generate text response")
        fun `should successfully generate text response`() = runTest {
            // Given
            val prompt = "Generate a creative story about AI"
            val expectedResponse = "Once upon a time, in a world where artificial intelligence..."
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"response":"$expectedResponse","tokens_used":150}"""
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isSuccess)
            assertEquals(expectedResponse, result.getOrNull())
            coVerify { mockHttpClient.post(any(), match { body -> 
                body.contains("\"prompt\":\"$prompt\"") 
            }) }
        }

        @Test
        @DisplayName("Should handle empty prompt gracefully")
        fun `should handle empty prompt gracefully`() = runTest {
            // Given
            val emptyPrompt = ""

            // When & Then
            assertThrows<IllegalArgumentException> {
                runBlocking { genesisBridgeService.generateText(emptyPrompt) }
            }
        }

        @Test
        @DisplayName("Should handle very long prompts")
        fun `should handle very long prompts`() = runTest {
            // Given
            val longPrompt = "a".repeat(10000)
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns false
                every { statusCode } returns 413
                every { body } returns """{"error":"Payload too large"}"""
            }

            // When
            val result = genesisBridgeService.generateText(longPrompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.warn("Prompt too long, consider truncating") }
        }

        @ParameterizedTest
        @CsvSource(
            "'What is AI?', 'AI is artificial intelligence'",
            "'Hello world', 'Hello! How can I help you today?'",
            "'Explain quantum computing', 'Quantum computing uses quantum mechanical phenomena...'"
        )
        @DisplayName("Should handle various prompt types")
        fun `should handle various prompt types`(prompt: String, expectedResponse: String) = runTest {
            // Given
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"response":"$expectedResponse","tokens_used":50}"""
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isSuccess)
            assertEquals(expectedResponse, result.getOrNull())
        }
    }

    @Nested
    @DisplayName("Configuration Management Tests")
    inner class ConfigurationTests {

        @Test
        @DisplayName("Should update configuration successfully")
        fun `should update configuration successfully`() = runTest {
            // Given
            val newConfig = mapOf(
                "temperature" to 0.7,
                "max_tokens" to 1000,
                "model" to "genesis-v2"
            )
            
            every { mockConfigService.updateConfig(any()) } returns true

            // When
            val result = genesisBridgeService.updateConfiguration(newConfig)

            // Then
            assertTrue(result)
            verify { mockConfigService.updateConfig(newConfig) }
            verify { mockLogger.info("Configuration updated successfully") }
        }

        @Test
        @DisplayName("Should validate configuration parameters")
        fun `should validate configuration parameters`() = runTest {
            // Given
            val invalidConfig = mapOf(
                "temperature" to 2.5, // Invalid: should be between 0 and 1
                "max_tokens" to -100  // Invalid: should be positive
            )

            // When & Then
            assertThrows<IllegalArgumentException> {
                runBlocking { genesisBridgeService.updateConfiguration(invalidConfig) }
            }
        }

        @Test
        @DisplayName("Should handle configuration update failures")
        fun `should handle configuration update failures`() = runTest {
            // Given
            val config = mapOf("temperature" to 0.5)
            every { mockConfigService.updateConfig(any()) } returns false

            // When
            val result = genesisBridgeService.updateConfiguration(config)

            // Then
            assertFalse(result)
            verify { mockLogger.error("Failed to update configuration") }
        }
    }

    @Nested
    @DisplayName("Retry Policy Tests")
    inner class RetryPolicyTests {

        @Test
        @DisplayName("Should retry on transient failures")
        fun `should retry on transient failures`() = runTest {
            // Given
            val prompt = "Test prompt"
            every { mockRetryPolicy.shouldRetry(any(), any()) } returnsMany listOf(true, true, false)
            
            coEvery { mockHttpClient.post(any(), any()) } throws Exception("Network error")

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify(exactly = 3) { mockRetryPolicy.shouldRetry(any(), any()) }
            coVerify(exactly = 3) { mockHttpClient.post(any(), any()) }
        }

        @Test
        @DisplayName("Should not retry on permanent failures")
        fun `should not retry on permanent failures`() = runTest {
            // Given
            val prompt = "Test prompt"
            every { mockRetryPolicy.shouldRetry(any(), any()) } returns false
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns false
                every { statusCode } returns 400
                every { body } returns """{"error":"Bad request"}"""
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify(exactly = 1) { mockRetryPolicy.shouldRetry(any(), any()) }
            coVerify(exactly = 1) { mockHttpClient.post(any(), any()) }
        }

        @Test
        @DisplayName("Should succeed after retries")
        fun `should succeed after retries`() = runTest {
            // Given
            val prompt = "Test prompt"
            val expectedResponse = "Success after retry"
            
            every { mockRetryPolicy.shouldRetry(any(), any()) } returnsMany listOf(true, false)
            
            coEvery { mockHttpClient.post(any(), any()) } 
                .throwsMany(listOf(Exception("Temporary failure"), Exception("Another failure")))
                .andThen(mockk<HttpResponse> {
                    every { isSuccessful } returns true
                    every { body } returns """{"response":"$expectedResponse","tokens_used":100}"""
                })

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isSuccess)
            assertEquals(expectedResponse, result.getOrNull())
            coVerify(exactly = 3) { mockHttpClient.post(any(), any()) }
        }
    }

    @Nested
    @DisplayName("Error Handling Tests")
    inner class ErrorHandlingTests {

        @Test
        @DisplayName("Should handle JSON parsing errors gracefully")
        fun `should handle JSON parsing errors gracefully`() = runTest {
            // Given
            val prompt = "Test prompt"
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns "Invalid JSON {{"
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("Failed to parse response JSON") }
        }

        @Test
        @DisplayName("Should handle service unavailable errors")
        fun `should handle service unavailable errors`() = runTest {
            // Given
            val prompt = "Test prompt"
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns false
                every { statusCode } returns 503
                every { body } returns """{"error":"Service unavailable"}"""
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.warn("Genesis AI service temporarily unavailable") }
        }

        @Test
        @DisplayName("Should handle rate limiting gracefully")
        fun `should handle rate limiting gracefully`() = runTest {
            // Given
            val prompt = "Test prompt"
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns false
                every { statusCode } returns 429
                every { body } returns """{"error":"Rate limit exceeded","retry_after":"60"}"""
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.warn("Rate limit exceeded, retry after 60 seconds") }
        }

        @Test
        @DisplayName("Should handle unexpected server errors")
        fun `should handle unexpected server errors`() = runTest {
            // Given
            val prompt = "Test prompt"
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns false
                every { statusCode } returns 500
                every { body } returns """{"error":"Internal server error"}"""
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("Internal server error from Genesis AI") }
        }
    }

    @Nested
    @DisplayName("State Management Tests")
    inner class StateManagementTests {

        @Test
        @DisplayName("Should maintain connection state correctly")
        fun `should maintain connection state correctly`() = runTest {
            // Given
            every { mockConfigService.getApiKey() } returns "valid-key"
            every { mockConfigService.getEndpoint() } returns "https://api.genesis.ai/v1"
            coEvery { mockHttpClient.get(any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"status":"connected"}"""
            }

            // When
            val connectResult = genesisBridgeService.connect()
            val isConnected = genesisBridgeService.isConnected()

            // Then
            assertTrue(connectResult.isSuccess)
            assertTrue(isConnected)
        }

        @Test
        @DisplayName("Should handle disconnection properly")
        fun `should handle disconnection properly`() = runTest {
            // Given - First establish connection
            every { mockConfigService.getApiKey() } returns "valid-key"
            every { mockConfigService.getEndpoint() } returns "https://api.genesis.ai/v1"
            coEvery { mockHttpClient.get(any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"status":"connected"}"""
            }

            // When
            genesisBridgeService.connect()
            genesisBridgeService.disconnect()
            val isConnected = genesisBridgeService.isConnected()

            // Then
            assertFalse(isConnected)
            verify { mockLogger.info("Disconnected from Genesis AI") }
        }

        @Test
        @DisplayName("Should track usage statistics")
        fun `should track usage statistics`() = runTest {
            // Given
            val prompt = "Test prompt"
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"response":"Test response","tokens_used":50}"""
            }

            // When
            genesisBridgeService.generateText(prompt)
            val stats = genesisBridgeService.getUsageStats()

            // Then
            assertNotNull(stats)
            assertEquals(1, stats.requestCount)
            assertEquals(50, stats.tokensUsed)
        }
    }

    @Nested
    @DisplayName("Concurrency Tests")
    inner class ConcurrencyTests {

        @Test
        @DisplayName("Should handle concurrent requests safely")
        fun `should handle concurrent requests safely`() = runTest {
            // Given
            val prompts = (1..10).map { "Prompt $it" }
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"response":"Response","tokens_used":25}"""
            }

            // When
            val results = prompts.map { prompt ->
                async { genesisBridgeService.generateText(prompt) }
            }.awaitAll()

            // Then
            assertTrue(results.all { it.isSuccess })
            coVerify(exactly = 10) { mockHttpClient.post(any(), any()) }
        }

        @Test
        @DisplayName("Should handle concurrent configuration updates")
        fun `should handle concurrent configuration updates`() = runTest {
            // Given
            val configs = (1..5).map { index ->
                mapOf("temperature" to 0.1 * index)
            }
            
            every { mockConfigService.updateConfig(any()) } returns true

            // When
            val results = configs.map { config ->
                async { genesisBridgeService.updateConfiguration(config) }
            }.awaitAll()

            // Then
            assertTrue(results.all { it })
            verify(exactly = 5) { mockConfigService.updateConfig(any()) }
        }
    }

    @Nested
    @DisplayName("Integration Edge Cases")
    inner class IntegrationEdgeCases {

        @Test
        @DisplayName("Should handle malformed API responses")
        fun `should handle malformed API responses`() = runTest {
            // Given
            val prompt = "Test prompt"
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"incomplete": true, "missing_field":"""
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error(match { it.contains("malformed response") }) }
        }

        @Test
        @DisplayName("Should handle special characters in prompts")
        fun `should handle special characters in prompts`() = runTest {
            // Given
            val specialPrompt = "Test with émojis 🚀 and unicode ñ characters"
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"response":"Handled special chars","tokens_used":30}"""
            }

            // When
            val result = genesisBridgeService.generateText(specialPrompt)

            // Then
            assertTrue(result.isSuccess)
            assertEquals("Handled special chars", result.getOrNull())
        }

        @Test
        @DisplayName("Should handle connection recovery after failure")
        fun `should handle connection recovery after failure`() = runTest {
            // Given
            every { mockConfigService.getApiKey() } returns "valid-key"
            every { mockConfigService.getEndpoint() } returns "https://api.genesis.ai/v1"
            
            // First call fails
            coEvery { mockHttpClient.get(any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns false
                every { statusCode } returns 500
            } andThen mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"status":"connected"}"""
            }

            // When
            val firstResult = genesisBridgeService.connect()
            val secondResult = genesisBridgeService.connect()

            // Then
            assertTrue(firstResult.isFailure)
            assertTrue(secondResult.isSuccess)
        }
    }

    companion object {
        @JvmStatic
        fun promptTestData(): Stream<Arguments> = Stream.of(
            Arguments.of("Simple question", "Simple answer"),
            Arguments.of("Complex query with multiple parts", "Detailed response"),
            Arguments.of("Edge case with numbers 123456", "Numeric response")
        )
    }
}
    @Nested
    @DisplayName("Additional Comprehensive Tests")
    inner class AdditionalComprehensiveTests {

        @Test
        @DisplayName("Should handle null HttpResponse gracefully")
        fun `should handle null HttpResponse gracefully`() = runTest {
            // Given
            val prompt = "Test prompt"
            coEvery { mockHttpClient.post(any(), any()) } returns null

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("Received null response from Genesis AI") }
        }

        @Test
        @DisplayName("Should handle empty response body")
        fun `should handle empty response body`() = runTest {
            // Given
            val prompt = "Test prompt"
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns ""
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("Received empty response body from Genesis AI") }
        }

        @Test
        @DisplayName("Should handle response with null body")
        fun `should handle response with null body`() = runTest {
            // Given
            val prompt = "Test prompt"
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns null
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("Received null response body from Genesis AI") }
        }

        @Test
        @DisplayName("Should handle configuration with null values")
        fun `should handle configuration with null values`() = runTest {
            // Given
            val configWithNulls = mapOf(
                "temperature" to null,
                "max_tokens" to 1000,
                "model" to null
            )

            // When & Then
            assertThrows<IllegalArgumentException> {
                runBlocking { genesisBridgeService.updateConfiguration(configWithNulls) }
            }
            verify { mockLogger.error("Configuration contains null values") }
        }

        @Test
        @DisplayName("Should handle empty configuration map")
        fun `should handle empty configuration map`() = runTest {
            // Given
            val emptyConfig = emptyMap<String, Any>()

            // When & Then
            assertThrows<IllegalArgumentException> {
                runBlocking { genesisBridgeService.updateConfiguration(emptyConfig) }
            }
            verify { mockLogger.error("Configuration cannot be empty") }
        }

        @Test
        @DisplayName("Should validate model names in configuration")
        fun `should validate model names in configuration`() = runTest {
            // Given
            val configWithInvalidModel = mapOf(
                "model" to "invalid-model-name-with-special-chars@#$"
            )

            // When & Then
            assertThrows<IllegalArgumentException> {
                runBlocking { genesisBridgeService.updateConfiguration(configWithInvalidModel) }
            }
            verify { mockLogger.error("Invalid model name in configuration") }
        }

        @Test
        @DisplayName("Should handle HTTP client throwing unexpected exceptions")
        fun `should handle HTTP client throwing unexpected exceptions`() = runTest {
            // Given
            val prompt = "Test prompt"
            coEvery { mockHttpClient.post(any(), any()) } throws RuntimeException("Unexpected error")

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("Unexpected error occurred: Unexpected error") }
        }

        @Test
        @DisplayName("Should handle SSL/TLS connection failures")
        fun `should handle SSL TLS connection failures`() = runTest {
            // Given
            every { mockConfigService.getApiKey() } returns "valid-key"
            every { mockConfigService.getEndpoint() } returns "https://api.genesis.ai/v1"
            coEvery { mockHttpClient.get(any()) } throws SSLException("SSL handshake failed")

            // When
            val result = genesisBridgeService.connect()

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("SSL/TLS connection failed: SSL handshake failed") }
        }

        @Test
        @DisplayName("Should handle DNS resolution failures")
        fun `should handle DNS resolution failures`() = runTest {
            // Given
            every { mockConfigService.getApiKey() } returns "valid-key"
            every { mockConfigService.getEndpoint() } returns "https://nonexistent.domain.com/v1"
            coEvery { mockHttpClient.get(any()) } throws UnknownHostException("Host not found")

            // When
            val result = genesisBridgeService.connect()

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("DNS resolution failed: Host not found") }
        }

        @Test
        @DisplayName("Should handle extremely large response payloads")
        fun `should handle extremely large response payloads`() = runTest {
            // Given
            val prompt = "Test prompt"
            val largeResponse = "x".repeat(1000000) // 1MB response
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"response":"$largeResponse","tokens_used":50000}"""
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isSuccess)
            assertEquals(largeResponse, result.getOrNull())
            verify { mockLogger.info("Received large response payload") }
        }

        @Test
        @DisplayName("Should handle concurrent connections attempts")
        fun `should handle concurrent connection attempts`() = runTest {
            // Given
            every { mockConfigService.getApiKey() } returns "valid-key"
            every { mockConfigService.getEndpoint() } returns "https://api.genesis.ai/v1"
            coEvery { mockHttpClient.get(any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"status":"connected","version":"1.0.0"}"""
            }

            // When
            val connections = (1..5).map {
                async { genesisBridgeService.connect() }
            }.awaitAll()

            // Then
            assertTrue(connections.all { it.isSuccess })
            coVerify(exactly = 5) { mockHttpClient.get(any()) }
        }

        @Test
        @DisplayName("Should handle response with missing required fields")
        fun `should handle response with missing required fields`() = runTest {
            // Given
            val prompt = "Test prompt"
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"tokens_used":100}""" // Missing response field
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("Response missing required fields") }
        }

        @Test
        @DisplayName("Should handle response with extra unexpected fields")
        fun `should handle response with extra unexpected fields`() = runTest {
            // Given
            val prompt = "Test prompt"
            val expectedResponse = "Test response"
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"response":"$expectedResponse","tokens_used":50,"unexpected_field":"value","another_field":123}"""
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isSuccess)
            assertEquals(expectedResponse, result.getOrNull())
            verify { mockLogger.debug("Response contains unexpected fields") }
        }

        @Test
        @DisplayName("Should handle configuration service throwing exceptions")
        fun `should handle configuration service throwing exceptions`() = runTest {
            // Given
            every { mockConfigService.getApiKey() } throws RuntimeException("Config service error")
            every { mockConfigService.getEndpoint() } returns "https://api.genesis.ai/v1"

            // When & Then
            assertThrows<RuntimeException> {
                runBlocking { genesisBridgeService.connect() }
            }
            verify { mockLogger.error("Configuration service error: Config service error") }
        }

        @Test
        @DisplayName("Should handle logger throwing exceptions")
        fun `should handle logger throwing exceptions`() = runTest {
            // Given
            val prompt = "Test prompt"
            every { mockLogger.info(any()) } throws RuntimeException("Logger error")
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"response":"Test response","tokens_used":50}"""
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isSuccess) // Should still succeed despite logger error
            verify { mockLogger.info(any()) }
        }

        @Test
        @DisplayName("Should handle retry policy throwing exceptions")
        fun `should handle retry policy throwing exceptions`() = runTest {
            // Given
            val prompt = "Test prompt"
            every { mockRetryPolicy.shouldRetry(any(), any()) } throws RuntimeException("Retry policy error")
            coEvery { mockHttpClient.post(any(), any()) } throws Exception("Network error")

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("Retry policy error: Retry policy error") }
        }

        @Test
        @DisplayName("Should handle multiple rapid disconnect/reconnect cycles")
        fun `should handle multiple rapid disconnect reconnect cycles`() = runTest {
            // Given
            every { mockConfigService.getApiKey() } returns "valid-key"
            every { mockConfigService.getEndpoint() } returns "https://api.genesis.ai/v1"
            coEvery { mockHttpClient.get(any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"status":"connected"}"""
            }

            // When
            repeat(10) {
                genesisBridgeService.connect()
                genesisBridgeService.disconnect()
            }

            // Then
            assertFalse(genesisBridgeService.isConnected())
            verify(exactly = 10) { mockLogger.info("Disconnected from Genesis AI") }
        }

        @Test
        @DisplayName("Should handle statistics overflow scenarios")
        fun `should handle statistics overflow scenarios`() = runTest {
            // Given
            val prompt = "Test prompt"
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"response":"Test response","tokens_used":${Long.MAX_VALUE}}"""
            }

            // When
            repeat(1000) {
                genesisBridgeService.generateText(prompt)
            }
            val stats = genesisBridgeService.getUsageStats()

            // Then
            assertNotNull(stats)
            assertTrue(stats.requestCount >= 1000)
            verify { mockLogger.warn("Usage statistics may have overflowed") }
        }

        @ParameterizedTest
        @ValueSource(ints = [0, -1, -100, Int.MIN_VALUE])
        @DisplayName("Should handle invalid max_tokens values")
        fun `should handle invalid max_tokens values`(maxTokens: Int) = runTest {
            // Given
            val config = mapOf("max_tokens" to maxTokens)

            // When & Then
            assertThrows<IllegalArgumentException> {
                runBlocking { genesisBridgeService.updateConfiguration(config) }
            }
        }

        @ParameterizedTest
        @ValueSource(doubles = [-0.1, 1.1, 2.0, Double.NaN, Double.POSITIVE_INFINITY])
        @DisplayName("Should handle invalid temperature values")
        fun `should handle invalid temperature values`(temperature: Double) = runTest {
            // Given
            val config = mapOf("temperature" to temperature)

            // When & Then
            assertThrows<IllegalArgumentException> {
                runBlocking { genesisBridgeService.updateConfiguration(config) }
            }
        }

        @Test
        @DisplayName("Should handle graceful shutdown scenarios")
        fun `should handle graceful shutdown scenarios`() = runTest {
            // Given
            every { mockConfigService.getApiKey() } returns "valid-key"
            every { mockConfigService.getEndpoint() } returns "https://api.genesis.ai/v1"
            coEvery { mockHttpClient.get(any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"status":"connected"}"""
            }

            // When
            genesisBridgeService.connect()
            genesisBridgeService.shutdown()

            // Then
            assertFalse(genesisBridgeService.isConnected())
            verify { mockLogger.info("GenesisBridgeService shutdown completed") }
        }

        @Test
        @DisplayName("Should handle partial response streaming scenarios")
        fun `should handle partial response streaming scenarios`() = runTest {
            // Given
            val prompt = "Test prompt"
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"response":"Partial response...","streaming":true,"tokens_used":25}"""
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isSuccess)
            assertEquals("Partial response...", result.getOrNull())
            verify { mockLogger.info("Received streaming response") }
        }
    }
}