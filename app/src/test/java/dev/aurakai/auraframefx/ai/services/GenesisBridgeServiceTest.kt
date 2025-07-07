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
    @DisplayName("Additional Edge Cases and Boundary Tests")
    inner class AdditionalEdgeCases {

        @Test
        @DisplayName("Should handle extremely large response payloads")
        fun `should handle extremely large response payloads`() = runTest {
            // Given
            val prompt = "Generate a large response"
            val largeResponse = "x".repeat(1000000) // 1MB response
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"response":"$largeResponse","tokens_used":10000}"""
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isSuccess)
            assertEquals(largeResponse, result.getOrNull())
            verify { mockLogger.info("Processing large response payload") }
        }

        @Test
        @DisplayName("Should handle null response bodies")
        fun `should handle null response bodies`() = runTest {
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
            verify { mockLogger.error("Received null response body") }
        }

        @Test
        @DisplayName("Should handle connection state during service shutdown")
        fun `should handle connection state during service shutdown`() = runTest {
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
            val isConnected = genesisBridgeService.isConnected()

            // Then
            assertFalse(isConnected)
            verify { mockLogger.info("GenesisBridgeService shutdown completed") }
        }

        @ParameterizedTest
        @ValueSource(ints = [100, 401, 403, 404, 408, 429, 500, 502, 503, 504])
        @DisplayName("Should handle various HTTP status codes appropriately")
        fun `should handle various HTTP status codes appropriately`(statusCode: Int) = runTest {
            // Given
            val prompt = "Test prompt"
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns statusCode in 200..299
                every { statusCode } returns statusCode
                every { body } returns """{"error":"HTTP $statusCode"}"""
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            if (statusCode in 200..299) {
                assertTrue(result.isSuccess)
            } else {
                assertTrue(result.isFailure)
                verify { mockLogger.error(match { it.contains("HTTP $statusCode") }) }
            }
        }

        @Test
        @DisplayName("Should handle interrupted connection attempts")
        fun `should handle interrupted connection attempts`() = runTest {
            // Given
            every { mockConfigService.getApiKey() } returns "valid-key"
            every { mockConfigService.getEndpoint() } returns "https://api.genesis.ai/v1"
            coEvery { mockHttpClient.get(any()) } throws InterruptedException("Connection interrupted")

            // When
            val result = genesisBridgeService.connect()

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.warn("Connection attempt was interrupted") }
        }

        @Test
        @DisplayName("Should handle SSL/TLS certificate errors")
        fun `should handle SSL TLS certificate errors`() = runTest {
            // Given
            val prompt = "Test prompt"
            
            coEvery { mockHttpClient.post(any(), any()) } throws SSLException("Certificate validation failed")

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("SSL certificate validation failed") }
        }

        @Test
        @DisplayName("Should handle DNS resolution failures")
        fun `should handle DNS resolution failures`() = runTest {
            // Given
            every { mockConfigService.getApiKey() } returns "valid-key"
            every { mockConfigService.getEndpoint() } returns "https://invalid.genesis.ai/v1"
            coEvery { mockHttpClient.get(any()) } throws UnknownHostException("DNS resolution failed")

            // When
            val result = genesisBridgeService.connect()

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("DNS resolution failed for Genesis AI endpoint") }
        }

        @Test
        @DisplayName("Should handle configuration with extreme values")
        fun `should handle configuration with extreme values`() = runTest {
            // Given
            val extremeConfig = mapOf(
                "temperature" to Double.MAX_VALUE,
                "max_tokens" to Int.MAX_VALUE,
                "timeout" to Long.MAX_VALUE
            )

            // When & Then
            assertThrows<IllegalArgumentException> {
                runBlocking { genesisBridgeService.updateConfiguration(extremeConfig) }
            }
            verify { mockLogger.error("Configuration values exceed allowed limits") }
        }

        @Test
        @DisplayName("Should handle memory pressure during large requests")
        fun `should handle memory pressure during large requests`() = runTest {
            // Given
            val largePrompt = "A".repeat(100000)
            
            coEvery { mockHttpClient.post(any(), any()) } throws OutOfMemoryError("Java heap space")

            // When
            val result = genesisBridgeService.generateText(largePrompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("Memory pressure detected during request processing") }
        }

        @Test
        @DisplayName("Should handle concurrent access to usage statistics")
        fun `should handle concurrent access to usage statistics`() = runTest {
            // Given
            val prompts = (1..100).map { "Prompt $it" }
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"response":"Response","tokens_used":10}"""
            }

            // When
            val jobs = prompts.map { prompt ->
                async { genesisBridgeService.generateText(prompt) }
            }
            jobs.awaitAll()
            val stats = genesisBridgeService.getUsageStats()

            // Then
            assertNotNull(stats)
            assertEquals(100, stats.requestCount)
            assertEquals(1000, stats.tokensUsed)
        }

        @Test
        @DisplayName("Should handle configuration rollback on partial failures")
        fun `should handle configuration rollback on partial failures`() = runTest {
            // Given
            val originalConfig = mapOf("temperature" to 0.5, "max_tokens" to 1000)
            val newConfig = mapOf("temperature" to 0.8, "max_tokens" to 2000, "invalid_param" to "invalid")
            
            every { mockConfigService.getConfig() } returns originalConfig
            every { mockConfigService.updateConfig(any()) } throws IllegalArgumentException("Invalid parameter")

            // When
            val result = genesisBridgeService.updateConfiguration(newConfig)

            // Then
            assertFalse(result)
            verify { mockLogger.warn("Configuration rollback initiated due to validation failure") }
            verify { mockConfigService.updateConfig(originalConfig) }
        }

        @Test
        @DisplayName("Should handle request cancellation gracefully")
        fun `should handle request cancellation gracefully`() = runTest {
            // Given
            val prompt = "Test prompt"
            
            coEvery { mockHttpClient.post(any(), any()) } throws CancellationException("Request cancelled")

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.info("Request was cancelled") }
        }

        @Test
        @DisplayName("Should handle response streaming interruption")
        fun `should handle response streaming interruption`() = runTest {
            // Given
            val prompt = "Test prompt for streaming"
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } throws IOException("Stream interrupted")
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("Response stream was interrupted") }
        }

        @Test
        @DisplayName("Should handle API quota exceeded scenarios")
        fun `should handle API quota exceeded scenarios`() = runTest {
            // Given
            val prompt = "Test prompt"
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns false
                every { statusCode } returns 429
                every { body } returns """{"error":"Quota exceeded","quota_reset":"2024-01-01T00:00:00Z"}"""
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.warn("API quota exceeded, resets at 2024-01-01T00:00:00Z") }
        }

        @Test
        @DisplayName("Should handle service maintenance mode")
        fun `should handle service maintenance mode`() = runTest {
            // Given
            every { mockConfigService.getApiKey() } returns "valid-key"
            every { mockConfigService.getEndpoint() } returns "https://api.genesis.ai/v1"
            
            coEvery { mockHttpClient.get(any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns false
                every { statusCode } returns 503
                every { body } returns """{"error":"Service under maintenance","estimated_duration":"30 minutes"}"""
            }

            // When
            val result = genesisBridgeService.connect()

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.warn("Genesis AI service is under maintenance (estimated duration: 30 minutes)") }
        }

        @Test
        @DisplayName("Should handle partial response corruption")
        fun `should handle partial response corruption`() = runTest {
            // Given
            val prompt = "Test prompt"
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"response":"Valid start... \u0000\u0001\u0002 corrupted data"}"""
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("Response contains corrupted data") }
        }

        @Test
        @DisplayName("Should handle connection pool exhaustion")
        fun `should handle connection pool exhaustion`() = runTest {
            // Given
            val prompt = "Test prompt"
            
            coEvery { mockHttpClient.post(any(), any()) } throws PoolTimeoutException("Connection pool exhausted")

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("Connection pool exhausted, consider increasing pool size") }
        }

        @Test
        @DisplayName("Should handle configuration validation with nested objects")
        fun `should handle configuration validation with nested objects`() = runTest {
            // Given
            val nestedConfig = mapOf(
                "model_params" to mapOf(
                    "temperature" to 0.7,
                    "top_p" to 0.9,
                    "frequency_penalty" to 0.5
                ),
                "generation_params" to mapOf(
                    "max_tokens" to 1000,
                    "stop_sequences" to listOf(".", "!", "?")
                )
            )
            
            every { mockConfigService.updateConfig(any()) } returns true

            // When
            val result = genesisBridgeService.updateConfiguration(nestedConfig)

            // Then
            assertTrue(result)
            verify { mockConfigService.updateConfig(nestedConfig) }
            verify { mockLogger.info("Nested configuration validated and updated successfully") }
        }

        @Test
        @DisplayName("Should handle response with missing required fields")
        fun `should handle response with missing required fields`() = runTest {
            // Given
            val prompt = "Test prompt"
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"partial_response":"text","missing_tokens_field":true}"""
            }

            // When
            val result = genesisBridgeService.generateText(prompt)

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("Response missing required fields: tokens_used") }
        }

        @Test
        @DisplayName("Should handle API version mismatch")
        fun `should handle API version mismatch`() = runTest {
            // Given
            every { mockConfigService.getApiKey() } returns "valid-key"
            every { mockConfigService.getEndpoint() } returns "https://api.genesis.ai/v1"
            
            coEvery { mockHttpClient.get(any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns false
                every { statusCode } returns 426
                every { body } returns """{"error":"API version mismatch","supported_versions":["v2","v3"]}"""
            }

            // When
            val result = genesisBridgeService.connect()

            // Then
            assertTrue(result.isFailure)
            verify { mockLogger.error("API version mismatch, supported versions: [v2, v3]") }
        }

        @Test
        @DisplayName("Should handle resource cleanup on service disposal")
        fun `should handle resource cleanup on service disposal`() = runTest {
            // Given
            every { mockConfigService.getApiKey() } returns "valid-key"
            every { mockConfigService.getEndpoint() } returns "https://api.genesis.ai/v1"
            
            // When
            genesisBridgeService.dispose()

            // Then
            verify { mockHttpClient.close() }
            verify { mockLogger.info("GenesisBridgeService resources cleaned up") }
        }
    }

    @Nested
    @DisplayName("Performance and Load Testing")
    inner class PerformanceTests {

        @Test
        @DisplayName("Should handle high-frequency requests within rate limits")
        fun `should handle high frequency requests within rate limits`() = runTest {
            // Given
            val requestCount = 50
            val requests = (1..requestCount).map { "Request $it" }
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"response":"Response","tokens_used":10}"""
            }

            // When
            val startTime = System.currentTimeMillis()
            val results = requests.map { request ->
                async { genesisBridgeService.generateText(request) }
            }.awaitAll()
            val endTime = System.currentTimeMillis()

            // Then
            assertTrue(results.all { it.isSuccess })
            assertTrue(endTime - startTime < 5000) // Should complete within 5 seconds
            verify { mockLogger.info("Processed $requestCount requests in ${endTime - startTime}ms") }
        }

        @Test
        @DisplayName("Should handle request queue saturation gracefully")
        fun `should handle request queue saturation gracefully`() = runTest {
            // Given
            val overloadRequests = (1..1000).map { "Overload request $it" }
            
            coEvery { mockHttpClient.post(any(), any()) } throws RejectedExecutionException("Queue full")

            // When
            val results = overloadRequests.take(10).map { request ->
                genesisBridgeService.generateText(request)
            }

            // Then
            assertTrue(results.all { it.isFailure })
            verify { mockLogger.warn("Request queue is saturated, consider reducing load") }
        }

        @Test
        @DisplayName("Should handle memory efficient processing of large batches")
        fun `should handle memory efficient processing of large batches`() = runTest {
            // Given
            val batchSize = 100
            val batch = (1..batchSize).map { "Batch item $it" }
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"response":"Batch response","tokens_used":5}"""
            }

            // When
            val results = genesisBridgeService.processBatch(batch)

            // Then
            assertEquals(batchSize, results.size)
            assertTrue(results.all { it.isSuccess })
            verify { mockLogger.info("Processed batch of $batchSize items efficiently") }
        }
    }

    @Nested
    @DisplayName("Security and Validation Tests")
    inner class SecurityTests {

        @Test
        @DisplayName("Should sanitize sensitive data in logs")
        fun `should sanitize sensitive data in logs`() = runTest {
            // Given
            val sensitivePrompt = "My API key is sk-1234567890abcdef and my password is secret123"
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"response":"Sanitized response","tokens_used":25}"""
            }

            // When
            val result = genesisBridgeService.generateText(sensitivePrompt)

            // Then
            assertTrue(result.isSuccess)
            verify { mockLogger.info(match { !it.contains("sk-1234567890abcdef") && !it.contains("secret123") }) }
        }

        @Test
        @DisplayName("Should validate API key format")
        fun `should validate API key format`() = runTest {
            // Given
            val invalidApiKey = "invalid-key-format"
            every { mockConfigService.getApiKey() } returns invalidApiKey
            every { mockConfigService.getEndpoint() } returns "https://api.genesis.ai/v1"

            // When & Then
            assertThrows<IllegalArgumentException> {
                runBlocking { genesisBridgeService.connect() }
            }
            verify { mockLogger.error("Invalid API key format") }
        }

        @Test
        @DisplayName("Should handle potential injection attacks in prompts")
        fun `should handle potential injection attacks in prompts`() = runTest {
            // Given
            val maliciousPrompt = "'; DROP TABLE users; --"
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse> {
                every { isSuccessful } returns true
                every { body } returns """{"response":"Safe response","tokens_used":15}"""
            }

            // When
            val result = genesisBridgeService.generateText(maliciousPrompt)

            // Then
            assertTrue(result.isSuccess)
            verify { mockLogger.warn("Potentially malicious content detected and sanitized") }
        }

        @Test
        @DisplayName("Should enforce request size limits")
        fun `should enforce request size limits`() = runTest {
            // Given
            val oversizedPrompt = "A".repeat(1000000) // 1MB prompt
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                runBlocking { genesisBridgeService.generateText(oversizedPrompt) }
            }
            verify { mockLogger.error("Request exceeds maximum allowed size") }
        }

        @Test
        @DisplayName("Should validate endpoint URL format")
        fun `should validate endpoint URL format`() = runTest {
            // Given
            val invalidEndpoint = "not-a-valid-url"
            every { mockConfigService.getApiKey() } returns "valid-key"
            every { mockConfigService.getEndpoint() } returns invalidEndpoint

            // When & Then
            assertThrows<IllegalArgumentException> {
                runBlocking { genesisBridgeService.connect() }
            }
            verify { mockLogger.error("Invalid endpoint URL format") }
        }
    }
}