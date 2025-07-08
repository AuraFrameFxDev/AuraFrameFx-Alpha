package dev.aurakai.auraframefx.ai.services

import dev.aurakai.auraframefx.ai.models.AIRequest
import dev.aurakai.auraframefx.ai.models.AIResponse
import dev.aurakai.auraframefx.ai.models.AIError
import dev.aurakai.auraframefx.ai.models.AIProvider
import dev.aurakai.auraframefx.config.AIConfiguration
import io.mockk.*
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.junit.jupiter.params.provider.EnumSource
import org.junit.jupiter.params.provider.CsvSource
import java.time.Duration
import java.util.concurrent.TimeoutException

/**
 * Comprehensive unit tests for AuraAIService
 * Testing Framework: JUnit 5 with MockK for mocking
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class AuraAIServiceTest {

    private lateinit var auraAIService: AuraAIService
    private lateinit var mockConfig: AIConfiguration
    private lateinit var mockHttpClient: HttpClient
    private lateinit var mockLogger: Logger

    @BeforeEach
    fun setUp() {
        mockConfig = mockk<AIConfiguration>()
        mockHttpClient = mockk<HttpClient>()
        mockLogger = mockk<Logger>(relaxed = true)
        
        every { mockConfig.apiKey } returns "test-api-key"
        every { mockConfig.baseUrl } returns "https://api.test.com"
        every { mockConfig.timeout } returns Duration.ofSeconds(30)
        every { mockConfig.maxRetries } returns 3
        every { mockConfig.defaultModel } returns "gpt-4"
        
        auraAIService = AuraAIService(mockConfig, mockHttpClient, mockLogger)
    }

    @AfterEach
    fun tearDown() {
        clearAllMocks()
    }

    @Nested
    @DisplayName("Service Initialization Tests")
    inner class InitializationTests {

        @Test
        @DisplayName("Should initialize service with valid configuration")
        fun `should initialize service with valid configuration`() {
            // Given
            val validConfig = mockk<AIConfiguration>()
            every { validConfig.apiKey } returns "valid-key"
            every { validConfig.baseUrl } returns "https://api.valid.com"
            every { validConfig.timeout } returns Duration.ofSeconds(60)

            // When
            val service = AuraAIService(validConfig, mockHttpClient, mockLogger)

            // Then
            assertNotNull(service)
            assertTrue(service.isInitialized())
        }

        @Test
        @DisplayName("Should throw exception when API key is null or empty")
        fun `should throw exception when API key is null or empty`() {
            // Given
            every { mockConfig.apiKey } returns null

            // When & Then
            assertThrows<IllegalArgumentException> {
                AuraAIService(mockConfig, mockHttpClient, mockLogger)
            }
        }

        @Test
        @DisplayName("Should throw exception when API key is blank")
        fun `should throw exception when API key is blank`() {
            // Given
            every { mockConfig.apiKey } returns "   "

            // When & Then
            assertThrows<IllegalArgumentException> {
                AuraAIService(mockConfig, mockHttpClient, mockLogger)
            }
        }

        @Test
        @DisplayName("Should throw exception when base URL is invalid")
        fun `should throw exception when base URL is invalid`() {
            // Given
            every { mockConfig.baseUrl } returns "invalid-url"

            // When & Then
            assertThrows<IllegalArgumentException> {
                AuraAIService(mockConfig, mockHttpClient, mockLogger)
            }
        }

        @Test
        @DisplayName("Should handle null timeout gracefully")
        fun `should handle null timeout gracefully`() {
            // Given
            every { mockConfig.timeout } returns null

            // When
            val service = AuraAIService(mockConfig, mockHttpClient, mockLogger)

            // Then
            assertNotNull(service)
            assertEquals(Duration.ofSeconds(30), service.getTimeout())
        }
    }

    @Nested
    @DisplayName("Request Processing Tests")
    inner class RequestProcessingTests {

        @Test
        @DisplayName("Should process valid AI request successfully")
        fun `should process valid AI request successfully`() = runTest {
            // Given
            val request = AIRequest(
                prompt = "Test prompt",
                model = "gpt-4",
                provider = AIProvider.OPENAI,
                temperature = 0.7f,
                maxTokens = 100
            )
            
            val expectedResponse = AIResponse(
                content = "Test response",
                model = "gpt-4",
                usage = AIResponse.Usage(50, 50, 100),
                finishReason = "stop"
            )

            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse>().apply {
                every { status } returns HttpStatus.OK
                every { body } returns """{"content": "Test response", "model": "gpt-4", "usage": {"prompt_tokens": 50, "completion_tokens": 50, "total_tokens": 100}, "finish_reason": "stop"}"""
            }

            // When
            val result = auraAIService.processRequest(request)

            // Then
            assertEquals(expectedResponse.content, result.content)
            assertEquals(expectedResponse.model, result.model)
            assertEquals(expectedResponse.usage.totalTokens, result.usage.totalTokens)
            verify { mockLogger.info(any()) }
        }

        @Test
        @DisplayName("Should handle empty prompt gracefully")
        fun `should handle empty prompt gracefully`() = runTest {
            // Given
            val request = AIRequest(
                prompt = "",
                model = "gpt-4",
                provider = AIProvider.OPENAI
            )

            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.processRequest(request)
            }
        }

        @Test
        @DisplayName("Should handle null prompt gracefully")
        fun `should handle null prompt gracefully`() = runTest {
            // Given
            val request = AIRequest(
                prompt = null,
                model = "gpt-4",
                provider = AIProvider.OPENAI
            )

            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.processRequest(request)
            }
        }

        @ParameterizedTest
        @ValueSource(strings = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "claude-3-opus"])
        @DisplayName("Should handle different model types")
        fun `should handle different model types`(model: String) = runTest {
            // Given
            val request = AIRequest(
                prompt = "Test prompt",
                model = model,
                provider = AIProvider.OPENAI
            )

            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse>().apply {
                every { status } returns HttpStatus.OK
                every { body } returns """{"content": "Response", "model": "$model", "usage": {"total_tokens": 100}, "finish_reason": "stop"}"""
            }

            // When
            val result = auraAIService.processRequest(request)

            // Then
            assertEquals(model, result.model)
            assertNotNull(result.content)
        }

        @ParameterizedTest
        @EnumSource(AIProvider::class)
        @DisplayName("Should handle different AI providers")
        fun `should handle different AI providers`(provider: AIProvider) = runTest {
            // Given
            val request = AIRequest(
                prompt = "Test prompt",
                model = "default-model",
                provider = provider
            )

            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse>().apply {
                every { status } returns HttpStatus.OK
                every { body } returns """{"content": "Response", "model": "default-model", "usage": {"total_tokens": 100}, "finish_reason": "stop"}"""
            }

            // When
            val result = auraAIService.processRequest(request)

            // Then
            assertNotNull(result)
            assertEquals("Response", result.content)
        }

        @ParameterizedTest
        @CsvSource(
            "0.0, 100",
            "0.5, 500",
            "1.0, 1000",
            "1.5, 2000"
        )
        @DisplayName("Should handle different temperature and max token values")
        fun `should handle different temperature and max token values`(temperature: Float, maxTokens: Int) = runTest {
            // Given
            val request = AIRequest(
                prompt = "Test prompt",
                model = "gpt-4",
                provider = AIProvider.OPENAI,
                temperature = temperature,
                maxTokens = maxTokens
            )

            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse>().apply {
                every { status } returns HttpStatus.OK
                every { body } returns """{"content": "Response", "model": "gpt-4", "usage": {"total_tokens": $maxTokens}, "finish_reason": "stop"}"""
            }

            // When
            val result = auraAIService.processRequest(request)

            // Then
            assertNotNull(result)
            assertTrue(result.usage.totalTokens <= maxTokens)
        }
    }

    @Nested
    @DisplayName("Error Handling Tests")
    inner class ErrorHandlingTests {

        @Test
        @DisplayName("Should handle HTTP 401 Unauthorized error")
        fun `should handle HTTP 401 Unauthorized error`() = runTest {
            // Given
            val request = AIRequest("Test prompt", "gpt-4", AIProvider.OPENAI)
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse>().apply {
                every { status } returns HttpStatus.UNAUTHORIZED
                every { body } returns """{"error": {"message": "Invalid API key", "type": "invalid_request_error"}}"""
            }

            // When & Then
            val exception = assertThrows<AIError.AuthenticationError> {
                auraAIService.processRequest(request)
            }
            
            assertEquals("Invalid API key", exception.message)
            verify { mockLogger.error(any(), any()) }
        }

        @Test
        @DisplayName("Should handle HTTP 429 Rate Limit error")
        fun `should handle HTTP 429 Rate Limit error`() = runTest {
            // Given
            val request = AIRequest("Test prompt", "gpt-4", AIProvider.OPENAI)
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse>().apply {
                every { status } returns HttpStatus.TOO_MANY_REQUESTS
                every { body } returns """{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}"""
            }

            // When & Then
            val exception = assertThrows<AIError.RateLimitError> {
                auraAIService.processRequest(request)
            }
            
            assertEquals("Rate limit exceeded", exception.message)
            verify { mockLogger.warn(any(), any()) }
        }

        @Test
        @DisplayName("Should handle HTTP 500 Server error")
        fun `should handle HTTP 500 Server error`() = runTest {
            // Given
            val request = AIRequest("Test prompt", "gpt-4", AIProvider.OPENAI)
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse>().apply {
                every { status } returns HttpStatus.INTERNAL_SERVER_ERROR
                every { body } returns """{"error": {"message": "Internal server error", "type": "server_error"}}"""
            }

            // When & Then
            val exception = assertThrows<AIError.ServerError> {
                auraAIService.processRequest(request)
            }
            
            assertEquals("Internal server error", exception.message)
            verify { mockLogger.error(any(), any()) }
        }

        @Test
        @DisplayName("Should handle network timeout")
        fun `should handle network timeout`() = runTest {
            // Given
            val request = AIRequest("Test prompt", "gpt-4", AIProvider.OPENAI)
            
            coEvery { mockHttpClient.post(any(), any()) } throws TimeoutException("Request timed out")

            // When & Then
            val exception = assertThrows<AIError.TimeoutError> {
                auraAIService.processRequest(request)
            }
            
            assertTrue(exception.message?.contains("Request timed out") == true)
            verify { mockLogger.error(any(), any()) }
        }

        @Test
        @DisplayName("Should handle malformed JSON response")
        fun `should handle malformed JSON response`() = runTest {
            // Given
            val request = AIRequest("Test prompt", "gpt-4", AIProvider.OPENAI)
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse>().apply {
                every { status } returns HttpStatus.OK
                every { body } returns "invalid json response"
            }

            // When & Then
            val exception = assertThrows<AIError.ParseError> {
                auraAIService.processRequest(request)
            }
            
            assertNotNull(exception.message)
            verify { mockLogger.error(any(), any()) }
        }

        @Test
        @DisplayName("Should handle network connectivity issues")
        fun `should handle network connectivity issues`() = runTest {
            // Given
            val request = AIRequest("Test prompt", "gpt-4", AIProvider.OPENAI)
            
            coEvery { mockHttpClient.post(any(), any()) } throws ConnectException("Connection refused")

            // When & Then
            val exception = assertThrows<AIError.NetworkError> {
                auraAIService.processRequest(request)
            }
            
            assertTrue(exception.message?.contains("Connection refused") == true)
            verify { mockLogger.error(any(), any()) }
        }
    }

    @Nested
    @DisplayName("Retry Logic Tests")
    inner class RetryLogicTests {

        @Test
        @DisplayName("Should retry on transient failures")
        fun `should retry on transient failures`() = runTest {
            // Given
            val request = AIRequest("Test prompt", "gpt-4", AIProvider.OPENAI)
            
            coEvery { mockHttpClient.post(any(), any()) } returnsMany listOf(
                mockk<HttpResponse>().apply {
                    every { status } returns HttpStatus.INTERNAL_SERVER_ERROR
                    every { body } returns """{"error": {"message": "Server error", "type": "server_error"}}"""
                },
                mockk<HttpResponse>().apply {
                    every { status } returns HttpStatus.OK
                    every { body } returns """{"content": "Success", "model": "gpt-4", "usage": {"total_tokens": 100}, "finish_reason": "stop"}"""
                }
            )

            // When
            val result = auraAIService.processRequest(request)

            // Then
            assertEquals("Success", result.content)
            coVerify(exactly = 2) { mockHttpClient.post(any(), any()) }
            verify { mockLogger.warn(any(), any()) }
        }

        @Test
        @DisplayName("Should not retry on non-transient failures")
        fun `should not retry on non-transient failures`() = runTest {
            // Given
            val request = AIRequest("Test prompt", "gpt-4", AIProvider.OPENAI)
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse>().apply {
                every { status } returns HttpStatus.UNAUTHORIZED
                every { body } returns """{"error": {"message": "Invalid API key", "type": "invalid_request_error"}}"""
            }

            // When & Then
            assertThrows<AIError.AuthenticationError> {
                auraAIService.processRequest(request)
            }
            
            coVerify(exactly = 1) { mockHttpClient.post(any(), any()) }
        }

        @Test
        @DisplayName("Should exhaust all retry attempts")
        fun `should exhaust all retry attempts`() = runTest {
            // Given
            val request = AIRequest("Test prompt", "gpt-4", AIProvider.OPENAI)
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse>().apply {
                every { status } returns HttpStatus.INTERNAL_SERVER_ERROR
                every { body } returns """{"error": {"message": "Server error", "type": "server_error"}}"""
            }

            // When & Then
            assertThrows<AIError.ServerError> {
                auraAIService.processRequest(request)
            }
            
            coVerify(exactly = 4) { mockHttpClient.post(any(), any()) } // 1 initial + 3 retries
            verify(atLeast = 3) { mockLogger.warn(any(), any()) }
        }
    }

    @Nested
    @DisplayName("Async Processing Tests")
    inner class AsyncProcessingTests {

        @Test
        @DisplayName("Should handle concurrent requests")
        fun `should handle concurrent requests`() = runTest {
            // Given
            val requests = (1..5).map { i ->
                AIRequest("Test prompt $i", "gpt-4", AIProvider.OPENAI)
            }
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse>().apply {
                every { status } returns HttpStatus.OK
                every { body } returns """{"content": "Response", "model": "gpt-4", "usage": {"total_tokens": 100}, "finish_reason": "stop"}"""
            }

            // When
            val results = requests.map { request ->
                async { auraAIService.processRequest(request) }
            }.awaitAll()

            // Then
            assertEquals(5, results.size)
            results.forEach { result ->
                assertNotNull(result)
                assertEquals("Response", result.content)
            }
            coVerify(exactly = 5) { mockHttpClient.post(any(), any()) }
        }

        @Test
        @DisplayName("Should handle request cancellation")
        fun `should handle request cancellation`() = runTest {
            // Given
            val request = AIRequest("Test prompt", "gpt-4", AIProvider.OPENAI)
            
            coEvery { mockHttpClient.post(any(), any()) } coAnswers {
                delay(1000)
                mockk<HttpResponse>().apply {
                    every { status } returns HttpStatus.OK
                    every { body } returns """{"content": "Response", "model": "gpt-4", "usage": {"total_tokens": 100}, "finish_reason": "stop"}"""
                }
            }

            // When
            val job = launch {
                auraAIService.processRequest(request)
            }
            
            delay(100)
            job.cancel()

            // Then
            assertTrue(job.isCancelled)
            verify { mockLogger.info(match { it.contains("cancelled") }) }
        }
    }

    @Nested
    @DisplayName("Configuration Tests")
    inner class ConfigurationTests {

        @Test
        @DisplayName("Should respect custom timeout configuration")
        fun `should respect custom timeout configuration`() = runTest {
            // Given
            val customTimeout = Duration.ofSeconds(5)
            every { mockConfig.timeout } returns customTimeout
            
            val service = AuraAIService(mockConfig, mockHttpClient, mockLogger)
            val request = AIRequest("Test prompt", "gpt-4", AIProvider.OPENAI)

            // When
            val timeout = service.getTimeout()

            // Then
            assertEquals(customTimeout, timeout)
        }

        @Test
        @DisplayName("Should use default model when not specified")
        fun `should use default model when not specified`() = runTest {
            // Given
            val request = AIRequest(
                prompt = "Test prompt",
                model = null,
                provider = AIProvider.OPENAI
            )
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse>().apply {
                every { status } returns HttpStatus.OK
                every { body } returns """{"content": "Response", "model": "gpt-4", "usage": {"total_tokens": 100}, "finish_reason": "stop"}"""
            }

            // When
            val result = auraAIService.processRequest(request)

            // Then
            assertEquals("gpt-4", result.model)
        }

        @Test
        @DisplayName("Should validate temperature bounds")
        fun `should validate temperature bounds`() = runTest {
            // Given
            val invalidRequest = AIRequest(
                prompt = "Test prompt",
                model = "gpt-4",
                provider = AIProvider.OPENAI,
                temperature = 2.5f // Invalid temperature > 2.0
            )

            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.processRequest(invalidRequest)
            }
        }

        @Test
        @DisplayName("Should validate max tokens bounds")
        fun `should validate max tokens bounds`() = runTest {
            // Given
            val invalidRequest = AIRequest(
                prompt = "Test prompt",
                model = "gpt-4",
                provider = AIProvider.OPENAI,
                maxTokens = -1 // Invalid negative max tokens
            )

            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.processRequest(invalidRequest)
            }
        }
    }

    @Nested
    @DisplayName("Integration Tests")
    inner class IntegrationTests {

        @Test
        @DisplayName("Should handle real-world scenario with streaming")
        fun `should handle real-world scenario with streaming`() = runTest {
            // Given
            val request = AIRequest(
                prompt = "Write a short story about AI",
                model = "gpt-4",
                provider = AIProvider.OPENAI,
                stream = true
            )
            
            val streamResponses = listOf(
                """{"delta": {"content": "Once"}, "finish_reason": null}""",
                """{"delta": {"content": " upon"}, "finish_reason": null}""",
                """{"delta": {"content": " a time"}, "finish_reason": "stop"}"""
            )

            coEvery { mockHttpClient.postStream(any(), any()) } returns streamResponses.asFlow()

            // When
            val result = auraAIService.processStreamingRequest(request)

            // Then
            val content = result.toList().joinToString("")
            assertEquals("Once upon a time", content)
            verify { mockLogger.info(match { it.contains("streaming") }) }
        }

        @Test
        @DisplayName("Should handle complex prompt with system messages")
        fun `should handle complex prompt with system messages`() = runTest {
            // Given
            val request = AIRequest(
                prompt = "User: Hello",
                model = "gpt-4",
                provider = AIProvider.OPENAI,
                systemMessage = "You are a helpful assistant."
            )
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse>().apply {
                every { status } returns HttpStatus.OK
                every { body } returns """{"content": "Hello! How can I help you today?", "model": "gpt-4", "usage": {"total_tokens": 50}, "finish_reason": "stop"}"""
            }

            // When
            val result = auraAIService.processRequest(request)

            // Then
            assertEquals("Hello! How can I help you today?", result.content)
            assertNotNull(result.usage)
            assertEquals(50, result.usage.totalTokens)
        }
    }

    @Nested
    @DisplayName("Performance Tests")
    inner class PerformanceTests {

        @Test
        @DisplayName("Should handle large prompt efficiently")
        fun `should handle large prompt efficiently`() = runTest {
            // Given
            val largePrompt = "x".repeat(10000)
            val request = AIRequest(largePrompt, "gpt-4", AIProvider.OPENAI)
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse>().apply {
                every { status } returns HttpStatus.OK
                every { body } returns """{"content": "Response", "model": "gpt-4", "usage": {"total_tokens": 5000}, "finish_reason": "stop"}"""
            }

            // When
            val startTime = System.currentTimeMillis()
            val result = auraAIService.processRequest(request)
            val endTime = System.currentTimeMillis()

            // Then
            assertNotNull(result)
            assertTrue(endTime - startTime < 5000) // Should complete within 5 seconds
        }

        @Test
        @DisplayName("Should maintain performance under load")
        fun `should maintain performance under load`() = runTest {
            // Given
            val requests = (1..10).map { i ->
                AIRequest("Prompt $i", "gpt-4", AIProvider.OPENAI)
            }
            
            coEvery { mockHttpClient.post(any(), any()) } returns mockk<HttpResponse>().apply {
                every { status } returns HttpStatus.OK
                every { body } returns """{"content": "Response", "model": "gpt-4", "usage": {"total_tokens": 100}, "finish_reason": "stop"}"""
            }

            // When
            val startTime = System.currentTimeMillis()
            val results = requests.map { request ->
                async { auraAIService.processRequest(request) }
            }.awaitAll()
            val endTime = System.currentTimeMillis()

            // Then
            assertEquals(10, results.size)
            assertTrue(endTime - startTime < 10000) // Should complete within 10 seconds
            results.forEach { result ->
                assertNotNull(result)
            }
        }
    }
}