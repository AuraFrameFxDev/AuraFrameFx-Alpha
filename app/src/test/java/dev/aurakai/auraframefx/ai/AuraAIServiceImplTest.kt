package dev.aurakai.auraframefx.ai

import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.extension.ExtendWith
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.junit.jupiter.MockitoExtension
import org.mockito.kotlin.*
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import kotlin.time.Duration.Companion.seconds

@ExtendWith(MockitoExtension::class)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class AuraAIServiceImplTest {

    @Mock
    private lateinit var mockHttpClient: HttpClient

    @Mock
    private lateinit var mockConfiguration: AuraAIConfiguration

    @Mock
    private lateinit var mockTokenManager: TokenManager

    @Mock
    private lateinit var mockRateLimiter: RateLimiter

    private lateinit var auraAIService: AuraAIServiceImpl
    private lateinit var closeable: AutoCloseable

    @BeforeEach
    fun setUp() {
        closeable = MockitoAnnotations.openMocks(this)

        // Setup default mock behaviors
        whenever(mockConfiguration.apiKey).thenReturn("test-api-key")
        whenever(mockConfiguration.baseUrl).thenReturn("https://api.aurai.test")
        whenever(mockConfiguration.timeout).thenReturn(30.seconds)
        whenever(mockConfiguration.maxRetries).thenReturn(3)
        whenever(mockRateLimiter.tryAcquire()).thenReturn(true)
        whenever(mockTokenManager.getValidToken()).thenReturn("valid-token")

        auraAIService = AuraAIServiceImpl(
            httpClient = mockHttpClient,
            configuration = mockConfiguration,
            tokenManager = mockTokenManager,
            rateLimiter = mockRateLimiter
        )
    }

    @AfterEach
    fun tearDown() {
        closeable.close()
    }

    @Nested
    @DisplayName("Initialization Tests")
    inner class InitializationTests {

        @Test
        @DisplayName("Should initialize with valid configuration")
        fun shouldInitializeWithValidConfiguration() {
            assertNotNull(auraAIService)
            verify(mockConfiguration).apiKey
            verify(mockConfiguration).baseUrl
        }

        @Test
        @DisplayName("Should throw exception with null configuration")
        fun shouldThrowExceptionWithNullConfiguration() {
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(
                    httpClient = mockHttpClient,
                    configuration = null,
                    tokenManager = mockTokenManager,
                    rateLimiter = mockRateLimiter
                )
            }
        }

        @Test
        @DisplayName("Should throw exception with invalid API key")
        fun shouldThrowExceptionWithInvalidApiKey() {
            whenever(mockConfiguration.apiKey).thenReturn("")

            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(
                    httpClient = mockHttpClient,
                    configuration = mockConfiguration,
                    tokenManager = mockTokenManager,
                    rateLimiter = mockRateLimiter
                )
            }
        }
    }

    @Nested
    @DisplayName("Generate Text Tests")
    inner class GenerateTextTests {

        @Test
        @DisplayName("Should generate text successfully with valid input")
        fun shouldGenerateTextSuccessfully() = runTest {
            val prompt = "Write a hello world program"
            val expectedResponse = "println(\"Hello, World!\")"
            val mockResponse = AIResponse(
                text = expectedResponse,
                usage = TokenUsage(promptTokens = 10, completionTokens = 15, totalTokens = 25),
                model = "gpt-3.5-turbo"
            )

            whenever(mockHttpClient.post(any(), any())).thenReturn(mockResponse)

            val result = auraAIService.generateText(prompt)

            assertEquals(expectedResponse, result.text)
            verify(mockHttpClient).post(any(), any())
            verify(mockRateLimiter).tryAcquire()
        }

        @Test
        @DisplayName("Should handle empty prompt")
        fun shouldHandleEmptyPrompt() = runTest {
            assertThrows<IllegalArgumentException> {
                auraAIService.generateText("")
            }
        }

        @Test
        @DisplayName("Should handle null prompt")
        fun shouldHandleNullPrompt() = runTest {
            assertThrows<IllegalArgumentException> {
                auraAIService.generateText(null)
            }
        }

        @Test
        @DisplayName("Should handle very long prompt")
        fun shouldHandleVeryLongPrompt() = runTest {
            val longPrompt = "A".repeat(100000)

            whenever(mockHttpClient.post(any(), any())).thenThrow(
                AIException("Prompt too long", AIErrorCode.PROMPT_TOO_LONG)
            )

            assertThrows<AIException> {
                auraAIService.generateText(longPrompt)
            }
        }

        @Test
        @DisplayName("Should handle rate limiting")
        fun shouldHandleRateLimiting() = runTest {
            whenever(mockRateLimiter.tryAcquire()).thenReturn(false)

            assertThrows<RateLimitExceededException> {
                auraAIService.generateText("test prompt")
            }
        }

        @Test
        @DisplayName("Should retry on transient failures")
        fun shouldRetryOnTransientFailures() = runTest {
            val prompt = "test prompt"
            val mockResponse = AIResponse(
                text = "response",
                usage = TokenUsage(5, 10, 15),
                model = "gpt-3.5-turbo"
            )

            whenever(mockHttpClient.post(any(), any()))
                .thenThrow(TransientException("Network error"))
                .thenThrow(TransientException("Server error"))
                .thenReturn(mockResponse)

            val result = auraAIService.generateText(prompt)

            assertEquals("response", result.text)
            verify(mockHttpClient, times(3)).post(any(), any())
        }

        @Test
        @DisplayName("Should fail after max retries")
        fun shouldFailAfterMaxRetries() = runTest {
            whenever(mockHttpClient.post(any(), any()))
                .thenThrow(TransientException("Persistent error"))

            assertThrows<AIException> {
                auraAIService.generateText("test prompt")
            }

            verify(mockHttpClient, times(4)).post(any(), any()) // initial + 3 retries
        }
    }

    @Nested
    @DisplayName("Generate Text with Parameters Tests")
    inner class GenerateTextWithParametersTests {

        @Test
        @DisplayName("Should generate text with custom parameters")
        fun shouldGenerateTextWithCustomParameters() = runTest {
            val prompt = "Generate code"
            val parameters = AIParameters(
                temperature = 0.7f,
                maxTokens = 1000,
                topP = 0.9f,
                presencePenalty = 0.1f,
                frequencyPenalty = 0.2f
            )

            val mockResponse = AIResponse(
                text = "Generated code here",
                usage = TokenUsage(20, 30, 50),
                model = "gpt-4"
            )

            whenever(mockHttpClient.post(any(), any())).thenReturn(mockResponse)

            val result = auraAIService.generateText(prompt, parameters)

            assertEquals("Generated code here", result.text)
            verify(mockHttpClient).post(any(), any())
        }

        @Test
        @DisplayName("Should validate temperature parameter")
        fun shouldValidateTemperatureParameter() = runTest {
            val parameters = AIParameters(temperature = 2.5f) // Invalid temperature

            assertThrows<IllegalArgumentException> {
                auraAIService.generateText("test", parameters)
            }
        }

        @Test
        @DisplayName("Should validate max tokens parameter")
        fun shouldValidateMaxTokensParameter() = runTest {
            val parameters = AIParameters(maxTokens = -1) // Invalid max tokens

            assertThrows<IllegalArgumentException> {
                auraAIService.generateText("test", parameters)
            }
        }

        @Test
        @DisplayName("Should validate top-p parameter")
        fun shouldValidateTopPParameter() = runTest {
            val parameters = AIParameters(topP = 1.5f) // Invalid top-p

            assertThrows<IllegalArgumentException> {
                auraAIService.generateText("test", parameters)
            }
        }
    }

    @Nested
    @DisplayName("Async Operations Tests")
    inner class AsyncOperationsTests {

        @Test
        @DisplayName("Should handle async text generation")
        fun shouldHandleAsyncTextGeneration() = runTest {
            val prompt = "Async test"
            val mockResponse = AIResponse(
                text = "Async response",
                usage = TokenUsage(5, 10, 15),
                model = "gpt-3.5-turbo"
            )

            whenever(mockHttpClient.post(any(), any())).thenReturn(mockResponse)

            val future = auraAIService.generateTextAsync(prompt)
            val result = future.get(5, TimeUnit.SECONDS)

            assertEquals("Async response", result.text)
            assertTrue(future.isDone)
            assertFalse(future.isCancelled)
        }

        @Test
        @DisplayName("Should handle async operation timeout")
        fun shouldHandleAsyncOperationTimeout() = runTest {
            whenever(mockHttpClient.post(any(), any())).thenAnswer {
                Thread.sleep(10000) // Simulate slow response
                AIResponse("", TokenUsage(0, 0, 0), "")
            }

            val future = auraAIService.generateTextAsync("test")

            assertThrows<TimeoutException> {
                future.get(1, TimeUnit.SECONDS)
            }
        }

        @Test
        @DisplayName("Should handle async operation cancellation")
        fun shouldHandleAsyncOperationCancellation() = runTest {
            whenever(mockHttpClient.post(any(), any())).thenAnswer {
                Thread.sleep(5000) // Simulate slow response
                AIResponse("", TokenUsage(0, 0, 0), "")
            }

            val future = auraAIService.generateTextAsync("test")
            future.cancel(true)

            assertTrue(future.isCancelled)
        }
    }

    @Nested
    @DisplayName("Token Management Tests")
    inner class TokenManagementTests {

        @Test
        @DisplayName("Should refresh token when expired")
        fun shouldRefreshTokenWhenExpired() = runTest {
            whenever(mockTokenManager.getValidToken())
                .thenReturn("expired-token")
                .thenReturn("new-token")

            whenever(mockHttpClient.post(any(), any()))
                .thenThrow(UnauthorizedException("Token expired"))
                .thenReturn(AIResponse("success", TokenUsage(5, 10, 15), "gpt-3.5-turbo"))

            val result = auraAIService.generateText("test")

            assertEquals("success", result.text)
            verify(mockTokenManager, times(2)).getValidToken()
        }

        @Test
        @DisplayName("Should handle token refresh failure")
        fun shouldHandleTokenRefreshFailure() = runTest {
            whenever(mockTokenManager.getValidToken())
                .thenThrow(TokenRefreshException("Cannot refresh token"))

            assertThrows<AuthenticationException> {
                auraAIService.generateText("test")
            }
        }
    }

    @Nested
    @DisplayName("Error Handling Tests")
    inner class ErrorHandlingTests {

        @Test
        @DisplayName("Should handle API quota exceeded")
        fun shouldHandleApiQuotaExceeded() = runTest {
            whenever(mockHttpClient.post(any(), any()))
                .thenThrow(QuotaExceededException("API quota exceeded"))

            assertThrows<QuotaExceededException> {
                auraAIService.generateText("test")
            }
        }

        @Test
        @DisplayName("Should handle server errors")
        fun shouldHandleServerErrors() = runTest {
            whenever(mockHttpClient.post(any(), any()))
                .thenThrow(ServerException("Internal server error", 500))

            assertThrows<AIException> {
                auraAIService.generateText("test")
            }
        }

        @Test
        @DisplayName("Should handle network connectivity issues")
        fun shouldHandleNetworkConnectivityIssues() = runTest {
            whenever(mockHttpClient.post(any(), any()))
                .thenThrow(NetworkException("Connection timeout"))

            assertThrows<AIException> {
                auraAIService.generateText("test")
            }
        }

        @Test
        @DisplayName("Should handle malformed responses")
        fun shouldHandleMalformedResponses() = runTest {
            whenever(mockHttpClient.post(any(), any()))
                .thenThrow(JsonParseException("Invalid JSON response"))

            assertThrows<AIException> {
                auraAIService.generateText("test")
            }
        }
    }

    @Nested
    @DisplayName("Configuration Tests")
    inner class ConfigurationTests {

        @Test
        @DisplayName("Should respect timeout configuration")
        fun shouldRespectTimeoutConfiguration() = runTest {
            whenever(mockConfiguration.timeout).thenReturn(1.seconds)

            whenever(mockHttpClient.post(any(), any())).thenAnswer {
                Thread.sleep(2000) // Simulate slow response
                AIResponse("", TokenUsage(0, 0, 0), "")
            }

            assertThrows<TimeoutException> {
                auraAIService.generateText("test")
            }
        }

        @Test
        @DisplayName("Should use configured base URL")
        fun shouldUseConfiguredBaseUrl() = runTest {
            val customUrl = "https://custom.api.url"
            whenever(mockConfiguration.baseUrl).thenReturn(customUrl)

            whenever(mockHttpClient.post(any(), any())).thenReturn(
                AIResponse("response", TokenUsage(5, 10, 15), "gpt-3.5-turbo")
            )

            auraAIService.generateText("test")

            verify(mockHttpClient).post(contains(customUrl), any())
        }

        @Test
        @DisplayName("Should handle configuration updates")
        fun shouldHandleConfigurationUpdates() = runTest {
            val newConfig = mockConfiguration.copy(
                apiKey = "new-api-key",
                baseUrl = "https://new.api.url"
            )

            auraAIService.updateConfiguration(newConfig)

            whenever(mockHttpClient.post(any(), any())).thenReturn(
                AIResponse("response", TokenUsage(5, 10, 15), "gpt-3.5-turbo")
            )

            auraAIService.generateText("test")

            verify(mockHttpClient).post(contains("https://new.api.url"), any())
        }
    }

    @Nested
    @DisplayName("Resource Management Tests")
    inner class ResourceManagementTests {

        @Test
        @DisplayName("Should cleanup resources on shutdown")
        fun shouldCleanupResourcesOnShutdown() = runTest {
            auraAIService.shutdown()

            verify(mockHttpClient).close()
            verify(mockTokenManager).cleanup()
            verify(mockRateLimiter).shutdown()
        }

        @Test
        @DisplayName("Should handle concurrent requests")
        fun shouldHandleConcurrentRequests() = runTest {
            val mockResponse = AIResponse(
                text = "concurrent response",
                usage = TokenUsage(5, 10, 15),
                model = "gpt-3.5-turbo"
            )

            whenever(mockHttpClient.post(any(), any())).thenReturn(mockResponse)

            val futures = (1..10).map { i ->
                auraAIService.generateTextAsync("test $i")
            }

            val results = futures.map { it.get(10, TimeUnit.SECONDS) }

            assertEquals(10, results.size)
            results.forEach { result ->
                assertEquals("concurrent response", result.text)
            }
        }
    }

    @Nested
    @DisplayName("Edge Cases Tests")
    inner class EdgeCasesTests {

        @Test
        @DisplayName("Should handle unicode characters in prompt")
        fun shouldHandleUnicodeCharactersInPrompt() = runTest {
            val unicodePrompt = "Generate code with emojis 🚀🎯💻"
            val mockResponse = AIResponse(
                text = "// Code with emojis ✨",
                usage = TokenUsage(10, 15, 25),
                model = "gpt-3.5-turbo"
            )

            whenever(mockHttpClient.post(any(), any())).thenReturn(mockResponse)

            val result = auraAIService.generateText(unicodePrompt)

            assertEquals("// Code with emojis ✨", result.text)
        }

        @Test
        @DisplayName("Should handle special characters in prompt")
        fun shouldHandleSpecialCharactersInPrompt() = runTest {
            val specialPrompt = "Generate code with special chars: \n\t\r\"'\\/"
            val mockResponse = AIResponse(
                text = "Code with special handling",
                usage = TokenUsage(15, 20, 35),
                model = "gpt-3.5-turbo"
            )

            whenever(mockHttpClient.post(any(), any())).thenReturn(mockResponse)

            val result = auraAIService.generateText(specialPrompt)

            assertEquals("Code with special handling", result.text)
        }

        @Test
        @DisplayName("Should handle very large response")
        fun shouldHandleVeryLargeResponse() = runTest {
            val largeResponse = "A".repeat(50000)
            val mockResponse = AIResponse(
                text = largeResponse,
                usage = TokenUsage(100, 12500, 12600),
                model = "gpt-4"
            )

            whenever(mockHttpClient.post(any(), any())).thenReturn(mockResponse)

            val result = auraAIService.generateText("Generate large text")

            assertEquals(largeResponse, result.text)
            assertEquals(50000, result.text.length)
        }
    }

    @Nested
    @DisplayName("Performance Tests")
    inner class PerformanceTests {

        @Test
        @DisplayName("Should complete request within reasonable time")
        fun shouldCompleteRequestWithinReasonableTime() = runTest {
            val mockResponse = AIResponse(
                text = "Fast response",
                usage = TokenUsage(5, 10, 15),
                model = "gpt-3.5-turbo"
            )

            whenever(mockHttpClient.post(any(), any())).thenReturn(mockResponse)

            val startTime = System.currentTimeMillis()
            val result = auraAIService.generateText("Quick test")
            val endTime = System.currentTimeMillis()

            assertEquals("Fast response", result.text)
            assertTrue(endTime - startTime < 5000) // Should complete within 5 seconds
        }

        @Test
        @DisplayName("Should handle multiple sequential requests efficiently")
        fun shouldHandleMultipleSequentialRequestsEfficiently() = runTest {
            val mockResponse = AIResponse(
                text = "Sequential response",
                usage = TokenUsage(5, 10, 15),
                model = "gpt-3.5-turbo"
            )

            whenever(mockHttpClient.post(any(), any())).thenReturn(mockResponse)

            val startTime = System.currentTimeMillis()

            repeat(5) { i ->
                val result = auraAIService.generateText("Sequential test $i")
                assertEquals("Sequential response", result.text)
            }

            val endTime = System.currentTimeMillis()
            assertTrue(endTime - startTime < 10000) // Should complete within 10 seconds
        }
    }
}