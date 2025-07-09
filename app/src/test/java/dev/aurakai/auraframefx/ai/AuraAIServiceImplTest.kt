package dev.aurakai.auraframefx.ai

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.extension.ExtendWith
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.junit.jupiter.MockitoExtension
import org.mockito.kotlin.*
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.runBlocking
import java.util.concurrent.CompletableFuture
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
    @Nested
    @DisplayName("Streaming Operations Tests")
    inner class StreamingOperationsTests {

        @Test
        @DisplayName("Should handle streaming text generation")
        fun shouldHandleStreamingTextGeneration() = runTest {
            val prompt = "Generate streaming response"
            val mockStreamResponse = StreamingResponse(
                chunks = listOf("Hello", " World", "!"),
                usage = TokenUsage(10, 15, 25),
                model = "gpt-3.5-turbo"
            )

            whenever(mockHttpClient.postStream(any(), any())).thenReturn(mockStreamResponse)

            val result = auraAIService.generateTextStreaming(prompt)
            val chunks = mutableListOf<String>()
            result.collect { chunk ->
                chunks.add(chunk)
            }

            assertEquals(listOf("Hello", " World", "!"), chunks)
            verify(mockHttpClient).postStream(any(), any())
        }

        @Test
        @DisplayName("Should handle streaming interruption")
        fun shouldHandleStreamingInterruption() = runTest {
            whenever(mockHttpClient.postStream(any(), any())).thenThrow(
                StreamingInterruptedException("Stream interrupted")
            )

            assertThrows<StreamingInterruptedException> {
                auraAIService.generateTextStreaming("test").collect()
            }
        }

        @Test
        @DisplayName("Should handle empty streaming response")
        fun shouldHandleEmptyStreamingResponse() = runTest {
            val mockStreamResponse = StreamingResponse(
                chunks = emptyList(),
                usage = TokenUsage(5, 0, 5),
                model = "gpt-3.5-turbo"
            )

            whenever(mockHttpClient.postStream(any(), any())).thenReturn(mockStreamResponse)

            val result = auraAIService.generateTextStreaming("test")
            val chunks = mutableListOf<String>()
            result.collect { chunk ->
                chunks.add(chunk)
            }

            assertTrue(chunks.isEmpty())
        }
    }

    @Nested
    @DisplayName("Batch Operations Tests")
    inner class BatchOperationsTests {

        @Test
        @DisplayName("Should handle batch text generation")
        fun shouldHandleBatchTextGeneration() = runTest {
            val prompts = listOf("Prompt 1", "Prompt 2", "Prompt 3")
            val mockBatchResponse = BatchResponse(
                responses = listOf(
                    AIResponse("Response 1", TokenUsage(5, 10, 15), "gpt-3.5-turbo"),
                    AIResponse("Response 2", TokenUsage(6, 11, 17), "gpt-3.5-turbo"),
                    AIResponse("Response 3", TokenUsage(7, 12, 19), "gpt-3.5-turbo")
                ),
                totalUsage = TokenUsage(18, 33, 51)
            )

            whenever(mockHttpClient.postBatch(any(), any())).thenReturn(mockBatchResponse)

            val result = auraAIService.generateTextBatch(prompts)

            assertEquals(3, result.responses.size)
            assertEquals("Response 1", result.responses[0].text)
            assertEquals("Response 2", result.responses[1].text)
            assertEquals("Response 3", result.responses[2].text)
            verify(mockHttpClient).postBatch(any(), any())
        }

        @Test
        @DisplayName("Should handle partial batch failures")
        fun shouldHandlePartialBatchFailures() = runTest {
            val prompts = listOf("Valid prompt", "Invalid prompt")
            val mockBatchResponse = BatchResponse(
                responses = listOf(
                    AIResponse("Valid response", TokenUsage(5, 10, 15), "gpt-3.5-turbo"),
                    AIResponse.error("Invalid request", AIErrorCode.INVALID_REQUEST)
                ),
                totalUsage = TokenUsage(5, 10, 15)
            )

            whenever(mockHttpClient.postBatch(any(), any())).thenReturn(mockBatchResponse)

            val result = auraAIService.generateTextBatch(prompts)

            assertEquals(2, result.responses.size)
            assertFalse(result.responses[0].isError)
            assertTrue(result.responses[1].isError)
        }

        @Test
        @DisplayName("Should handle empty batch request")
        fun shouldHandleEmptyBatchRequest() = runTest {
            assertThrows<IllegalArgumentException> {
                auraAIService.generateTextBatch(emptyList())
            }
        }

        @Test
        @DisplayName("Should handle batch size limits")
        fun shouldHandleBatchSizeLimits() = runTest {
            val largeBatch = (1..1000).map { "Prompt $it" }

            whenever(mockHttpClient.postBatch(any(), any())).thenThrow(
                BatchSizeLimitExceededException("Batch size exceeds limit")
            )

            assertThrows<BatchSizeLimitExceededException> {
                auraAIService.generateTextBatch(largeBatch)
            }
        }
    }

    @Nested
    @DisplayName("Model Management Tests")
    inner class ModelManagementTests {

        @Test
        @DisplayName("Should list available models")
        fun shouldListAvailableModels() = runTest {
            val mockModels = listOf(
                ModelInfo("gpt-3.5-turbo", "OpenAI GPT-3.5 Turbo", 4096, true),
                ModelInfo("gpt-4", "OpenAI GPT-4", 8192, true),
                ModelInfo("claude-v1", "Anthropic Claude", 9000, false)
            )

            whenever(mockHttpClient.getModels()).thenReturn(mockModels)

            val result = auraAIService.getAvailableModels()

            assertEquals(3, result.size)
            assertTrue(result.any { it.name == "gpt-3.5-turbo" })
            assertTrue(result.any { it.name == "gpt-4" })
            assertTrue(result.any { it.name == "claude-v1" })
        }

        @Test
        @DisplayName("Should get model capabilities")
        fun shouldGetModelCapabilities() = runTest {
            val modelName = "gpt-4"
            val mockCapabilities = ModelCapabilities(
                supportStreaming = true,
                supportBatch = true,
                maxTokens = 8192,
                supportedFeatures = setOf("text-generation", "code-completion", "chat")
            )

            whenever(mockHttpClient.getModelCapabilities(modelName)).thenReturn(mockCapabilities)

            val result = auraAIService.getModelCapabilities(modelName)

            assertTrue(result.supportStreaming)
            assertTrue(result.supportBatch)
            assertEquals(8192, result.maxTokens)
            assertTrue(result.supportedFeatures.contains("text-generation"))
        }

        @Test
        @DisplayName("Should handle model not found")
        fun shouldHandleModelNotFound() = runTest {
            whenever(mockHttpClient.getModelCapabilities("nonexistent-model"))
                .thenThrow(ModelNotFoundException("Model not found"))

            assertThrows<ModelNotFoundException> {
                auraAIService.getModelCapabilities("nonexistent-model")
            }
        }
    }

    @Nested
    @DisplayName("Advanced Parameter Validation Tests")
    inner class AdvancedParameterValidationTests {

        @Test
        @DisplayName("Should validate presence penalty bounds")
        fun shouldValidatePresencePenaltyBounds() = runTest {
            val parameters = AIParameters(presencePenalty = 3.0f) // Out of bounds

            assertThrows<IllegalArgumentException> {
                auraAIService.generateText("test", parameters)
            }
        }

        @Test
        @DisplayName("Should validate frequency penalty bounds")
        fun shouldValidateFrequencyPenaltyBounds() = runTest {
            val parameters = AIParameters(frequencyPenalty = -3.0f) // Out of bounds

            assertThrows<IllegalArgumentException> {
                auraAIService.generateText("test", parameters)
            }
        }

        @Test
        @DisplayName("Should validate stop sequences")
        fun shouldValidateStopSequences() = runTest {
            val parameters = AIParameters(stopSequences = (1..100).map { "stop$it" }) // Too many

            assertThrows<IllegalArgumentException> {
                auraAIService.generateText("test", parameters)
            }
        }

        @Test
        @DisplayName("Should validate logit bias parameters")
        fun shouldValidateLogitBiasParameters() = runTest {
            val parameters = AIParameters(
                logitBias = mapOf(
                    "token1" to 150.0f, // Out of bounds
                    "token2" to -150.0f  // Out of bounds
                )
            )

            assertThrows<IllegalArgumentException> {
                auraAIService.generateText("test", parameters)
            }
        }

        @Test
        @DisplayName("Should validate model selection")
        fun shouldValidateModelSelection() = runTest {
            val parameters = AIParameters(model = "invalid-model-name")

            whenever(mockHttpClient.post(any(), any())).thenThrow(
                ModelNotFoundException("Invalid model")
            )

            assertThrows<ModelNotFoundException> {
                auraAIService.generateText("test", parameters)
            }
        }
    }

    @Nested
    @DisplayName("Health Check and Monitoring Tests")
    inner class HealthCheckTests {

        @Test
        @DisplayName("Should perform health check")
        fun shouldPerformHealthCheck() = runTest {
            val mockHealthStatus = HealthStatus(
                isHealthy = true,
                responseTime = 150L,
                apiVersion = "v1.0.0",
                availableModels = 5
            )

            whenever(mockHttpClient.healthCheck()).thenReturn(mockHealthStatus)

            val result = auraAIService.healthCheck()

            assertTrue(result.isHealthy)
            assertEquals(150L, result.responseTime)
            assertEquals("v1.0.0", result.apiVersion)
            assertEquals(5, result.availableModels)
        }

        @Test
        @DisplayName("Should handle unhealthy service")
        fun shouldHandleUnhealthyService() = runTest {
            val mockHealthStatus = HealthStatus(
                isHealthy = false,
                responseTime = 5000L,
                apiVersion = "v1.0.0",
                availableModels = 0,
                errorMessage = "Service degraded"
            )

            whenever(mockHttpClient.healthCheck()).thenReturn(mockHealthStatus)

            val result = auraAIService.healthCheck()

            assertFalse(result.isHealthy)
            assertEquals("Service degraded", result.errorMessage)
        }

        @Test
        @DisplayName("Should get usage statistics")
        fun shouldGetUsageStatistics() = runTest {
            val mockUsageStats = UsageStatistics(
                totalRequests = 1000L,
                totalTokens = 50000L,
                averageResponseTime = 200L,
                errorRate = 0.01f,
                quotaUsed = 0.75f
            )

            whenever(mockHttpClient.getUsageStats()).thenReturn(mockUsageStats)

            val result = auraAIService.getUsageStatistics()

            assertEquals(1000L, result.totalRequests)
            assertEquals(50000L, result.totalTokens)
            assertEquals(200L, result.averageResponseTime)
            assertEquals(0.01f, result.errorRate)
            assertEquals(0.75f, result.quotaUsed)
        }
    }

    @Nested
    @DisplayName("Security and Authentication Tests")
    inner class SecurityTests {

        @Test
        @DisplayName("Should handle API key rotation")
        fun shouldHandleApiKeyRotation() = runTest {
            val newApiKey = "new-api-key-123"
            
            auraAIService.rotateApiKey(newApiKey)

            whenever(mockHttpClient.post(any(), any())).thenReturn(
                AIResponse("success", TokenUsage(5, 10, 15), "gpt-3.5-turbo")
            )

            auraAIService.generateText("test")

            verify(mockConfiguration).updateApiKey(newApiKey)
        }

        @Test
        @DisplayName("Should handle invalid API key format")
        fun shouldHandleInvalidApiKeyFormat() = runTest {
            val invalidApiKey = "invalid-key"

            assertThrows<IllegalArgumentException> {
                auraAIService.rotateApiKey(invalidApiKey)
            }
        }

        @Test
        @DisplayName("Should handle unauthorized access")
        fun shouldHandleUnauthorizedAccess() = runTest {
            whenever(mockHttpClient.post(any(), any())).thenThrow(
                UnauthorizedException("Invalid API key")
            )

            assertThrows<UnauthorizedException> {
                auraAIService.generateText("test")
            }
        }

        @Test
        @DisplayName("Should handle forbidden operations")
        fun shouldHandleForbiddenOperations() = runTest {
            whenever(mockHttpClient.post(any(), any())).thenThrow(
                ForbiddenException("Operation not allowed")
            )

            assertThrows<ForbiddenException> {
                auraAIService.generateText("test")
            }
        }
    }

    @Nested
    @DisplayName("Content Filtering Tests")
    inner class ContentFilteringTests {

        @Test
        @DisplayName("Should handle content policy violations")
        fun shouldHandleContentPolicyViolations() = runTest {
            whenever(mockHttpClient.post(any(), any())).thenThrow(
                ContentPolicyViolationException("Content violates policy")
            )

            assertThrows<ContentPolicyViolationException> {
                auraAIService.generateText("inappropriate content")
            }
        }

        @Test
        @DisplayName("Should handle content filtering in responses")
        fun shouldHandleContentFilteringInResponses() = runTest {
            val mockResponse = AIResponse(
                text = "[FILTERED]",
                usage = TokenUsage(10, 0, 10),
                model = "gpt-3.5-turbo",
                contentFiltered = true,
                filterReason = "inappropriate_content"
            )

            whenever(mockHttpClient.post(any(), any())).thenReturn(mockResponse)

            val result = auraAIService.generateText("test")

            assertTrue(result.contentFiltered)
            assertEquals("inappropriate_content", result.filterReason)
            assertEquals("[FILTERED]", result.text)
        }

        @Test
        @DisplayName("Should handle safety settings")
        fun shouldHandleSafetySettings() = runTest {
            val parameters = AIParameters(
                safetySettings = SafetySettings(
                    blockHate = true,
                    blockViolence = true,
                    blockSexual = true,
                    blockDangerous = true
                )
            )

            val mockResponse = AIResponse(
                text = "Safe response",
                usage = TokenUsage(5, 10, 15),
                model = "gpt-3.5-turbo"
            )

            whenever(mockHttpClient.post(any(), any())).thenReturn(mockResponse)

            val result = auraAIService.generateText("test", parameters)

            assertEquals("Safe response", result.text)
            assertFalse(result.contentFiltered)
        }
    }

    @Nested
    @DisplayName("Memory and Resource Management Tests")
    inner class MemoryManagementTests {

        @Test
        @DisplayName("Should handle memory pressure")
        fun shouldHandleMemoryPressure() = runTest {
            whenever(mockHttpClient.post(any(), any())).thenThrow(
                OutOfMemoryError("Insufficient memory")
            )

            assertThrows<AIException> {
                auraAIService.generateText("test")
            }
        }

        @Test
        @DisplayName("Should cleanup resources after failure")
        fun shouldCleanupResourcesAfterFailure() = runTest {
            whenever(mockHttpClient.post(any(), any())).thenThrow(
                RuntimeException("Unexpected error")
            )

            try {
                auraAIService.generateText("test")
            } catch (e: Exception) {
                // Expected
            }

            verify(mockHttpClient).cleanup()
        }

        @Test
        @DisplayName("Should handle connection pool exhaustion")
        fun shouldHandleConnectionPoolExhaustion() = runTest {
            whenever(mockHttpClient.post(any(), any())).thenThrow(
                ConnectionPoolExhaustedException("No available connections")
            )

            assertThrows<ConnectionPoolExhaustedException> {
                auraAIService.generateText("test")
            }
        }
    }

    @Nested
    @DisplayName("Circuit Breaker Tests")
    inner class CircuitBreakerTests {

        @Test
        @DisplayName("Should open circuit breaker after failures")
        fun shouldOpenCircuitBreakerAfterFailures() = runTest {
            // Simulate repeated failures
            whenever(mockHttpClient.post(any(), any())).thenThrow(
                ServerException("Server error", 500)
            )

            repeat(5) {
                try {
                    auraAIService.generateText("test")
                } catch (e: Exception) {
                    // Expected
                }
            }

            // Circuit breaker should be open now
            assertThrows<CircuitBreakerOpenException> {
                auraAIService.generateText("test")
            }
        }

        @Test
        @DisplayName("Should close circuit breaker after recovery")
        fun shouldCloseCircuitBreakerAfterRecovery() = runTest {
            // Open circuit breaker
            whenever(mockHttpClient.post(any(), any())).thenThrow(
                ServerException("Server error", 500)
            )

            repeat(5) {
                try {
                    auraAIService.generateText("test")
                } catch (e: Exception) {
                    // Expected
                }
            }

            // Simulate recovery
            whenever(mockHttpClient.post(any(), any())).thenReturn(
                AIResponse("success", TokenUsage(5, 10, 15), "gpt-3.5-turbo")
            )

            // Wait for circuit breaker to half-open
            Thread.sleep(1000)

            val result = auraAIService.generateText("test")
            assertEquals("success", result.text)
        }
    }

    @Nested
    @DisplayName("Caching Tests")
    inner class CachingTests {

        @Test
        @DisplayName("Should cache repeated identical requests")
        fun shouldCacheRepeatedIdenticalRequests() = runTest {
            val prompt = "Cached prompt"
            val mockResponse = AIResponse(
                text = "Cached response",
                usage = TokenUsage(5, 10, 15),
                model = "gpt-3.5-turbo"
            )

            whenever(mockHttpClient.post(any(), any())).thenReturn(mockResponse)

            // First request
            val result1 = auraAIService.generateText(prompt)
            // Second request should be cached
            val result2 = auraAIService.generateText(prompt)

            assertEquals("Cached response", result1.text)
            assertEquals("Cached response", result2.text)
            
            // Should only call HTTP client once due to caching
            verify(mockHttpClient, times(1)).post(any(), any())
        }

        @Test
        @DisplayName("Should invalidate cache after timeout")
        fun shouldInvalidateCacheAfterTimeout() = runTest {
            val prompt = "Timeout test"
            val mockResponse = AIResponse(
                text = "Response",
                usage = TokenUsage(5, 10, 15),
                model = "gpt-3.5-turbo"
            )

            whenever(mockHttpClient.post(any(), any())).thenReturn(mockResponse)

            // First request
            auraAIService.generateText(prompt)
            
            // Simulate cache timeout
            Thread.sleep(1100) // Assuming 1 second cache timeout
            
            // Second request should hit the service again
            auraAIService.generateText(prompt)

            verify(mockHttpClient, times(2)).post(any(), any())
        }

        @Test
        @DisplayName("Should handle cache miss gracefully")
        fun shouldHandleCacheMissGracefully() = runTest {
            val prompt = "Cache miss test"
            val mockResponse = AIResponse(
                text = "Fresh response",
                usage = TokenUsage(5, 10, 15),
                model = "gpt-3.5-turbo"
            )

            whenever(mockHttpClient.post(any(), any())).thenReturn(mockResponse)

            val result = auraAIService.generateText(prompt)

            assertEquals("Fresh response", result.text)
            verify(mockHttpClient).post(any(), any())
        }
    }

    @Nested
    @DisplayName("Instrumentation and Observability Tests")
    inner class InstrumentationTests {

        @Test
        @DisplayName("Should emit metrics for successful requests")
        fun shouldEmitMetricsForSuccessfulRequests() = runTest {
            val mockResponse = AIResponse(
                text = "Success",
                usage = TokenUsage(5, 10, 15),
                model = "gpt-3.5-turbo"
            )

            whenever(mockHttpClient.post(any(), any())).thenReturn(mockResponse)

            auraAIService.generateText("test")

            verify(mockMetricsCollector).recordSuccessfulRequest(any(), any())
            verify(mockMetricsCollector).recordTokenUsage(15)
            verify(mockMetricsCollector).recordResponseTime(any())
        }

        @Test
        @DisplayName("Should emit metrics for failed requests")
        fun shouldEmitMetricsForFailedRequests() = runTest {
            whenever(mockHttpClient.post(any(), any())).thenThrow(
                ServerException("Server error", 500)
            )

            try {
                auraAIService.generateText("test")
            } catch (e: Exception) {
                // Expected
            }

            verify(mockMetricsCollector).recordFailedRequest(any(), any())
            verify(mockMetricsCollector).recordError("ServerException")
        }

        @Test
        @DisplayName("Should log request details")
        fun shouldLogRequestDetails() = runTest {
            val mockResponse = AIResponse(
                text = "Logged response",
                usage = TokenUsage(5, 10, 15),
                model = "gpt-3.5-turbo"
            )

            whenever(mockHttpClient.post(any(), any())).thenReturn(mockResponse)

            auraAIService.generateText("test prompt")

            verify(mockLogger).info(contains("Generating text for prompt"))
            verify(mockLogger).info(contains("Text generation completed"))
        }
    }
}