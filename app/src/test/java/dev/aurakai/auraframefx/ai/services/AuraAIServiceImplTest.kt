package dev.aurakai.auraframefx.ai.services

import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.async
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.awaitAll
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.RepeatedTest
import org.junit.jupiter.api.Timeout
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.junit.jupiter.params.provider.CsvSource
import org.junit.jupiter.params.provider.NullAndEmptySource
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.kotlin.*
import java.io.IOException
import java.util.concurrent.TimeoutException
import java.time.Duration
import java.net.URL
import kotlin.test.assertNotNull

@DisplayName("AuraAIServiceImpl Unit Tests")
class AuraAIServiceImplTest {
    
    @Mock
    private lateinit var mockHttpClient: HttpClient
    
    @Mock
    private lateinit var mockConfigurationService: ConfigurationService
    
    @Mock
    private lateinit var mockLogger: Logger
    
    private lateinit var auraAIService: AuraAIServiceImpl
    
    private val testApiKey = "test-api-key-123"
    private val testBaseUrl = "https://api.test.com"
    private val testTimeout = 30000L
    
    @BeforeEach
    fun setUp() {
        MockitoAnnotations.openMocks(this)
        
        // Setup default mock behaviors
        whenever(mockConfigurationService.getApiKey()).thenReturn(testApiKey)
        whenever(mockConfigurationService.getBaseUrl()).thenReturn(testBaseUrl)
        whenever(mockConfigurationService.getTimeout()).thenReturn(testTimeout)
        
        auraAIService = AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
    }
    
    @AfterEach
    fun tearDown() {
        // Clean up any resources if needed
        reset(mockHttpClient, mockConfigurationService, mockLogger)
    }
    
    @Nested
    @DisplayName("Initialization Tests")
    inner class InitializationTests {
        
        @Test
        @DisplayName("Should initialize with valid dependencies")
        fun shouldInitializeWithValidDependencies() {
            // Given & When
            val service = AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            
            // Then
            assertNotNull(service)
            verify(mockConfigurationService).getApiKey()
            verify(mockConfigurationService).getBaseUrl()
            verify(mockConfigurationService).getTimeout()
        }
        
        @Test
        @DisplayName("Should throw exception when API key is null")
        fun shouldThrowExceptionWhenApiKeyIsNull() {
            // Given
            whenever(mockConfigurationService.getApiKey()).thenReturn(null)
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }
        
        @Test
        @DisplayName("Should throw exception when API key is empty")
        fun shouldThrowExceptionWhenApiKeyIsEmpty() {
            // Given
            whenever(mockConfigurationService.getApiKey()).thenReturn("")
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }
        
        @ParameterizedTest
        @ValueSource(strings = ["", " ", "  ", "\t", "\n"])
        @DisplayName("Should throw exception for whitespace-only API keys")
        fun shouldThrowExceptionForWhitespaceApiKeys(apiKey: String) {
            // Given
            whenever(mockConfigurationService.getApiKey()).thenReturn(apiKey)
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }
        
        @Test
        @DisplayName("Should throw exception when base URL is invalid")
        fun shouldThrowExceptionWhenBaseUrlIsInvalid() {
            // Given
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("invalid-url")
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }
        
        @ParameterizedTest
        @ValueSource(strings = ["", "not-a-url", "ftp://invalid.com", "javascript:alert(1)", "file:///etc/passwd"])
        @DisplayName("Should throw exception for invalid URL formats")
        fun shouldThrowExceptionForInvalidUrlFormats(url: String) {
            // Given
            whenever(mockConfigurationService.getBaseUrl()).thenReturn(url)
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }
        
        @Test
        @DisplayName("Should throw exception when timeout is negative")
        fun shouldThrowExceptionWhenTimeoutIsNegative() {
            // Given
            whenever(mockConfigurationService.getTimeout()).thenReturn(-1L)
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }
        
        @Test
        @DisplayName("Should throw exception when timeout is zero")
        fun shouldThrowExceptionWhenTimeoutIsZero() {
            // Given
            whenever(mockConfigurationService.getTimeout()).thenReturn(0L)
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }
        
        @Test
        @DisplayName("Should accept valid HTTPS URLs")
        fun shouldAcceptValidHttpsUrls() {
            // Given
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://valid-api.example.com")
            
            // When & Then
            assertDoesNotThrow {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }
        
        @Test
        @DisplayName("Should accept valid HTTP URLs")
        fun shouldAcceptValidHttpUrls() {
            // Given
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("http://localhost:8080")
            
            // When & Then
            assertDoesNotThrow {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }
    }
    
    @Nested
    @DisplayName("Generate Response Tests")
    inner class GenerateResponseTests {
        
        @Test
        @DisplayName("Should generate response for valid prompt")
        fun shouldGenerateResponseForValidPrompt() = runTest {
            // Given
            val prompt = "What is the capital of France?"
            val expectedResponse = "The capital of France is Paris."
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(mockHttpClient).post(any())
            verify(mockLogger).info("Generating AI response for prompt length: ${prompt.length}")
        }
        
        @ParameterizedTest
        @NullAndEmptySource
        @ValueSource(strings = [" ", "  ", "\t", "\n"])
        @DisplayName("Should handle invalid prompts")
        fun shouldHandleInvalidPrompts(prompt: String?) = runTest {
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.generateResponse(prompt ?: "")
            }
        }
        
        @Test
        @DisplayName("Should handle very long prompt")
        fun shouldHandleVeryLongPrompt() = runTest {
            // Given
            val prompt = "A".repeat(10000)
            val expectedResponse = "Response for long prompt"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(mockLogger).info("Generating AI response for prompt length: ${prompt.length}")
        }
        
        @Test
        @DisplayName("Should handle extremely long prompt")
        fun shouldHandleExtremelyLongPrompt() = runTest {
            // Given
            val prompt = "A".repeat(100000) // 100K characters
            val expectedResponse = "Response for extremely long prompt"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(mockLogger).info("Generating AI response for prompt length: ${prompt.length}")
        }
        
        @ParameterizedTest
        @CsvSource(
            "400, Bad Request",
            "401, Unauthorized", 
            "403, Forbidden",
            "404, Not Found",
            "429, Too Many Requests",
            "500, Internal Server Error",
            "502, Bad Gateway",
            "503, Service Unavailable",
            "504, Gateway Timeout"
        )
        @DisplayName("Should handle various HTTP error responses")
        fun shouldHandleVariousHttpErrorResponses(statusCode: Int, statusMessage: String) = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(statusCode, statusMessage)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<IOException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("HTTP error response: $statusCode - $statusMessage")
        }
        
        @Test
        @DisplayName("Should handle network timeout")
        fun shouldHandleNetworkTimeout() = runTest {
            // Given
            val prompt = "Test prompt"
            
            whenever(mockHttpClient.post(any())).thenThrow(TimeoutException("Request timed out"))
            
            // When & Then
            assertThrows<TimeoutException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("Request timed out")
        }
        
        @Test
        @DisplayName("Should handle network connection error")
        fun shouldHandleNetworkConnectionError() = runTest {
            // Given
            val prompt = "Test prompt"
            
            whenever(mockHttpClient.post(any())).thenThrow(IOException("Connection refused"))
            
            // When & Then
            assertThrows<IOException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("Network connection error: Connection refused")
        }
        
        @Test
        @DisplayName("Should handle malformed JSON response")
        fun shouldHandleMalformedJsonResponse() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(200, "Invalid JSON{")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<JsonParseException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("Failed to parse JSON response")
        }
        
        @Test
        @DisplayName("Should handle empty response body")
        fun shouldHandleEmptyResponseBody() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(200, "")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<IllegalStateException> {
                auraAIService.generateResponse(prompt)
            }
        }
        
        @Test
        @DisplayName("Should handle null response body")
        fun shouldHandleNullResponseBody() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mock<HttpResponse>()
            whenever(mockHttpResponse.statusCode).thenReturn(200)
            whenever(mockHttpResponse.body).thenReturn(null)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<IllegalStateException> {
                auraAIService.generateResponse(prompt)
            }
        }
        
        @RepeatedTest(5)
        @DisplayName("Should be consistent across multiple calls")
        fun shouldBeConsistentAcrossMultipleCalls() = runTest {
            // Given
            val prompt = "Consistent test prompt"
            val expectedResponse = "Consistent response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals(expectedResponse, result)
        }
        
        @Test
        @Timeout(value = 5) // 5 seconds timeout
        @DisplayName("Should complete within reasonable time")
        fun shouldCompleteWithinReasonableTime() = runTest {
            // Given
            val prompt = "Fast response test"
            val expectedResponse = "Fast response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals(expectedResponse, result)
        }
    }
    
    @Nested
    @DisplayName("Generate Batch Responses Tests")
    inner class GenerateBatchResponsesTests {
        
        @Test
        @DisplayName("Should generate batch responses for multiple prompts")
        fun shouldGenerateBatchResponsesForMultiplePrompts() = runTest {
            // Given
            val prompts = listOf("Prompt 1", "Prompt 2", "Prompt 3")
            val expectedResponses = listOf("Response 1", "Response 2", "Response 3")
            val mockHttpResponse = mockHttpResponse(200, expectedResponses.toString())
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val results = auraAIService.generateBatchResponses(prompts)
            
            // Then
            assertEquals(expectedResponses.size, results.size)
            verify(mockHttpClient).post(any())
            verify(mockLogger).info("Generating batch AI responses for ${prompts.size} prompts")
        }
        
        @Test
        @DisplayName("Should handle empty prompt list")
        fun shouldHandleEmptyPromptList() = runTest {
            // Given
            val prompts = emptyList<String>()
            
            // When
            val results = auraAIService.generateBatchResponses(prompts)
            
            // Then
            assertTrue(results.isEmpty())
            verify(mockLogger).info("No prompts provided for batch processing")
        }
        
        @Test
        @DisplayName("Should handle single prompt in batch")
        fun shouldHandleSinglePromptInBatch() = runTest {
            // Given
            val prompts = listOf("Single prompt")
            val expectedResponse = listOf("Single response")
            val mockHttpResponse = mockHttpResponse(200, expectedResponse.toString())
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val results = auraAIService.generateBatchResponses(prompts)
            
            // Then
            assertEquals(1, results.size)
            assertEquals(expectedResponse[0], results[0])
        }
        
        @Test
        @DisplayName("Should handle batch processing with partial failures")
        fun shouldHandleBatchProcessingWithPartialFailures() = runTest {
            // Given
            val prompts = listOf("Valid prompt", "", "Another valid prompt")
            
            // When & Then
            assertThrows<BatchProcessingException> {
                auraAIService.generateBatchResponses(prompts)
            }
            verify(mockLogger).error("Batch processing failed due to invalid prompts")
        }
        
        @Test
        @DisplayName("Should handle large batch sizes")
        fun shouldHandleLargeBatchSizes() = runTest {
            // Given
            val prompts = (1..100).map { "Prompt $it" }
            val expectedResponses = (1..100).map { "Response $it" }
            val mockHttpResponse = mockHttpResponse(200, expectedResponses.toString())
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val results = auraAIService.generateBatchResponses(prompts)
            
            // Then
            assertEquals(100, results.size)
            verify(mockLogger).info("Generating batch AI responses for ${prompts.size} prompts")
        }
        
        @Test
        @DisplayName("Should handle batch with mixed prompt lengths")
        fun shouldHandleBatchWithMixedPromptLengths() = runTest {
            // Given
            val prompts = listOf(
                "Short",
                "Medium length prompt with more details",
                "A".repeat(1000), // Very long prompt
                "Another short one"
            )
            val expectedResponses = listOf("Resp 1", "Resp 2", "Resp 3", "Resp 4")
            val mockHttpResponse = mockHttpResponse(200, expectedResponses.toString())
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val results = auraAIService.generateBatchResponses(prompts)
            
            // Then
            assertEquals(4, results.size)
        }
        
        @Test
        @DisplayName("Should handle batch processing timeout")
        fun shouldHandleBatchProcessingTimeout() = runTest {
            // Given
            val prompts = listOf("Prompt 1", "Prompt 2")
            
            whenever(mockHttpClient.post(any())).thenThrow(TimeoutException("Batch processing timed out"))
            
            // When & Then
            assertThrows<TimeoutException> {
                auraAIService.generateBatchResponses(prompts)
            }
        }
        
        @Test
        @DisplayName("Should handle batch with null prompts")
        fun shouldHandleBatchWithNullPrompts() = runTest {
            // Given
            val prompts = listOf("Valid prompt", null, "Another valid prompt")
            
            // When & Then
            assertThrows<BatchProcessingException> {
                auraAIService.generateBatchResponses(prompts.filterNotNull())
            }
        }
    }
    
    @Nested
    @DisplayName("Configuration Tests")
    inner class ConfigurationTests {
        
        @Test
        @DisplayName("Should update API key")
        fun shouldUpdateApiKey() {
            // Given
            val newApiKey = "new-api-key-456"
            
            // When
            auraAIService.updateApiKey(newApiKey)
            
            // Then
            verify(mockConfigurationService).updateApiKey(newApiKey)
            verify(mockLogger).info("API key updated successfully")
        }
        
        @Test
        @DisplayName("Should throw exception when updating with null API key")
        fun shouldThrowExceptionWhenUpdatingWithNullApiKey() {
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.updateApiKey(null)
            }
        }
        
        @ParameterizedTest
        @ValueSource(strings = ["", " ", "  ", "\t", "\n"])
        @DisplayName("Should throw exception for invalid API key updates")
        fun shouldThrowExceptionForInvalidApiKeyUpdates(apiKey: String) {
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.updateApiKey(apiKey)
            }
        }
        
        @Test
        @DisplayName("Should update base URL")
        fun shouldUpdateBaseUrl() {
            // Given
            val newBaseUrl = "https://new-api.test.com"
            
            // When
            auraAIService.updateBaseUrl(newBaseUrl)
            
            // Then
            verify(mockConfigurationService).updateBaseUrl(newBaseUrl)
            verify(mockLogger).info("Base URL updated successfully")
        }
        
        @Test
        @DisplayName("Should validate base URL format")
        fun shouldValidateBaseUrlFormat() {
            // Given
            val invalidUrl = "not-a-valid-url"
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.updateBaseUrl(invalidUrl)
            }
            verify(mockLogger).error("Invalid base URL format: $invalidUrl")
        }
        
        @ParameterizedTest
        @CsvSource(
            "'', 'Empty URL'",
            "'not-a-url', 'Invalid format'",
            "'ftp://test.com', 'Unsupported protocol'",
            "'javascript:alert(1)', 'Security risk'",
            "'file:///etc/passwd', 'Local file access'"
        )
        @DisplayName("Should reject invalid URL formats with specific reasons")
        fun shouldRejectInvalidUrlFormats(url: String, reason: String) {
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.updateBaseUrl(url)
            }
        }
        
        @Test
        @DisplayName("Should update timeout value")
        fun shouldUpdateTimeoutValue() {
            // Given
            val newTimeout = 60000L
            
            // When
            auraAIService.updateTimeout(newTimeout)
            
            // Then
            verify(mockConfigurationService).updateTimeout(newTimeout)
            verify(mockLogger).info("Timeout updated to $newTimeout ms")
        }
        
        @Test
        @DisplayName("Should validate timeout is positive")
        fun shouldValidateTimeoutIsPositive() {
            // Given
            val negativeTimeout = -1000L
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.updateTimeout(negativeTimeout)
            }
            verify(mockLogger).error("Timeout must be positive: $negativeTimeout")
        }
        
        @ParameterizedTest
        @ValueSource(longs = [-1L, 0L, -100L, -9999L])
        @DisplayName("Should reject non-positive timeout values")
        fun shouldRejectNonPositiveTimeoutValues(timeout: Long) {
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.updateTimeout(timeout)
            }
        }
        
        @Test
        @DisplayName("Should handle very large timeout values")
        fun shouldHandleVeryLargeTimeoutValues() {
            // Given
            val largeTimeout = Long.MAX_VALUE
            
            // When & Then
            assertDoesNotThrow {
                auraAIService.updateTimeout(largeTimeout)
            }
            verify(mockConfigurationService).updateTimeout(largeTimeout)
        }
        
        @Test
        @DisplayName("Should handle configuration service failures")
        fun shouldHandleConfigurationServiceFailures() {
            // Given
            val newApiKey = "test-key"
            whenever(mockConfigurationService.updateApiKey(any())).thenThrow(RuntimeException("Config service error"))
            
            // When & Then
            assertThrows<RuntimeException> {
                auraAIService.updateApiKey(newApiKey)
            }
        }
    }
    
    @Nested
    @DisplayName("Health Check Tests")
    inner class HealthCheckTests {
        
        @Test
        @DisplayName("Should return healthy status when service is available")
        fun shouldReturnHealthyStatusWhenServiceIsAvailable() = runTest {
            // Given
            val mockHttpResponse = mockHttpResponse(200, "OK")
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.healthCheck()
            
            // Then
            assertTrue(result.isHealthy)
            assertEquals("Service is healthy", result.message)
        }
        
        @Test
        @DisplayName("Should return unhealthy status when service is unavailable")
        fun shouldReturnUnhealthyStatusWhenServiceIsUnavailable() = runTest {
            // Given
            whenever(mockHttpClient.get(any())).thenThrow(IOException("Service unavailable"))
            
            // When
            val result = auraAIService.healthCheck()
            
            // Then
            assertFalse(result.isHealthy)
            assertEquals("Service is unhealthy: Service unavailable", result.message)
        }
        
        @Test
        @DisplayName("Should handle health check timeout")
        fun shouldHandleHealthCheckTimeout() = runTest {
            // Given
            whenever(mockHttpClient.get(any())).thenThrow(TimeoutException("Health check timed out"))
            
            // When
            val result = auraAIService.healthCheck()
            
            // Then
            assertFalse(result.isHealthy)
            assertEquals("Service is unhealthy: Health check timed out", result.message)
        }
        
        @ParameterizedTest
        @ValueSource(ints = [400, 401, 403, 404, 500, 502, 503, 504])
        @DisplayName("Should return unhealthy for various HTTP error codes")
        fun shouldReturnUnhealthyForHttpErrors(statusCode: Int) = runTest {
            // Given
            val mockHttpResponse = mockHttpResponse(statusCode, "Error")
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.healthCheck()
            
            // Then
            assertFalse(result.isHealthy)
            assertTrue(result.message.contains("unhealthy"))
        }
        
        @RepeatedTest(3)
        @DisplayName("Should consistently report health status")
        fun shouldConsistentlyReportHealthStatus() = runTest {
            // Given
            val mockHttpResponse = mockHttpResponse(200, "OK")
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.healthCheck()
            
            // Then
            assertTrue(result.isHealthy)
        }
        
        @Test
        @DisplayName("Should handle network connection issues in health check")
        fun shouldHandleNetworkConnectionIssuesInHealthCheck() = runTest {
            // Given
            whenever(mockHttpClient.get(any())).thenThrow(IOException("Network unreachable"))
            
            // When
            val result = auraAIService.healthCheck()
            
            // Then
            assertFalse(result.isHealthy)
            assertTrue(result.message.contains("Network unreachable"))
        }
        
        @Test
        @Timeout(value = 3)
        @DisplayName("Should complete health check within timeout")
        fun shouldCompleteHealthCheckWithinTimeout() = runTest {
            // Given
            val mockHttpResponse = mockHttpResponse(200, "OK")
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.healthCheck()
            
            // Then
            assertNotNull(result)
        }
    }
    
    @Nested
    @DisplayName("Edge Cases and Error Handling")
    inner class EdgeCasesAndErrorHandlingTests {
        
        @Test
        @DisplayName("Should handle concurrent requests gracefully")
        fun shouldHandleConcurrentRequestsGracefully() = runTest {
            // Given
            val prompts = (1..10).map { "Prompt $it" }
            val mockHttpResponse = mockHttpResponse(200, "Response")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val results = prompts.map { prompt ->
                async { auraAIService.generateResponse(prompt) }
            }.awaitAll()
            
            // Then
            assertEquals(10, results.size)
            verify(mockHttpClient, times(10)).post(any())
        }
        
        @Test
        @DisplayName("Should handle rate limiting gracefully")
        fun shouldHandleRateLimitingGracefully() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(429, "Rate limit exceeded")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<RateLimitException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).warn("Rate limit exceeded, retrying...")
        }
        
        @Test
        @DisplayName("Should handle authentication failures")
        fun shouldHandleAuthenticationFailures() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(401, "Unauthorized")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<AuthenticationException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("Authentication failed: Invalid API key")
        }
        
        @Test
        @DisplayName("Should handle quota exceeded")
        fun shouldHandleQuotaExceeded() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(403, "Quota exceeded")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<QuotaExceededException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("API quota exceeded")
        }
        
        @Test
        @DisplayName("Should handle special characters in prompt")
        fun shouldHandleSpecialCharactersInPrompt() = runTest {
            // Given
            val prompt = "Test with special characters: éñüß@#$%^&*()[]{}|\\:;\"'<>,.?/~`"
            val expectedResponse = "Response with special characters"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals(expectedResponse, result)
        }
        
        @Test
        @DisplayName("Should handle Unicode characters in prompt")
        fun shouldHandleUnicodeCharactersInPrompt() = runTest {
            // Given
            val prompt = "Test with Unicode: 你好世界 🌍 🚀 ✨"
            val expectedResponse = "Response with Unicode"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals(expectedResponse, result)
        }
        
        @Test
        @DisplayName("Should handle newlines and formatting in prompt")
        fun shouldHandleNewlinesAndFormattingInPrompt() = runTest {
            // Given
            val prompt = """
                Multi-line prompt
                with various formatting:
                - Bullet points
                - Numbers: 1, 2, 3
                - Special chars: @#$%
                
                And some extra whitespace.
            """.trimIndent()
            val expectedResponse = "Formatted response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals(expectedResponse, result)
        }
        
        @Test
        @DisplayName("Should handle JSON injection attempts in prompt")
        fun shouldHandleJsonInjectionAttemptsInPrompt() = runTest {
            // Given
            val prompt = """{"malicious": "payload", "prompt": "actual content"}"""
            val expectedResponse = "Secure response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals(expectedResponse, result)
        }
        
        @Test
        @DisplayName("Should handle SQL injection attempts in prompt")
        fun shouldHandleSqlInjectionAttemptsInPrompt() = runTest {
            // Given
            val prompt = "'; DROP TABLE users; --"
            val expectedResponse = "Secure response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals(expectedResponse, result)
        }
        
        @Test
        @DisplayName("Should handle memory pressure during large requests")
        fun shouldHandleMemoryPressureDuringLargeRequests() = runTest {
            // Given
            val prompt = "A".repeat(50000) // 50K characters
            val expectedResponse = "Large response".repeat(1000)
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertNotNull(result)
            assertEquals(expectedResponse, result)
        }
        
        @Test
        @DisplayName("Should handle service degradation gracefully")
        fun shouldHandleServiceDegradationGracefully() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(503, "Service temporarily unavailable")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<ServiceUnavailableException> {
                auraAIService.generateResponse(prompt)
            }
        }
        
        @Test
        @DisplayName("Should handle corrupted response data")
        fun shouldHandleCorruptedResponseData() = runTest {
            // Given
            val prompt = "Test prompt"
            val corruptedResponse = "\u0000\u0001\u0002Invalid binary data\u0003"
            val mockHttpResponse = mockHttpResponse(200, corruptedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<DataCorruptionException> {
                auraAIService.generateResponse(prompt)
            }
        }
    }
    
    @Nested
    @DisplayName("Performance and Load Tests")
    inner class PerformanceAndLoadTests {
        
        @RepeatedTest(10)
        @DisplayName("Should maintain performance under repeated calls")
        fun shouldMaintainPerformanceUnderRepeatedCalls() = runTest {
            // Given
            val prompt = "Performance test prompt"
            val expectedResponse = "Performance response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val startTime = System.currentTimeMillis()
            val result = auraAIService.generateResponse(prompt)
            val endTime = System.currentTimeMillis()
            
            // Then
            assertEquals(expectedResponse, result)
            assertTrue(endTime - startTime < 1000) // Should complete within 1 second
        }
        
        @Test
        @DisplayName("Should handle burst requests efficiently")
        fun shouldHandleBurstRequestsEfficiently() = runTest {
            // Given
            val prompts = (1..50).map { "Burst prompt $it" }
            val mockHttpResponse = mockHttpResponse(200, "Response")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val startTime = System.currentTimeMillis()
            val results = prompts.map { prompt ->
                async { auraAIService.generateResponse(prompt) }
            }.awaitAll()
            val endTime = System.currentTimeMillis()
            
            // Then
            assertEquals(50, results.size)
            assertTrue(endTime - startTime < 5000) // Should complete within 5 seconds
        }
    }
    
    // Helper methods
    private fun mockHttpResponse(statusCode: Int, body: String): HttpResponse {
        val mockResponse = mock<HttpResponse>()
        whenever(mockResponse.statusCode).thenReturn(statusCode)
        whenever(mockResponse.body).thenReturn(body)
        return mockResponse
    }
}

// Custom exception classes for testing
class BatchProcessingException(message: String) : Exception(message)
class RateLimitException(message: String) : Exception(message)
class AuthenticationException(message: String) : Exception(message)
class QuotaExceededException(message: String) : Exception(message)
class JsonParseException(message: String) : Exception(message)
class ServiceUnavailableException(message: String) : Exception(message)
class DataCorruptionException(message: String) : Exception(message)

// Mock data classes
data class HealthCheckResult(val isHealthy: Boolean, val message: String)

// Mock interfaces for compilation
interface HttpClient {
    suspend fun post(request: Any): HttpResponse
    suspend fun get(request: Any): HttpResponse
}

interface HttpResponse {
    val statusCode: Int
    val body: String?
}

interface ConfigurationService {
    fun getApiKey(): String?
    fun getBaseUrl(): String
    fun getTimeout(): Long
    fun updateApiKey(apiKey: String)
    fun updateBaseUrl(baseUrl: String) 
    fun updateTimeout(timeout: Long)
}

interface Logger {
    fun info(message: String)
    fun error(message: String)
    fun warn(message: String)
}

// Mock implementation class interface for compilation
interface AuraAIService {
    suspend fun generateResponse(prompt: String): String
    suspend fun generateBatchResponses(prompts: List<String>): List<String>
    suspend fun healthCheck(): HealthCheckResult
    fun updateApiKey(apiKey: String)
    fun updateBaseUrl(baseUrl: String)
    fun updateTimeout(timeout: Long)
}

class AuraAIServiceImpl(
    private val httpClient: HttpClient,
    private val configurationService: ConfigurationService,
    private val logger: Logger
) : AuraAIService {
    
    init {
        val apiKey = configurationService.getApiKey()
        require(!apiKey.isNullOrBlank()) { "API key cannot be null or empty" }
        
        val baseUrl = configurationService.getBaseUrl()
        require(isValidUrl(baseUrl)) { "Invalid base URL format" }
        
        val timeout = configurationService.getTimeout()
        require(timeout > 0) { "Timeout must be positive" }
    }
    
    override suspend fun generateResponse(prompt: String): String {
        require(prompt.isNotBlank()) { "Prompt cannot be empty" }
        logger.info("Generating AI response for prompt length: ${prompt.length}")
        
        val response = httpClient.post(createRequest(prompt))
        
        return when (response.statusCode) {
            200 -> response.body ?: throw IllegalStateException("Empty response body")
            401 -> throw AuthenticationException("Authentication failed: Invalid API key")
            403 -> throw QuotaExceededException("API quota exceeded")
            429 -> throw RateLimitException("Rate limit exceeded")
            else -> throw IOException("HTTP error response: ${response.statusCode} - ${response.body}")
        }
    }
    
    override suspend fun generateBatchResponses(prompts: List<String>): List<String> {
        if (prompts.isEmpty()) {
            logger.info("No prompts provided for batch processing")
            return emptyList()
        }
        
        if (prompts.any { it.isBlank() }) {
            throw BatchProcessingException("Batch processing failed due to invalid prompts")
        }
        
        logger.info("Generating batch AI responses for ${prompts.size} prompts")
        
        val response = httpClient.post(createBatchRequest(prompts))
        return parseBatchResponse(response)
    }
    
    override suspend fun healthCheck(): HealthCheckResult {
        return try {
            val response = httpClient.get(createHealthCheckRequest())
            if (response.statusCode == 200) {
                HealthCheckResult(true, "Service is healthy")
            } else {
                HealthCheckResult(false, "Service is unhealthy: HTTP ${response.statusCode}")
            }
        } catch (e: Exception) {
            HealthCheckResult(false, "Service is unhealthy: ${e.message}")
        }
    }
    
    override fun updateApiKey(apiKey: String) {
        require(apiKey.isNotBlank()) { "API key cannot be empty" }
        configurationService.updateApiKey(apiKey)
        logger.info("API key updated successfully")
    }
    
    override fun updateBaseUrl(baseUrl: String) {
        require(isValidUrl(baseUrl)) { "Invalid base URL format" }
        configurationService.updateBaseUrl(baseUrl)
        logger.info("Base URL updated successfully")
    }
    
    override fun updateTimeout(timeout: Long) {
        require(timeout > 0) { "Timeout must be positive" }
        configurationService.updateTimeout(timeout)
        logger.info("Timeout updated to $timeout ms")
    }
    
    private fun isValidUrl(url: String): Boolean {
        return try {
            val parsedUrl = URL(url)
            parsedUrl.protocol in listOf("http", "https")
        } catch (e: Exception) {
            false
        }
    }
    
    private fun createRequest(prompt: String): Any = mapOf("prompt" to prompt)
    private fun createBatchRequest(prompts: List<String>): Any = mapOf("prompts" to prompts)
    private fun createHealthCheckRequest(): Any = mapOf("action" to "health")
    private fun parseBatchResponse(response: HttpResponse): List<String> = listOf("Mock response")
}