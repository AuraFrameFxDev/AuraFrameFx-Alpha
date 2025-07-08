package dev.aurakai.auraframefx.ai.services

import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.Assertions.*
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.kotlin.*
import java.io.IOException
import java.util.concurrent.TimeoutException

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
        
        @Test
        @DisplayName("Should handle empty prompt")
        fun shouldHandleEmptyPrompt() = runTest {
            // Given
            val prompt = ""

            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("Prompt cannot be empty")
        }
        
        @Test
        @DisplayName("Should handle null prompt")
        fun shouldHandleNullPrompt() = runTest {
            // Given
            val prompt: String? = null
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.generateResponse(prompt!!)
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
        @DisplayName("Should handle HTTP error responses")
        fun shouldHandleHttpErrorResponses() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(500, "Internal Server Error")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<IOException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("HTTP error response: 500 - Internal Server Error")
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
            // Given
            val newApiKey: String? = null

            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.updateApiKey(newApiKey!!)
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
    }
    
    // Helper methods
    private fun mockHttpResponse(statusCode: Int, body: String): HttpResponse {
        val mockResponse = mock<HttpResponse>()
        whenever(mockResponse.statusCode).thenReturn(statusCode)
        whenever(mockResponse.body).thenReturn(body)
        return mockResponse
    }

    private fun async(block: suspend () -> Any): Deferred<Any> {
        // Mock async implementation for testing
        return mock<Deferred<Any>>()
    }

    private fun <T> List<Deferred<T>>.awaitAll(): List<T> {
        // Mock awaitAll implementation for testing
        return this.map { mock<T>() }
    }
}

// Custom exception classes for testing
class BatchProcessingException(message: String) : Exception(message)
class RateLimitException(message: String) : Exception(message)
class AuthenticationException(message: String) : Exception(message)
class QuotaExceededException(message: String) : Exception(message)
class JsonParseException(message: String) : Exception(message)

// Mock data classes
data class HealthCheckResult(val isHealthy: Boolean, val message: String)
    @Nested
    @DisplayName("Retry Logic Tests")
    inner class RetryLogicTests {
        
        @Test
        @DisplayName("Should retry on transient failures")
        fun shouldRetryOnTransientFailures() = runTest {
            // Given
            val prompt = "Test prompt"
            val expectedResponse = "Success after retry"
            val failureResponse = mockHttpResponse(503, "Service Temporarily Unavailable")
            val successResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any()))
                .thenReturn(failureResponse)
                .thenReturn(failureResponse)
                .thenReturn(successResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(mockHttpClient, times(3)).post(any())
            verify(mockLogger).warn("Retrying request after transient failure")
        }
        
        @Test
        @DisplayName("Should fail after max retries exceeded")
        fun shouldFailAfterMaxRetriesExceeded() = runTest {
            // Given
            val prompt = "Test prompt"
            val failureResponse = mockHttpResponse(503, "Service Temporarily Unavailable")
            
            whenever(mockHttpClient.post(any())).thenReturn(failureResponse)
            
            // When & Then
            assertThrows<MaxRetriesExceededException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("Max retries exceeded for request")
        }
        
        @Test
        @DisplayName("Should not retry on client errors")
        fun shouldNotRetryOnClientErrors() = runTest {
            // Given
            val prompt = "Test prompt"
            val clientErrorResponse = mockHttpResponse(400, "Bad Request")
            
            whenever(mockHttpClient.post(any())).thenReturn(clientErrorResponse)
            
            // When & Then
            assertThrows<ClientErrorException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockHttpClient, times(1)).post(any())
            verify(mockLogger).error("Client error, not retrying: 400 - Bad Request")
        }
    }
    
    @Nested
    @DisplayName("Request Validation Tests")
    inner class RequestValidationTests {
        
        @Test
        @DisplayName("Should validate prompt length limits")
        fun shouldValidatePromptLengthLimits() = runTest {
            // Given
            val tooLongPrompt = "A".repeat(100001) // Assuming 100k character limit
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.generateResponse(tooLongPrompt)
            }
            verify(mockLogger).error("Prompt exceeds maximum length limit")
        }
        
        @Test
        @DisplayName("Should validate prompt contains only allowed characters")
        fun shouldValidatePromptContainsOnlyAllowedCharacters() = runTest {
            // Given
            val promptWithControlCharacters = "Valid text\u0000\u0001\u0002"
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.generateResponse(promptWithControlCharacters)
            }
            verify(mockLogger).error("Prompt contains invalid characters")
        }
        
        @Test
        @DisplayName("Should validate batch size limits")
        fun shouldValidateBatchSizeLimits() = runTest {
            // Given
            val tooManyPrompts = (1..1001).map { "Prompt $it" } // Assuming 1000 limit
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.generateBatchResponses(tooManyPrompts)
            }
            verify(mockLogger).error("Batch size exceeds maximum limit")
        }
        
        @Test
        @DisplayName("Should validate batch prompts are not null")
        fun shouldValidateBatchPromptsAreNotNull() = runTest {
            // Given
            val promptsWithNull = listOf("Valid prompt", null, "Another valid prompt")
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.generateBatchResponses(promptsWithNull as List<String>)
            }
            verify(mockLogger).error("Batch contains null prompts")
        }
    }
    
    @Nested
    @DisplayName("Threading and Concurrency Tests")
    inner class ThreadingAndConcurrencyTests {
        
        @Test
        @DisplayName("Should handle thread interruption gracefully")
        fun shouldHandleThreadInterruptionGracefully() = runTest {
            // Given
            val prompt = "Test prompt"
            
            whenever(mockHttpClient.post(any())).thenAnswer {
                Thread.currentThread().interrupt()
                throw InterruptedException("Thread interrupted")
            }
            
            // When & Then
            assertThrows<InterruptedException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).warn("Thread was interrupted during request")
        }
        
        @Test
        @DisplayName("Should handle concurrent modifications to configuration")
        fun shouldHandleConcurrentModificationsToConfiguration() = runTest {
            // Given
            val newApiKey = "new-key"
            val newTimeout = 45000L
            
            // When
            val updateApiKeyJob = async { auraAIService.updateApiKey(newApiKey) }
            val updateTimeoutJob = async { auraAIService.updateTimeout(newTimeout) }
            
            // Then
            updateApiKeyJob.await()
            updateTimeoutJob.await()
            
            verify(mockConfigurationService).updateApiKey(newApiKey)
            verify(mockConfigurationService).updateTimeout(newTimeout)
        }
        
        @Test
        @DisplayName("Should maintain thread safety during bulk operations")
        fun shouldMaintainThreadSafetyDuringBulkOperations() = runTest {
            // Given
            val prompts = (1..50).map { "Prompt $it" }
            val mockHttpResponse = mockHttpResponse(200, "Response")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val results = prompts.chunked(10).map { chunk ->
                async { auraAIService.generateBatchResponses(chunk) }
            }.awaitAll()
            
            // Then
            assertEquals(5, results.size)
            verify(mockHttpClient, atLeast(5)).post(any())
        }
    }
    
    @Nested
    @DisplayName("Performance and Resource Tests")
    inner class PerformanceAndResourceTests {
        
        @Test
        @DisplayName("Should handle memory pressure gracefully")
        fun shouldHandleMemoryPressureGracefully() = runTest {
            // Given
            val prompt = "Test prompt"
            
            whenever(mockHttpClient.post(any())).thenThrow(OutOfMemoryError("Java heap space"))
            
            // When & Then
            assertThrows<OutOfMemoryError> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("Out of memory during request processing")
        }
        
        @Test
        @DisplayName("Should cleanup resources on service destruction")
        fun shouldCleanupResourcesOnServiceDestruction() {
            // Given
            val service = AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            
            // When
            service.close()
            
            // Then
            verify(mockHttpClient).close()
            verify(mockLogger).info("AuraAI service closed")
        }
        
        @Test
        @DisplayName("Should handle large response payloads")
        fun shouldHandleLargeResponsePayloads() = runTest {
            // Given
            val prompt = "Test prompt"
            val largeResponse = "A".repeat(1000000) // 1MB response
            val mockHttpResponse = mockHttpResponse(200, largeResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals(largeResponse, result)
            verify(mockLogger).info("Processing large response payload")
        }
    }
    
    @Nested
    @DisplayName("Configuration Edge Cases")
    inner class ConfigurationEdgeCasesTests {
        
        @Test
        @DisplayName("Should handle configuration service failures")
        fun shouldHandleConfigurationServiceFailures() {
            // Given
            whenever(mockConfigurationService.getApiKey()).thenThrow(RuntimeException("Config service unavailable"))
            
            // When & Then
            assertThrows<RuntimeException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
            verify(mockLogger).error("Failed to initialize configuration")
        }
        
        @Test
        @DisplayName("Should handle extremely long timeouts")
        fun shouldHandleExtremelyLongTimeouts() {
            // Given
            val extremeTimeout = Long.MAX_VALUE
            
            // When
            auraAIService.updateTimeout(extremeTimeout)
            
            // Then
            verify(mockConfigurationService).updateTimeout(extremeTimeout)
            verify(mockLogger).warn("Extremely long timeout configured: $extremeTimeout ms")
        }
        
        @Test
        @DisplayName("Should handle zero timeout")
        fun shouldHandleZeroTimeout() {
            // Given
            val zeroTimeout = 0L
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.updateTimeout(zeroTimeout)
            }
            verify(mockLogger).error("Timeout must be positive: $zeroTimeout")
        }
        
        @Test
        @DisplayName("Should handle base URL with trailing slash")
        fun shouldHandleBaseUrlWithTrailingSlash() {
            // Given
            val urlWithTrailingSlash = "https://api.test.com/"
            
            // When
            auraAIService.updateBaseUrl(urlWithTrailingSlash)
            
            // Then
            verify(mockConfigurationService).updateBaseUrl("https://api.test.com")
            verify(mockLogger).info("Base URL normalized: removed trailing slash")
        }
        
        @Test
        @DisplayName("Should handle localhost URLs")
        fun shouldHandleLocalhostUrls() {
            // Given
            val localhostUrl = "http://localhost:8080"
            
            // When
            auraAIService.updateBaseUrl(localhostUrl)
            
            // Then
            verify(mockConfigurationService).updateBaseUrl(localhostUrl)
            verify(mockLogger).warn("Using localhost URL for base URL")
        }
    }
    
    @Nested
    @DisplayName("Network Edge Cases")
    inner class NetworkEdgeCasesTests {
        
        @Test
        @DisplayName("Should handle DNS resolution failures")
        fun shouldHandleDnsResolutionFailures() = runTest {
            // Given
            val prompt = "Test prompt"
            
            whenever(mockHttpClient.post(any())).thenThrow(UnknownHostException("DNS resolution failed"))
            
            // When & Then
            assertThrows<UnknownHostException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("DNS resolution failed")
        }
        
        @Test
        @DisplayName("Should handle SSL certificate errors")
        fun shouldHandleSslCertificateErrors() = runTest {
            // Given
            val prompt = "Test prompt"
            
            whenever(mockHttpClient.post(any())).thenThrow(SSLHandshakeException("SSL certificate validation failed"))
            
            // When & Then
            assertThrows<SSLHandshakeException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("SSL certificate validation failed")
        }
        
        @Test
        @DisplayName("Should handle connection reset by peer")
        fun shouldHandleConnectionResetByPeer() = runTest {
            // Given
            val prompt = "Test prompt"
            
            whenever(mockHttpClient.post(any())).thenThrow(SocketException("Connection reset by peer"))
            
            // When & Then
            assertThrows<SocketException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("Connection reset by peer")
        }
        
        @Test
        @DisplayName("Should handle slow network responses")
        fun shouldHandleSlowNetworkResponses() = runTest {
            // Given
            val prompt = "Test prompt"
            val expectedResponse = "Slow response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenAnswer {
                Thread.sleep(5000) // Simulate slow response
                mockHttpResponse
            }
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(mockLogger).warn("Slow network response detected")
        }
    }
    
    @Nested
    @DisplayName("Response Format Tests")
    inner class ResponseFormatTests {
        
        @Test
        @DisplayName("Should handle empty response body")
        fun shouldHandleEmptyResponseBody() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(200, "")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<EmptyResponseException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("Received empty response body")
        }
        
        @Test
        @DisplayName("Should handle response with only whitespace")
        fun shouldHandleResponseWithOnlyWhitespace() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(200, "   \n\t   ")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<EmptyResponseException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("Received response with only whitespace")
        }
        
        @Test
        @DisplayName("Should handle response with unexpected content type")
        fun shouldHandleResponseWithUnexpectedContentType() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(200, "<html><body>Error</body></html>")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<UnexpectedContentTypeException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("Received unexpected content type")
        }
        
        @Test
        @DisplayName("Should handle partial JSON response")
        fun shouldHandlePartialJsonResponse() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(200, "{\"response\":\"Partial")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<JsonParseException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("Received partial JSON response")
        }
    }
    
    @Nested
    @DisplayName("Logging and Monitoring Tests")
    inner class LoggingAndMonitoringTests {
        
        @Test
        @DisplayName("Should log request metrics")
        fun shouldLogRequestMetrics() = runTest {
            // Given
            val prompt = "Test prompt"
            val expectedResponse = "Test response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            auraAIService.generateResponse(prompt)
            
            // Then
            verify(mockLogger).info("Request processed in {} ms", any<Long>())
            verify(mockLogger).info("Response size: {} bytes", any<Int>())
        }
        
        @Test
        @DisplayName("Should log batch processing metrics")
        fun shouldLogBatchProcessingMetrics() = runTest {
            // Given
            val prompts = listOf("Prompt 1", "Prompt 2", "Prompt 3")
            val expectedResponses = listOf("Response 1", "Response 2", "Response 3")
            val mockHttpResponse = mockHttpResponse(200, expectedResponses.toString())
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            auraAIService.generateBatchResponses(prompts)
            
            // Then
            verify(mockLogger).info("Batch processing completed: {} prompts in {} ms", eq(3), any<Long>())
            verify(mockLogger).info("Average response time per prompt: {} ms", any<Long>())
        }
        
        @Test
        @DisplayName("Should log health check results")
        fun shouldLogHealthCheckResults() = runTest {
            // Given
            val mockHttpResponse = mockHttpResponse(200, "OK")
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse)
            
            // When
            auraAIService.healthCheck()
            
            // Then
            verify(mockLogger).info("Health check completed successfully")
            verify(mockLogger).info("Service response time: {} ms", any<Long>())
        }
    }
}

// Additional exception classes for comprehensive testing
class MaxRetriesExceededException(message: String) : Exception(message)
class ClientErrorException(message: String) : Exception(message)
class EmptyResponseException(message: String) : Exception(message)
class UnexpectedContentTypeException(message: String) : Exception(message)
class UnknownHostException(message: String) : Exception(message)
class SSLHandshakeException(message: String) : Exception(message)
class SocketException(message: String) : Exception(message)