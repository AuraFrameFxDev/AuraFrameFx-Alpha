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
    @DisplayName("Request Builder Tests")
    inner class RequestBuilderTests {
        
        @Test
        @DisplayName("Should build request with correct headers")
        fun shouldBuildRequestWithCorrectHeaders() = runTest {
            // Given
            val prompt = "Test prompt"
            val expectedResponse = "Test response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            auraAIService.generateResponse(prompt)
            
            // Then
            verify(mockHttpClient).post(argThat { request ->
                request.headers.containsKey("Authorization") &&
                request.headers.containsKey("Content-Type") &&
                request.headers["Authorization"] == "Bearer $testApiKey" &&
                request.headers["Content-Type"] == "application/json"
            })
        }
        
        @Test
        @DisplayName("Should include user-agent header")
        fun shouldIncludeUserAgentHeader() = runTest {
            // Given
            val prompt = "Test prompt"
            val expectedResponse = "Test response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            auraAIService.generateResponse(prompt)
            
            // Then
            verify(mockHttpClient).post(argThat { request ->
                request.headers.containsKey("User-Agent") &&
                request.headers["User-Agent"]?.contains("AuraFrameFX") == true
            })
        }
        
        @Test
        @DisplayName("Should build request with correct timeout")
        fun shouldBuildRequestWithCorrectTimeout() = runTest {
            // Given
            val prompt = "Test prompt"
            val expectedResponse = "Test response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            auraAIService.generateResponse(prompt)
            
            // Then
            verify(mockHttpClient).post(argThat { request ->
                request.timeout == testTimeout
            })
        }
        
        @Test
        @DisplayName("Should escape special characters in JSON payload")
        fun shouldEscapeSpecialCharactersInJsonPayload() = runTest {
            // Given
            val prompt = "Test with \"quotes\" and \n newlines"
            val expectedResponse = "Test response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            auraAIService.generateResponse(prompt)
            
            // Then
            verify(mockHttpClient).post(argThat { request ->
                request.body.contains("\\\"") && request.body.contains("\\n")
            })
        }
    }
    
    @Nested
    @DisplayName("Response Parser Tests")
    inner class ResponseParserTests {
        
        @Test
        @DisplayName("Should parse successful JSON response")
        fun shouldParseSuccessfulJsonResponse() = runTest {
            // Given
            val prompt = "Test prompt"
            val jsonResponse = """{"response": "Test response", "status": "success"}"""
            val mockHttpResponse = mockHttpResponse(200, jsonResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals("Test response", result)
        }
        
        @Test
        @DisplayName("Should handle response with nested JSON")
        fun shouldHandleResponseWithNestedJson() = runTest {
            // Given
            val prompt = "Test prompt"
            val jsonResponse = """{"data": {"response": "Nested response"}, "metadata": {"tokens": 10}}"""
            val mockHttpResponse = mockHttpResponse(200, jsonResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals("Nested response", result)
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
            verify(mockLogger).error("Empty response body received")
        }
        
        @Test
        @DisplayName("Should handle response with missing required fields")
        fun shouldHandleResponseWithMissingRequiredFields() = runTest {
            // Given
            val prompt = "Test prompt"
            val jsonResponse = """{"status": "success"}"""
            val mockHttpResponse = mockHttpResponse(200, jsonResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<IllegalStateException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("Response missing required 'response' field")
        }
        
        @Test
        @DisplayName("Should handle response with null values")
        fun shouldHandleResponseWithNullValues() = runTest {
            // Given
            val prompt = "Test prompt"
            val jsonResponse = """{"response": null, "status": "success"}"""
            val mockHttpResponse = mockHttpResponse(200, jsonResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<IllegalStateException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("Response field contains null value")
        }
    }
    
    @Nested
    @DisplayName("Retry Logic Tests")
    inner class RetryLogicTests {
        
        @Test
        @DisplayName("Should retry on transient failures")
        fun shouldRetryOnTransientFailures() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(200, "Success after retry")
            
            whenever(mockHttpClient.post(any()))
                .thenThrow(IOException("Connection refused"))
                .thenThrow(IOException("Connection refused"))
                .thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals("Success after retry", result)
            verify(mockHttpClient, times(3)).post(any())
            verify(mockLogger, times(2)).warn("Retrying request due to transient failure")
        }
        
        @Test
        @DisplayName("Should not retry on permanent failures")
        fun shouldNotRetryOnPermanentFailures() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(400, "Bad Request")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockHttpClient, times(1)).post(any())
            verify(mockLogger, never()).warn(contains("Retrying"))
        }
        
        @Test
        @DisplayName("Should respect maximum retry attempts")
        fun shouldRespectMaximumRetryAttempts() = runTest {
            // Given
            val prompt = "Test prompt"
            
            whenever(mockHttpClient.post(any())).thenThrow(IOException("Connection refused"))
            
            // When & Then
            assertThrows<IOException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockHttpClient, times(3)).post(any()) // Default max retries
            verify(mockLogger).error("Maximum retry attempts exceeded")
        }
        
        @Test
        @DisplayName("Should implement exponential backoff")
        fun shouldImplementExponentialBackoff() = runTest {
            // Given
            val prompt = "Test prompt"
            val startTime = System.currentTimeMillis()
            
            whenever(mockHttpClient.post(any())).thenThrow(IOException("Connection refused"))
            
            // When
            try {
                auraAIService.generateResponse(prompt)
            } catch (e: IOException) {
                // Expected
            }
            
            // Then
            val endTime = System.currentTimeMillis()
            val totalTime = endTime - startTime
            assertTrue(totalTime >= 1000) // Should have some delay due to backoff
            verify(mockLogger).debug(contains("Backing off for"))
        }
    }
    
    @Nested
    @DisplayName("Streaming Response Tests")
    inner class StreamingResponseTests {
        
        @Test
        @DisplayName("Should handle streaming response")
        fun shouldHandleStreamingResponse() = runTest {
            // Given
            val prompt = "Test prompt"
            val streamingResponse = listOf("Part 1", "Part 2", "Part 3")
            val mockStreamResponse = mock<StreamingHttpResponse>()
            whenever(mockStreamResponse.stream()).thenReturn(streamingResponse.asSequence())
            
            whenever(mockHttpClient.postStreaming(any())).thenReturn(mockStreamResponse)
            
            // When
            val results = mutableListOf<String>()
            auraAIService.generateStreamingResponse(prompt) { chunk ->
                results.add(chunk)
            }
            
            // Then
            assertEquals(streamingResponse, results)
            verify(mockLogger).info("Starting streaming response for prompt length: ${prompt.length}")
        }
        
        @Test
        @DisplayName("Should handle streaming response interruption")
        fun shouldHandleStreamingResponseInterruption() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockStreamResponse = mock<StreamingHttpResponse>()
            whenever(mockStreamResponse.stream()).thenThrow(IOException("Stream interrupted"))
            
            whenever(mockHttpClient.postStreaming(any())).thenReturn(mockStreamResponse)
            
            // When & Then
            assertThrows<IOException> {
                auraAIService.generateStreamingResponse(prompt) { }
            }
            verify(mockLogger).error("Streaming response interrupted")
        }
        
        @Test
        @DisplayName("Should handle empty streaming response")
        fun shouldHandleEmptyStreamingResponse() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockStreamResponse = mock<StreamingHttpResponse>()
            whenever(mockStreamResponse.stream()).thenReturn(emptySequence())
            
            whenever(mockHttpClient.postStreaming(any())).thenReturn(mockStreamResponse)
            
            // When
            val results = mutableListOf<String>()
            auraAIService.generateStreamingResponse(prompt) { chunk ->
                results.add(chunk)
            }
            
            // Then
            assertTrue(results.isEmpty())
            verify(mockLogger).warn("Received empty streaming response")
        }
    }
    
    @Nested
    @DisplayName("Caching Tests")
    inner class CachingTests {
        
        @Test
        @DisplayName("Should cache successful responses")
        fun shouldCacheSuccessfulResponses() = runTest {
            // Given
            val prompt = "Test prompt"
            val expectedResponse = "Cached response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result1 = auraAIService.generateResponse(prompt)
            val result2 = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals(expectedResponse, result1)
            assertEquals(expectedResponse, result2)
            verify(mockHttpClient, times(1)).post(any()) // Only one actual HTTP call
            verify(mockLogger).info("Response retrieved from cache")
        }
        
        @Test
        @DisplayName("Should not cache error responses")
        fun shouldNotCacheErrorResponses() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(500, "Internal Server Error")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<IOException> {
                auraAIService.generateResponse(prompt)
            }
            assertThrows<IOException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockHttpClient, times(2)).post(any()) // Two separate HTTP calls
            verify(mockLogger, never()).info("Response retrieved from cache")
        }
        
        @Test
        @DisplayName("Should respect cache expiration")
        fun shouldRespectCacheExpiration() = runTest {
            // Given
            val prompt = "Test prompt"
            val expectedResponse = "Fresh response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            auraAIService.generateResponse(prompt)
            
            // Simulate cache expiration
            Thread.sleep(1100) // Assuming 1 second cache TTL
            
            auraAIService.generateResponse(prompt)
            
            // Then
            verify(mockHttpClient, times(2)).post(any())
            verify(mockLogger).info("Cache entry expired, making fresh request")
        }
        
        @Test
        @DisplayName("Should handle cache invalidation")
        fun shouldHandleCacheInvalidation() = runTest {
            // Given
            val prompt = "Test prompt"
            val expectedResponse = "Response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            auraAIService.generateResponse(prompt)
            auraAIService.invalidateCache()
            auraAIService.generateResponse(prompt)
            
            // Then
            verify(mockHttpClient, times(2)).post(any())
            verify(mockLogger).info("Cache invalidated")
        }
    }
    
    @Nested
    @DisplayName("Performance and Load Tests")
    inner class PerformanceAndLoadTests {
        
        @Test
        @DisplayName("Should handle high volume of requests")
        fun shouldHandleHighVolumeOfRequests() = runTest {
            // Given
            val prompts = (1..100).map { "Prompt $it" }
            val mockHttpResponse = mockHttpResponse(200, "Response")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val startTime = System.currentTimeMillis()
            val results = prompts.map { prompt ->
                async { auraAIService.generateResponse(prompt) }
            }.awaitAll()
            val endTime = System.currentTimeMillis()
            
            // Then
            assertEquals(100, results.size)
            assertTrue(endTime - startTime < 5000) // Should complete within 5 seconds
            verify(mockHttpClient, times(100)).post(any())
        }
        
        @Test
        @DisplayName("Should handle memory pressure gracefully")
        fun shouldHandleMemoryPressureGracefully() = runTest {
            // Given
            val largePrompt = "A".repeat(1000000) // 1MB prompt
            val expectedResponse = "Response for large prompt"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(largePrompt)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(mockLogger).warn("Processing large prompt of size: ${largePrompt.length}")
        }
        
        @Test
        @DisplayName("Should implement request throttling")
        fun shouldImplementRequestThrottling() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(200, "Response")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val startTime = System.currentTimeMillis()
            repeat(5) {
                auraAIService.generateResponse(prompt)
            }
            val endTime = System.currentTimeMillis()
            
            // Then
            assertTrue(endTime - startTime >= 1000) // Should be throttled
            verify(mockLogger, atLeast(1)).debug("Request throttled")
        }
    }
    
    @Nested
    @DisplayName("Security Tests")
    inner class SecurityTests {
        
        @Test
        @DisplayName("Should sanitize sensitive data in logs")
        fun shouldSanitizeSensitiveDataInLogs() = runTest {
            // Given
            val prompt = "My API key is sk-1234567890 and password is secret123"
            val expectedResponse = "Sanitized response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            auraAIService.generateResponse(prompt)
            
            // Then
            verify(mockLogger).info(argThat { message ->
                !message.contains("sk-1234567890") && !message.contains("secret123")
            })
        }
        
        @Test
        @DisplayName("Should validate API key format")
        fun shouldValidateApiKeyFormat() {
            // Given
            val invalidApiKey = "invalid-key-format"
            whenever(mockConfigurationService.getApiKey()).thenReturn(invalidApiKey)
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
            verify(mockLogger).error("Invalid API key format")
        }
        
        @Test
        @DisplayName("Should reject potentially malicious prompts")
        fun shouldRejectPotentiallyMaliciousPrompts() = runTest {
            // Given
            val maliciousPrompt = "<script>alert('xss')</script>"
            
            // When & Then
            assertThrows<SecurityException> {
                auraAIService.generateResponse(maliciousPrompt)
            }
            verify(mockLogger).warn("Potentially malicious prompt detected and rejected")
        }
        
        @Test
        @DisplayName("Should implement rate limiting per client")
        fun shouldImplementRateLimitingPerClient() = runTest {
            // Given
            val prompt = "Test prompt"
            val clientId = "test-client-123"
            
            // When
            repeat(10) {
                try {
                    auraAIService.generateResponse(prompt, clientId)
                } catch (e: RateLimitException) {
                    // Expected after hitting rate limit
                }
            }
            
            // Then
            verify(mockLogger).warn("Rate limit exceeded for client: $clientId")
        }
    }
    
    @Nested
    @DisplayName("Monitoring and Observability Tests")
    inner class MonitoringAndObservabilityTests {
        
        @Test
        @DisplayName("Should emit metrics for successful requests")
        fun shouldEmitMetricsForSuccessfulRequests() = runTest {
            // Given
            val prompt = "Test prompt"
            val expectedResponse = "Test response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            auraAIService.generateResponse(prompt)
            
            // Then
            verify(mockMetricsService).incrementCounter("aura_ai_requests_total", mapOf("status" to "success"))
            verify(mockMetricsService).recordTimer("aura_ai_request_duration", any())
        }
        
        @Test
        @DisplayName("Should emit metrics for failed requests")
        fun shouldEmitMetricsForFailedRequests() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(500, "Internal Server Error")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            try {
                auraAIService.generateResponse(prompt)
   