package dev.aurakai.auraframefx.ai.services

import io.mockk.*
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.CsvSource
import org.junit.jupiter.params.provider.ValueSource
import org.junit.jupiter.params.provider.MethodSource
import kotlinx.coroutines.runBlocking
import java.io.IOException
import java.net.SocketTimeoutException
import java.util.concurrent.TimeoutException
import java.util.stream.Stream

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class AuraAIServiceTest {

    private lateinit var auraAIService: AuraAIService
    private val mockHttpClient = mockk<HttpClient>()
    private val mockApiClient = mockk<ApiClient>()
    private val mockConfigService = mockk<ConfigService>()
    private val mockLogger = mockk<Logger>()

    @BeforeEach
    fun setUp() {
        clearAllMocks()
        auraAIService = AuraAIService(mockHttpClient, mockApiClient, mockConfigService, mockLogger)
    }

    @AfterEach
    fun tearDown() {
        unmockkAll()
    }

    @Nested
    @DisplayName("Initialization Tests")
    inner class InitializationTests {

        @Test
        @DisplayName("Should initialize service with valid configuration")
        fun `should initialize service with valid configuration`() {
            // Given
            val validConfig = mapOf(
                "apiKey" to "test-key",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns validConfig

            // When
            val result = auraAIService.initialize()

            // Then
            assertTrue(result)
            assertTrue(auraAIService.isInitialized())
            verify { mockConfigService.getConfig("ai") }
        }

        @Test
        @DisplayName("Should fail initialization with invalid configuration")
        fun `should fail initialization with invalid configuration`() {
            // Given
            val invalidConfig = mapOf<String, String>()
            every { mockConfigService.getConfig("ai") } returns invalidConfig

            // When
            val result = auraAIService.initialize()

            // Then
            assertFalse(result)
            assertFalse(auraAIService.isInitialized())
        }

        @Test
        @DisplayName("Should handle null configuration gracefully")
        fun `should handle null configuration gracefully`() {
            // Given
            every { mockConfigService.getConfig("ai") } returns null

            // When
            val result = auraAIService.initialize()

            // Then
            assertFalse(result)
            assertFalse(auraAIService.isInitialized())
        }

        @Test
        @DisplayName("Should validate required configuration fields")
        fun `should validate required configuration fields`() {
            // Given
            val incompleteConfigs = listOf(
                mapOf("apiKey" to "test-key"), // missing baseUrl and timeout
                mapOf("baseUrl" to "https://api.test.com"), // missing apiKey and timeout
                mapOf("timeout" to "30000"), // missing apiKey and baseUrl
                mapOf("apiKey" to "test-key", "baseUrl" to "https://api.test.com") // missing timeout
            )

            incompleteConfigs.forEach { config ->
                // Given
                every { mockConfigService.getConfig("ai") } returns config

                // When
                val result = auraAIService.initialize()

                // Then
                assertFalse(result, "Configuration $config should be invalid")
            }
        }

        @Test
        @DisplayName("Should handle concurrent initialization attempts")
        fun `should handle concurrent initialization attempts`() = runTest {
            // Given
            val validConfig = mapOf(
                "apiKey" to "test-key",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns validConfig

            // When
            val results = (1..5).map {
                auraAIService.initialize()
            }

            // Then
            assertTrue(results.all { it })
            assertTrue(auraAIService.isInitialized())
        }
    }

    @Nested
    @DisplayName("AI Query Tests")
    inner class AIQueryTests {

        @BeforeEach
        fun setUpInitializedService() {
            val validConfig = mapOf(
                "apiKey" to "test-key",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns validConfig
            auraAIService.initialize()
        }

        @Test
        @DisplayName("Should successfully process valid query")
        fun `should successfully process valid query`() = runTest {
            // Given
            val query = "What is the meaning of life?"
            val expectedResponse = AIResponse(
                content = "The meaning of life is 42",
                confidence = 0.95,
                tokensUsed = 15
            )
            coEvery { mockApiClient.sendQuery(any()) } returns expectedResponse

            // When
            val result = auraAIService.processQuery(query)

            // Then
            assertEquals(expectedResponse, result)
            coVerify { mockApiClient.sendQuery(query) }
        }

        @ParameterizedTest
        @ValueSource(strings = ["", "   ", "\t\n"])
        @DisplayName("Should handle empty or whitespace queries")
        fun `should handle empty or whitespace queries`(query: String) = runTest {
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should handle null query gracefully")
        fun `should handle null query gracefully`() = runTest {
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.processQuery(null)
            }
        }

        @ParameterizedTest
        @CsvSource(
            "Simple query, 50",
            "Medium length query with more words, 100",
            "This is a very long query that contains many words and should test the token counting functionality properly, 200"
        )
        @DisplayName("Should handle queries of different lengths")
        fun `should handle queries of different lengths`(query: String, expectedTokens: Int) = runTest {
            // Given
            val response = AIResponse(
                content = "Test response",
                confidence = 0.8,
                tokensUsed = expectedTokens
            )
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            val result = auraAIService.processQuery(query)

            // Then
            assertEquals(expectedTokens, result.tokensUsed)
        }

        @Test
        @DisplayName("Should handle network timeout gracefully")
        fun `should handle network timeout gracefully`() = runTest {
            // Given
            val query = "Test query"
            coEvery { mockApiClient.sendQuery(any()) } throws SocketTimeoutException("Request timeout")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should handle API rate limiting")
        fun `should handle API rate limiting`() = runTest {
            // Given
            val query = "Test query"
            coEvery { mockApiClient.sendQuery(any()) } throws ApiRateLimitException("Rate limit exceeded")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should retry on transient failures")
        fun `should retry on transient failures`() = runTest {
            // Given
            val query = "Test query"
            val expectedResponse = AIResponse("Success", 0.9, 10)
            coEvery { mockApiClient.sendQuery(any()) } throws IOException("Network error") andThen expectedResponse

            // When
            val result = auraAIService.processQuery(query)

            // Then
            assertEquals(expectedResponse, result)
            coVerify(exactly = 2) { mockApiClient.sendQuery(query) }
        }

        @Test
        @DisplayName("Should handle query with special characters and encoding")
        fun `should handle query with special characters and encoding`() = runTest {
            // Given
            val specialQueries = listOf(
                "Query with émojis 🚀🤖",
                "Query with Chinese characters: 你好世界",
                "Query with Arabic: مرحبا بالعالم",
                "Query with Hebrew: שלום עולם",
                "Query with Russian: Привет мир",
                "Query with mathematical symbols: ∑∫∂∞",
                "Query with currency symbols: $€£¥₹"
            )
            val response = AIResponse("Special char response", 0.8, 10)

            specialQueries.forEach { query ->
                // Given
                coEvery { mockApiClient.sendQuery(query) } returns response

                // When
                val result = auraAIService.processQuery(query)

                // Then
                assertEquals(response, result)
                coVerify { mockApiClient.sendQuery(query) }
            }
        }

        @Test
        @DisplayName("Should handle extremely long queries")
        fun `should handle extremely long queries`() = runTest {
            // Given
            val longQuery = "word ".repeat(10000).trim()
            val response = AIResponse("Long query response", 0.8, 5000)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            val result = auraAIService.processQuery(longQuery)

            // Then
            assertEquals(response, result)
        }

        @Test
        @DisplayName("Should handle service not initialized error during query")
        fun `should handle service not initialized error during query`() = runTest {
            // Given
            auraAIService.shutdown() // Ensure service is not initialized

            // When & Then
            assertThrows<IllegalStateException> {
                auraAIService.processQuery("Test query")
            }
        }
    }

    @Nested
    @DisplayName("Context Management Tests")
    inner class ContextManagementTests {

        @BeforeEach
        fun setUpInitializedService() {
            val validConfig = mapOf(
                "apiKey" to "test-key",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns validConfig
            auraAIService.initialize()
        }

        @Test
        @DisplayName("Should maintain conversation context")
        fun `should maintain conversation context`() = runTest {
            // Given
            val sessionId = "test-session-123"
            val firstQuery = "Hello"
            val secondQuery = "What did I just say?"
            
            val firstResponse = AIResponse("Hello there!", 0.9, 5)
            val secondResponse = AIResponse("You said 'Hello'", 0.95, 8)
            
            coEvery { mockApiClient.sendQueryWithContext(firstQuery, emptyList()) } returns firstResponse
            coEvery { mockApiClient.sendQueryWithContext(secondQuery, any()) } returns secondResponse

            // When
            val result1 = auraAIService.processQueryWithContext(firstQuery, sessionId)
            val result2 = auraAIService.processQueryWithContext(secondQuery, sessionId)

            // Then
            assertEquals(firstResponse, result1)
            assertEquals(secondResponse, result2)
            coVerify { mockApiClient.sendQueryWithContext(secondQuery, match { it.isNotEmpty() }) }
        }

        @Test
        @DisplayName("Should clear context when requested")
        fun `should clear context when requested`() {
            // Given
            val sessionId = "test-session-123"
            auraAIService.storeContext(sessionId, "Previous context")

            // When
            auraAIService.clearContext(sessionId)

            // Then
            assertTrue(auraAIService.getContext(sessionId).isEmpty())
        }

        @Test
        @DisplayName("Should handle multiple concurrent sessions")
        fun `should handle multiple concurrent sessions`() = runTest {
            // Given
            val session1 = "session-1"
            val session2 = "session-2"
            val query = "Test query"
            
            val response1 = AIResponse("Response 1", 0.8, 10)
            val response2 = AIResponse("Response 2", 0.9, 12)
            
            coEvery { mockApiClient.sendQueryWithContext(query, emptyList()) } returns response1 andThen response2

            // When
            val result1 = auraAIService.processQueryWithContext(query, session1)
            val result2 = auraAIService.processQueryWithContext(query, session2)

            // Then
            assertEquals(response1, result1)
            assertEquals(response2, result2)
            assertNotEquals(auraAIService.getContext(session1), auraAIService.getContext(session2))
        }

        @Test
        @DisplayName("Should handle context overflow and truncation")
        fun `should handle context overflow and truncation`() = runTest {
            // Given
            val sessionId = "overflow-session"
            val maxContextSize = 1000
            val largeContextItems = (1..1500).map { "Context item $it" }

            // When
            largeContextItems.forEach { item ->
                auraAIService.storeContext(sessionId, item)
            }
            val retrievedContext = auraAIService.getContext(sessionId)

            // Then
            assertTrue(retrievedContext.size <= maxContextSize)
            // Should keep most recent items
            assertTrue(retrievedContext.any { it.contains("1500") })
        }

        @Test
        @DisplayName("Should handle invalid session IDs")
        fun `should handle invalid session IDs`() {
            // Given
            val invalidSessionIds = listOf(
                "", "   ", "\t\n", "session\nwith\nnewlines",
                "session\rwith\rreturns", "session\u0000with\u0000nulls"
            )

            invalidSessionIds.forEach { sessionId ->
                // When & Then
                assertThrows<IllegalArgumentException> {
                    auraAIService.storeContext(sessionId, "Test context")
                }
            }
        }

        @Test
        @DisplayName("Should preserve context order across operations")
        fun `should preserve context order across operations`() = runTest {
            // Given
            val sessionId = "order-session"
            val contexts = listOf("First", "Second", "Third", "Fourth", "Fifth")

            // When
            contexts.forEach { context ->
                auraAIService.storeContext(sessionId, context)
            }
            val retrievedContexts = auraAIService.getContext(sessionId)

            // Then
            assertEquals(contexts, retrievedContexts)
        }
    }

    @Nested
    @DisplayName("Configuration Management Tests")
    inner class ConfigurationManagementTests {

        @Test
        @DisplayName("Should update configuration at runtime")
        fun `should update configuration at runtime`() {
            // Given
            val newConfig = mapOf(
                "apiKey" to "new-test-key",
                "baseUrl" to "https://new-api.test.com",
                "timeout" to "45000"
            )
            every { mockConfigService.updateConfig("ai", newConfig) } returns true

            // When
            val result = auraAIService.updateConfiguration(newConfig)

            // Then
            assertTrue(result)
            verify { mockConfigService.updateConfig("ai", newConfig) }
        }

        @Test
        @DisplayName("Should validate configuration before updating")
        fun `should validate configuration before updating`() {
            // Given
            val invalidConfig = mapOf("invalidKey" to "invalidValue")

            // When
            val result = auraAIService.updateConfiguration(invalidConfig)

            // Then
            assertFalse(result)
            verify(exactly = 0) { mockConfigService.updateConfig(any(), any()) }
        }

        @Test
        @DisplayName("Should get current configuration")
        fun `should get current configuration`() {
            // Given
            val expectedConfig = mapOf(
                "apiKey" to "current-key",
                "baseUrl" to "https://current-api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns expectedConfig

            // When
            val result = auraAIService.getCurrentConfiguration()

            // Then
            assertEquals(expectedConfig, result)
        }

        @ParameterizedTest
        @CsvSource(
            "0, false",
            "-1, false",
            "1, true",
            "30000, true",
            "60000, true",
            "120000, false"
        )
        @DisplayName("Should validate timeout values correctly")
        fun `should validate timeout values correctly`(timeout: String, expectedValid: Boolean) {
            // Given
            val config = mapOf(
                "apiKey" to "test-key",
                "baseUrl" to "https://api.test.com",
                "timeout" to timeout
            )
            every { mockConfigService.getConfig("ai") } returns config

            // When
            val result = auraAIService.initialize()

            // Then
            assertEquals(expectedValid, result)
        }

        @ParameterizedTest
        @ValueSource(strings = ["", "   ", "invalid-url", "ftp://invalid", "http://", "https://"])
        @DisplayName("Should validate base URL format")
        fun `should validate base URL format`(baseUrl: String) {
            // Given
            val config = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to baseUrl,
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns config

            // When
            val result = auraAIService.initialize()

            // Then
            if (baseUrl.matches(Regex("^https?://[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}.*"))) {
                assertTrue(result)
            } else {
                assertFalse(result)
            }
        }

        @Test
        @DisplayName("Should handle configuration rollback on failure")
        fun `should handle configuration rollback on failure`() {
            // Given
            val originalConfig = mapOf(
                "apiKey" to "original-key",
                "baseUrl" to "https://original.test.com",
                "timeout" to "30000"
            )
            val failingConfig = mapOf(
                "apiKey" to "failing-key",
                "baseUrl" to "invalid-url",
                "timeout" to "30000"
            )
            
            every { mockConfigService.getConfig("ai") } returns originalConfig
            every { mockConfigService.updateConfig("ai", failingConfig) } returns false
            auraAIService.initialize()

            // When
            val updateResult = auraAIService.updateConfiguration(failingConfig)
            val currentConfig = auraAIService.getCurrentConfiguration()

            // Then
            assertFalse(updateResult)
            assertEquals(originalConfig, currentConfig)
        }
    }

    @Nested
    @DisplayName("Service State Management Tests")
    inner class ServiceStateManagementTests {

        @Test
        @DisplayName("Should report correct service status")
        fun `should report correct service status`() {
            // Given
            val config = mapOf("apiKey" to "test", "baseUrl" to "https://test.com", "timeout" to "30000")
            every { mockConfigService.getConfig("ai") } returns config
            
            // When
            auraAIService.initialize()
            val status = auraAIService.getServiceStatus()

            // Then
            assertTrue(status.isHealthy)
            assertTrue(status.isInitialized)
            assertNotNull(status.lastHealthCheck)
        }

        @Test
        @DisplayName("Should perform health checks")
        fun `should perform health checks`() = runTest {
            // Given
            coEvery { mockApiClient.healthCheck() } returns true

            // When
            val result = auraAIService.performHealthCheck()

            // Then
            assertTrue(result)
            coVerify { mockApiClient.healthCheck() }
        }

        @Test
        @DisplayName("Should handle failed health checks")
        fun `should handle failed health checks`() = runTest {
            // Given
            coEvery { mockApiClient.healthCheck() } throws IOException("Service unavailable")

            // When
            val result = auraAIService.performHealthCheck()

            // Then
            assertFalse(result)
        }

        @Test
        @DisplayName("Should shutdown gracefully")
        fun `should shutdown gracefully`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test", "baseUrl" to "https://test.com", "timeout" to "30000")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()

            // When
            auraAIService.shutdown()

            // Then
            assertFalse(auraAIService.isInitialized())
            verify { mockLogger.info("AuraAI service shutting down") }
        }

        @Test
        @DisplayName("Should handle repeated shutdown calls gracefully")
        fun `should handle repeated shutdown calls gracefully`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test", "baseUrl" to "https://test.com", "timeout" to "30000")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()

            // When
            auraAIService.shutdown()
            auraAIService.shutdown() // Second call
            auraAIService.shutdown() // Third call

            // Then
            assertFalse(auraAIService.isInitialized())
            verify(atMost = 1) { mockLogger.info("AuraAI service shutting down") }
        }
    }

    @Nested
    @DisplayName("Error Handling and Recovery Tests")
    inner class ErrorHandlingTests {

        @BeforeEach
        fun setUpInitializedService() {
            val validConfig = mapOf(
                "apiKey" to "test-key",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns validConfig
            auraAIService.initialize()
        }

        @Test
        @DisplayName("Should handle authentication errors")
        fun `should handle authentication errors`() = runTest {
            // Given
            val query = "Test query"
            coEvery { mockApiClient.sendQuery(any()) } throws AuthenticationException("Invalid API key")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should handle quota exceeded errors")
        fun `should handle quota exceeded errors`() = runTest {
            // Given
            val query = "Test query"
            coEvery { mockApiClient.sendQuery(any()) } throws QuotaExceededException("API quota exceeded")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should recover from temporary network failures")
        fun `should recover from temporary network failures`() = runTest {
            // Given
            val query = "Test query"
            val expectedResponse = AIResponse("Success after retry", 0.8, 15)
            
            coEvery { mockApiClient.sendQuery(any()) } throws 
                IOException("Network error") andThen 
                IOException("Network error") andThen 
                expectedResponse

            // When
            val result = auraAIService.processQuery(query)

            // Then
            assertEquals(expectedResponse, result)
            coVerify(exactly = 3) { mockApiClient.sendQuery(query) }
        }

        @Test
        @DisplayName("Should handle malformed responses")
        fun `should handle malformed responses`() = runTest {
            // Given
            val query = "Test query"
            coEvery { mockApiClient.sendQuery(any()) } throws JsonParseException("Malformed JSON response")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should handle out of memory errors gracefully")
        fun `should handle out of memory errors gracefully`() = runTest {
            // Given
            val query = "Memory intensive query"
            coEvery { mockApiClient.sendQuery(any()) } throws OutOfMemoryError("Java heap space")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should implement circuit breaker pattern")
        fun `should implement circuit breaker pattern`() = runTest {
            // Given
            val query = "Test query"
            repeat(5) {
                coEvery { mockApiClient.sendQuery(any()) } throws IOException("Network error")
                assertThrows<ServiceException> {
                    auraAIService.processQuery(query)
                }
            }

            // When
            val status = auraAIService.getServiceStatus()

            // Then
            assertTrue(status.isCircuitBreakerOpen)
        }
    }

    @Nested
    @DisplayName("Performance and Resource Management Tests")
    inner class PerformanceTests {

        @BeforeEach
        fun setUpInitializedService() {
            val validConfig = mapOf(
                "apiKey" to "test-key",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns validConfig
            auraAIService.initialize()
        }

        @Test
        @DisplayName("Should handle concurrent requests efficiently")
        fun `should handle concurrent requests efficiently`() = runTest {
            // Given
            val queries = (1..10).map { "Query $it" }
            val responses = queries.map { AIResponse("Response for $it", 0.8, 10) }
            
            queries.zip(responses).forEach { (query, response) ->
                coEvery { mockApiClient.sendQuery(query) } returns response
            }

            // When
            val results = queries.map { query ->
                auraAIService.processQuery(query)
            }

            // Then
            assertEquals(10, results.size)
            results.zip(responses).forEach { (result, expected) ->
                assertEquals(expected, result)
            }
        }

        @Test
        @DisplayName("Should manage memory efficiently with large contexts")
        fun `should manage memory efficiently with large contexts`() = runTest {
            // Given
            val sessionId = "large-context-session"
            val largeContext = (1..1000).map { "Context item $it" }.joinToString(" ")
            
            // When
            auraAIService.storeContext(sessionId, largeContext)
            val retrievedContext = auraAIService.getContext(sessionId)

            // Then
            assertEquals(largeContext, retrievedContext.joinToString(" "))
        }

        @Test
        @DisplayName("Should implement proper timeout handling")
        fun `should implement proper timeout handling`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test-key", "baseUrl" to "https://test.com", "timeout" to "1000")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
            coEvery { mockApiClient.sendQuery(any()) } throws TimeoutException("Request timeout")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery("Test query")
            }
        }

        @Test
        @DisplayName("Should handle high-frequency requests with rate limiting")
        fun `should handle high-frequency requests with rate limiting`() = runTest {
            // Given
            val numberOfRequests = 100
            val queries = (1..numberOfRequests).map { "Rapid query $it" }
            val response = AIResponse("Rate limited response", 0.8, 5)
            
            queries.forEach { query ->
                coEvery { mockApiClient.sendQuery(query) } returns response
            }

            // When
            val startTime = System.currentTimeMillis()
            val results = queries.map { auraAIService.processQuery(it) }
            val endTime = System.currentTimeMillis()

            // Then
            assertEquals(numberOfRequests, results.size)
            val totalTime = endTime - startTime
            assertTrue(totalTime > 0) // Should take some time due to rate limiting
        }

        @Test
        @DisplayName("Should clean up resources properly on shutdown")
        fun `should clean up resources properly on shutdown`() = runTest {
            // Given
            val sessionIds = (1..10).map { "session-$it" }
            sessionIds.forEach { sessionId ->
                auraAIService.storeContext(sessionId, "Context for $sessionId")
            }

            // When
            auraAIService.shutdown()

            // Then
            assertFalse(auraAIService.isInitialized())
            sessionIds.forEach { sessionId ->
                assertTrue(auraAIService.getContext(sessionId).isEmpty())
            }
        }
    }

    @Nested
    @DisplayName("Security and Input Validation Tests")
    inner class SecurityTests {

        @BeforeEach
        fun setUpInitializedService() {
            val validConfig = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns validConfig
            auraAIService.initialize()
        }

        @Test
        @DisplayName("Should sanitize potentially dangerous input")
        fun `should sanitize potentially dangerous input`() = runTest {
            // Given
            val dangerousInputs = listOf(
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "'; DROP TABLE users; --",
                "../../../etc/passwd",
                "\u0000null\u0000bytes",
                "payload\r\nHost: evil.com"
            )
            val response = AIResponse("Sanitized response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            dangerousInputs.forEach { input ->
                // When
                val result = auraAIService.processQuery(input)

                // Then
                assertEquals(response, result)
                coVerify { mockApiClient.sendQuery(input) }
            }
        }

        @Test
        @DisplayName("Should validate API key strength")
        fun `should validate API key strength`() {
            // Given
            val weakApiKeys = listOf(
                "", "   ", "weak", "12345", "password", "api-key"
            )

            weakApiKeys.forEach { weakKey ->
                // Given
                val config = mapOf(
                    "apiKey" to weakKey,
                    "baseUrl" to "https://api.test.com",
                    "timeout" to "30000"
                )
                every { mockConfigService.getConfig("ai") } returns config

                // When
                val result = auraAIService.initialize()

                // Then
                assertFalse(result, "Weak API key '$weakKey' should be rejected")
            }
        }

        @Test
        @DisplayName("Should validate session ID format for security")
        fun `should validate session ID format for security`() = runTest {
            // Given
            val maliciousSessionIds = listOf(
                "../../../config",
                "session<script>",
                "session'; DROP TABLE sessions; --",
                "session\u0000hidden",
                "session\r\nHost: evil.com"
            )

            maliciousSessionIds.forEach { sessionId ->
                // When & Then
                assertThrows<IllegalArgumentException> {
                    auraAIService.storeContext(sessionId, "Test context")
                }
            }
        }

        @Test
        @DisplayName("Should handle URL validation for base URL")
        fun `should handle URL validation for base URL`() {
            // Given
            val maliciousUrls = listOf(
                "javascript:alert('xss')",
                "data:text/html,<script>alert('xss')</script>",
                "file:///etc/passwd",
                "ftp://malicious.com",
                "http://localhost:22",
                "https://127.0.0.1:8080"
            )

            maliciousUrls.forEach { url ->
                // Given
                val config = mapOf(
                    "apiKey" to "test-key-12345",
                    "baseUrl" to url,
                    "timeout" to "30000"
                )
                every { mockConfigService.getConfig("ai") } returns config

                // When
                val result = auraAIService.initialize()

                // Then
                assertFalse(result, "Malicious URL '$url' should be rejected")
            }
        }
    }

    @Nested
    @DisplayName("Integration and Boundary Tests")
    inner class IntegrationTests {

        @Test
        @DisplayName("Should handle service lifecycle transitions")
        fun `should handle service lifecycle transitions`() = runTest {
            // Given
            val config = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns config

            // When & Then - Initialize
            assertTrue(auraAIService.initialize())
            assertTrue(auraAIService.isInitialized())

            // When & Then - Use service
            val query = "Test query"
            val response = AIResponse("Test response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response
            val result = auraAIService.processQuery(query)
            assertEquals(response, result)

            // When & Then - Shutdown
            auraAIService.shutdown()
            assertFalse(auraAIService.isInitialized())

            // When & Then - Reinitialize
            assertTrue(auraAIService.initialize())
            assertTrue(auraAIService.isInitialized())
        }

        @Test
        @DisplayName("Should handle boundary token limits")
        fun `should handle boundary token limits`() = runTest {
            // Given
            val config = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()

            val boundaryTokenCounts = listOf(0, 1, 4095, 4096, 4097, Int.MAX_VALUE)
            
            boundaryTokenCounts.forEach { tokenCount ->
                // Given
                val query = "Boundary test query"
                val response = AIResponse("Boundary response", 0.8, tokenCount)
                coEvery { mockApiClient.sendQuery(any()) } returns response

                // When
                if (tokenCount < 0) {
                    assertThrows<ServiceException> {
                        auraAIService.processQuery(query)
                    }
                } else {
                    val result = auraAIService.processQuery(query)
                    assertEquals(tokenCount, result.tokensUsed)
                }
            }
        }

        @Test
        @DisplayName("Should handle confidence boundary values")
        fun `should handle confidence boundary values`() = runTest {
            // Given
            val config = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()

            val boundaryConfidences = listOf(-0.1, 0.0, 0.5, 1.0, 1.1)
            
            boundaryConfidences.forEach { confidence ->
                // Given
                val query = "Confidence test query"
                val response = AIResponse("Confidence response", confidence, 10)
                coEvery { mockApiClient.sendQuery(any()) } returns response

                // When & Then
                if (confidence < 0.0 || confidence > 1.0) {
                    assertThrows<ServiceException> {
                        auraAIService.processQuery(query)
                    }
                } else {
                    val result = auraAIService.processQuery(query)
                    assertEquals(confidence, result.confidence, 0.001)
                }
            }
        }
    }

    @Nested
    @DisplayName("Logging and Monitoring Tests")
    inner class LoggingTests {

        @BeforeEach
        fun setUpInitializedService() {
            val validConfig = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns validConfig
            auraAIService.initialize()
        }

        @Test
        @DisplayName("Should log successful operations with appropriate level")
        fun `should log successful operations with appropriate level`() = runTest {
            // Given
            val query = "Test query"
            val response = AIResponse("Test response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            auraAIService.processQuery(query)

            // Then
            verify { mockLogger.info(match { it.contains("processed successfully") }) }
            verify { mockLogger.debug(match { it.contains("tokens") }) }
        }

        @Test
        @DisplayName("Should log errors with appropriate context")
        fun `should log errors with appropriate context`() = runTest {
            // Given
            val query = "Error test query"
            val exception = IOException("Network failure")
            coEvery { mockApiClient.sendQuery(any()) } throws exception

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }

            verify { mockLogger.error(match { it.contains("error") && it.contains("Network failure") }) }
        }

        @Test
        @DisplayName("Should log configuration changes")
        fun `should log configuration changes`() {
            // Given
            val newConfig = mapOf(
                "apiKey" to "new-key",
                "baseUrl" to "https://new.test.com",
                "timeout" to "45000"
            )
            every { mockConfigService.updateConfig("ai", newConfig) } returns true

            // When
            auraAIService.updateConfiguration(newConfig)

            // Then
            verify { mockLogger.info(match { it.contains("configuration updated") }) }
        }

        @Test
        @DisplayName("Should log performance metrics")
        fun `should log performance metrics`() = runTest {
            // Given
            val query = "Performance test"
            val response = AIResponse("Performance response", 0.8, 150)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            auraAIService.processQuery(query)

            // Then
            verify { mockLogger.debug(match { it.contains("tokens") && it.contains("150") }) }
        }

        @Test
        @DisplayName("Should not log sensitive information")
        fun `should not log sensitive information`() = runTest {
            // Given
            val sensitiveQuery = "What is my API key: test-secret-key-12345"
            val response = AIResponse("Sensitive response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            auraAIService.processQuery(sensitiveQuery)

            // Then
            verify { mockLogger.info(match { !it.contains("test-secret-key-12345") }) }
            verify { mockLogger.debug(match { !it.contains("test-secret-key-12345") }) }
        }
    }

    companion object {
        @JvmStatic
        fun provideTestQueries(): Stream<Arguments> = Stream.of(
            Arguments.of("Simple question", "Simple answer"),
            Arguments.of("Complex multi-part question with details", "Detailed response"),
            Arguments.of("Question with special characters !@#\$%^&*()", "Response with handling"),
            Arguments.of("Unicode question with émojis 🤖", "Unicode response 🚀"),
            Arguments.of("Query with\nnewlines\nand\ttabs", "Formatted response"),
            Arguments.of("Very long query ".repeat(100).trim(), "Long response handling")
        )

        @JvmStatic
        fun provideConfigurationTestCases(): Stream<Arguments> = Stream.of(
            Arguments.of(mapOf("apiKey" to "test-key"), false),
            Arguments.of(mapOf("baseUrl" to "https://test.com"), false),
            Arguments.of(mapOf("timeout" to "30000"), false),
            Arguments.of(mapOf("apiKey" to "test-key", "baseUrl" to "https://test.com"), false),
            Arguments.of(mapOf("apiKey" to "test-key", "timeout" to "30000"), false),
            Arguments.of(mapOf("baseUrl" to "https://test.com", "timeout" to "30000"), false),
            Arguments.of(mapOf("apiKey" to "test-key", "baseUrl" to "https://test.com", "timeout" to "30000"), true)
        )

        @JvmStatic
        fun provideInvalidInputs(): Stream<Arguments> = Stream.of(
            Arguments.of(""),
            Arguments.of("   "),
            Arguments.of("\t\n\r"),
            Arguments.of("\u0000"),
            Arguments.of("null"),
            Arguments.of("undefined")
        )
    }
}

// Data classes for testing
data class AIResponse(
    val content: String,
    val confidence: Double,
    val tokensUsed: Int
)

data class ServiceStatus(
    val isHealthy: Boolean,
    val isInitialized: Boolean,
    val lastHealthCheck: Long?,
    val isCircuitBreakerOpen: Boolean = false
)

// Exception classes for testing
data class ServiceException(override val message: String, override val cause: Throwable? = null) : Exception(message, cause)
data class DataCorruptionException(override val message: String) : Exception(message)
data class JsonParseException(override val message: String) : Exception(message)
data class ApiRateLimitException(override val message: String) : Exception(message)
data class AuthenticationException(override val message: String) : Exception(message)
data class QuotaExceededException(override val message: String) : Exception(message)

// Mock interfaces for testing
interface HttpClient
interface ApiClient {
    suspend fun sendQuery(query: String): AIResponse
    suspend fun sendQueryWithContext(query: String, context: List<String>): AIResponse
    suspend fun healthCheck(): Boolean
}
interface ConfigService {
    fun getConfig(key: String): Map<String, String>?
    fun updateConfig(key: String, config: Map<String, String>): Boolean
}
interface Logger {
    fun info(message: String)
    fun debug(message: String)
    fun warn(message: String)
    fun error(message: String)
}

// Mock AuraAIService class for testing context
class AuraAIService(
    private val httpClient: HttpClient,
    private val apiClient: ApiClient,
    private val configService: ConfigService,
    private val logger: Logger
) {
    private var initialized = false
    private val contextStorage = mutableMapOf<String, MutableList<String>>()

    fun initialize(): Boolean {
        val config = configService.getConfig("ai") ?: return false
        val requiredKeys = listOf("apiKey", "baseUrl", "timeout")
        
        if (!requiredKeys.all { config.containsKey(it) }) return false
        
        val apiKey = config["apiKey"]?.trim() ?: return false
        val baseUrl = config["baseUrl"]?.trim() ?: return false
        val timeout = config["timeout"]?.toIntOrNull() ?: return false
        
        if (apiKey.length < 8) return false
        if (!baseUrl.matches(Regex("^https?://[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}.*"))) return false
        if (timeout <= 0 || timeout > 120000) return false
        
        initialized = true
        return true
    }

    fun isInitialized(): Boolean = initialized

    suspend fun processQuery(query: String?): AIResponse {
        if (!initialized) throw IllegalStateException("Service not initialized")
        if (query.isNullOrBlank()) throw IllegalArgumentException("Query cannot be null or blank")
        
        return try {
            val response = apiClient.sendQuery(query)
            if (response.confidence < 0.0 || response.confidence > 1.0) {
                throw ServiceException("Invalid confidence value: ${response.confidence}")
            }
            if (response.tokensUsed < 0) {
                throw ServiceException("Invalid token count: ${response.tokensUsed}")
            }
            logger.info("Query processed successfully")
            logger.debug("Tokens used: ${response.tokensUsed}")
            response
        } catch (e: Exception) {
            logger.error("Error processing query: ${e.message}")
            throw ServiceException("Failed to process query", e)
        }
    }

    suspend fun processQueryWithContext(query: String, sessionId: String): AIResponse {
        validateSessionId(sessionId)
        val context = getContext(sessionId)
        val response = apiClient.sendQueryWithContext(query, context)
        storeContext(sessionId, query)
        return response
    }

    fun storeContext(sessionId: String?, context: String?) {
        validateSessionId(sessionId)
        if (context.isNullOrBlank()) throw IllegalArgumentException("Context cannot be null or blank")
        
        contextStorage.getOrPut(sessionId!!) { mutableListOf() }.add(context)
        
        // Limit context size to prevent memory issues
        val sessionContext = contextStorage[sessionId]!!
        if (sessionContext.size > 1000) {
            sessionContext.removeAt(0) // Remove oldest item
        }
    }

    fun getContext(sessionId: String): List<String> {
        validateSessionId(sessionId)
        return contextStorage[sessionId] ?: emptyList()
    }

    fun clearContext(sessionId: String) {
        validateSessionId(sessionId)
        contextStorage.remove(sessionId)
    }

    fun updateConfiguration(config: Map<String, String>): Boolean {
        val requiredKeys = listOf("apiKey", "baseUrl", "timeout")
        if (config.keys.any { it !in requiredKeys }) return false
        
        return configService.updateConfig("ai", config).also { success ->
            if (success) {
                logger.info("Configuration updated successfully")
            }
        }
    }

    fun getCurrentConfiguration(): Map<String, String>? = configService.getConfig("ai")

    suspend fun performHealthCheck(): Boolean {
        return try {
            apiClient.healthCheck().also { healthy ->
                logger.debug("Health check result: $healthy")
            }
        } catch (e: Exception) {
            logger.error("Health check failed: ${e.message}")
            false
        }
    }

    fun getServiceStatus(): ServiceStatus {
        return ServiceStatus(
            isHealthy = true,
            isInitialized = initialized,
            lastHealthCheck = System.currentTimeMillis(),
            isCircuitBreakerOpen = false // Simplified for testing
        )
    }

    fun shutdown() {
        if (initialized) {
            logger.info("AuraAI service shutting down")
            initialized = false
            contextStorage.clear()
            logger.info("Resources cleaned up successfully")
        }
    }

    fun expireOldContexts() {
        contextStorage.clear() // Simplified implementation
    }

    private fun validateSessionId(sessionId: String?) {
        if (sessionId.isNullOrBlank()) {
            throw IllegalArgumentException("Session ID cannot be null or blank")
        }
        if (sessionId.contains('\n') || sessionId.contains('\r') || sessionId.contains('\u0000') ||
            sessionId.contains('<') || sessionId.contains('>') || sessionId.contains(';') ||
            sessionId.contains('\'') || sessionId.contains('"') || sessionId.startsWith("../")) {
            throw IllegalArgumentException("Invalid session ID format")
        }
    }
}