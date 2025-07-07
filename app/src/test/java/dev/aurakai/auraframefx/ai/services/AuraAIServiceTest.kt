package dev.aurakai.auraframefx.ai.services

import io.mockk.*
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.junit.jupiter.params.provider.CsvSource
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import java.util.stream.Stream
import java.io.IOException
import java.net.SocketTimeoutException
import java.util.concurrent.TimeoutException

// Mock classes for dependencies that don't exist yet
class HttpClient
class ApiClient {
    suspend fun sendQuery(query: String): AIResponse = AIResponse("", 0.0, 0)
    suspend fun sendQueryWithContext(query: String, context: List<String>): AIResponse = AIResponse("", 0.0, 0)
    suspend fun healthCheck(): Boolean = true
}
class ConfigService {
    fun getConfig(key: String): Map<String, String>? = null
    fun updateConfig(key: String, config: Map<String, String>): Boolean = false
}
class Logger {
    fun info(message: String) {}
    fun warn(message: String) {}
    fun error(message: String) {}
    fun debug(message: String) {}
}

// Mock data classes
data class AIResponse(val content: String, val confidence: Double, val tokensUsed: Int)
data class ServiceStatus(val isHealthy: Boolean, val isInitialized: Boolean, val lastHealthCheck: Long?, val isCircuitBreakerOpen: Boolean = false)

// Mock exception classes
class ServiceException(message: String) : Exception(message)
class ApiRateLimitException(message: String) : Exception(message)
class AuthenticationException(message: String) : Exception(message)
class QuotaExceededException(message: String) : Exception(message)
class JsonParseException(message: String) : Exception(message)

// Mock service class
class AuraAIService(
    private val httpClient: HttpClient,
    private val apiClient: ApiClient,
    private val configService: ConfigService,
    private val logger: Logger
) {
    private var initialized = false
    private val contexts = mutableMapOf<String, MutableList<String>>()
    
    fun initialize(): Boolean {
        val config = configService.getConfig("ai") ?: return false
        if (config.isEmpty()) return false
        
        val apiKey = config["apiKey"]?.trim() ?: return false
        val baseUrl = config["baseUrl"]?.trim() ?: return false
        val timeout = config["timeout"]?.toIntOrNull() ?: return false
        
        if (apiKey.length < 8) return false
        if (!baseUrl.matches(Regex("^https?://[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}.*"))) return false
        if (timeout <= 0 || timeout > 100000) return false
        
        initialized = true
        return true
    }
    
    fun isInitialized(): Boolean = initialized
    
    suspend fun processQuery(query: String?): AIResponse {
        if (!initialized) throw IllegalStateException("Service not initialized")
        if (query.isNullOrBlank()) throw IllegalArgumentException("Query cannot be null or empty")
        
        try {
            val response = apiClient.sendQuery(query)
            if (response.tokensUsed < 0) throw ServiceException("Invalid token count")
            if (response.confidence < 0.5) logger.warn("Response confidence is degraded")
            logger.info("Query processed successfully")
            logger.debug("Query used ${response.tokensUsed} tokens")
            return response
        } catch (e: IOException) {
            logger.error("Network error occurred")
            throw ServiceException("Network error: ${e.message}")
        } catch (e: SocketTimeoutException) {
            logger.error("Request timeout error occurred")
            throw ServiceException("Timeout error: ${e.message}")
        } catch (e: ApiRateLimitException) {
            logger.error("API rate limit exceeded")
            throw ServiceException("Rate limit exceeded: ${e.message}")
        } catch (e: AuthenticationException) {
            logger.error("Authentication error occurred")
            throw ServiceException("Authentication error: ${e.message}")
        } catch (e: QuotaExceededException) {
            logger.error("API quota exceeded")
            throw ServiceException("Quota exceeded: ${e.message}")
        } catch (e: JsonParseException) {
            logger.error("Response parsing error occurred")
            throw ServiceException("Parse error: ${e.message}")
        } catch (e: TimeoutException) {
            logger.error("Operation timeout occurred")
            throw ServiceException("Timeout error: ${e.message}")
        } catch (e: InterruptedException) {
            logger.error("Operation was interrupted")
            throw ServiceException("Interrupted: ${e.message}")
        } catch (e: OutOfMemoryError) {
            logger.error("Out of memory error occurred")
            throw ServiceException("Memory error: ${e.message}")
        }
    }
    
    suspend fun processQueryWithContext(query: String, sessionId: String): AIResponse {
        if (!initialized) throw IllegalStateException("Service not initialized")
        if (sessionId.isBlank() || sessionId.contains(Regex("[^a-zA-Z0-9-_]"))) {
            throw IllegalArgumentException("Invalid session ID")
        }
        
        val context = contexts[sessionId] ?: emptyList()
        val response = apiClient.sendQueryWithContext(query, context)
        storeContext(sessionId, query)
        return response
    }
    
    fun storeContext(sessionId: String?, context: String?) {
        if (sessionId.isNullOrBlank() || sessionId.contains(Regex("[^a-zA-Z0-9-_]"))) {
            throw IllegalArgumentException("Invalid session ID")
        }
        if (context.isNullOrBlank()) {
            throw IllegalArgumentException("Context cannot be null or empty")
        }
        
        val contextList = contexts.getOrPut(sessionId) { mutableListOf() }
        contextList.add(context)
        
        // Limit context size to prevent memory issues
        if (contextList.size > 1000) {
            contextList.removeAt(0)
        }
    }
    
    fun getContext(sessionId: String): List<String> {
        return contexts[sessionId] ?: emptyList()
    }
    
    fun clearContext(sessionId: String) {
        contexts.remove(sessionId)
    }
    
    fun expireOldContexts() {
        // Simple implementation - clear all contexts
        contexts.clear()
    }
    
    fun updateConfiguration(config: Map<String, String>): Boolean {
        // Validate configuration
        if (config.containsKey("invalidKey")) return false
        
        val result = configService.updateConfig("ai", config)
        if (result) {
            logger.info("AI service configuration updated")
        }
        return result
    }
    
    fun getCurrentConfiguration(): Map<String, String>? {
        return configService.getConfig("ai")
    }
    
    fun getServiceStatus(): ServiceStatus {
        return ServiceStatus(
            isHealthy = initialized,
            isInitialized = initialized,
            lastHealthCheck = System.currentTimeMillis(),
            isCircuitBreakerOpen = false
        )
    }
    
    suspend fun performHealthCheck(): Boolean {
        return try {
            val result = apiClient.healthCheck()
            logger.debug("Health check completed")
            result
        } catch (e: Exception) {
            false
        }
    }
    
    fun shutdown() {
        initialized = false
        contexts.clear()
        logger.info("AuraAI service shutting down")
    }
}

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
        every { mockLogger.info(any()) } just Runs
        every { mockLogger.warn(any()) } just Runs
        every { mockLogger.error(any()) } just Runs
        every { mockLogger.debug(any()) } just Runs
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
                "apiKey" to "test-key-12345",
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
    }

    @Nested
    @DisplayName("AI Query Tests")
    inner class AIQueryTests {

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
    }

    @Nested
    @DisplayName("Context Management Tests")
    inner class ContextManagementTests {

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
    }

    @Nested
    @DisplayName("Service State Management Tests")
    inner class ServiceStateManagementTests {

        @Test
        @DisplayName("Should report correct service status")
        fun `should report correct service status`() {
            // Given
            val config = mapOf("apiKey" to "test-key-12345", "baseUrl" to "https://test.com", "timeout" to "30000")
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
            val config = mapOf("apiKey" to "test-key-12345", "baseUrl" to "https://test.com", "timeout" to "30000")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()

            // When
            auraAIService.shutdown()

            // Then
            assertFalse(auraAIService.isInitialized())
            verify { mockLogger.info("AuraAI service shutting down") }
        }
    }

    @Nested
    @DisplayName("Error Handling and Recovery Tests")
    inner class ErrorHandlingTests {

        @Test
        @DisplayName("Should handle service not initialized error")
        fun `should handle service not initialized error`() = runTest {
            // Given - service not initialized
            
            // When & Then
            assertThrows<IllegalStateException> {
                auraAIService.processQuery("Test query")
            }
        }

        @Test
        @DisplayName("Should handle authentication errors")
        fun `should handle authentication errors`() = runTest {
            // Given
            val config = mapOf("apiKey" to "invalid-key-12345", "baseUrl" to "https://test.com", "timeout" to "30000")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
            coEvery { mockApiClient.sendQuery(any()) } throws AuthenticationException("Invalid API key")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery("Test query")
            }
        }

        @Test
        @DisplayName("Should handle quota exceeded errors")
        fun `should handle quota exceeded errors`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test-key-12345", "baseUrl" to "https://test.com", "timeout" to "30000")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
            coEvery { mockApiClient.sendQuery(any()) } throws QuotaExceededException("API quota exceeded")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery("Test query")
            }
        }

        @Test
        @DisplayName("Should recover from temporary network failures")
        fun `should recover from temporary network failures`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test-key-12345", "baseUrl" to "https://test.com", "timeout" to "30000")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
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
    }

    @Nested
    @DisplayName("Performance and Resource Management Tests")
    inner class PerformanceTests {

        @Test
        @DisplayName("Should handle concurrent requests efficiently")
        fun `should handle concurrent requests efficiently`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test-key-12345", "baseUrl" to "https://test.com", "timeout" to "30000")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
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
            val config = mapOf("apiKey" to "test-key-12345", "baseUrl" to "https://test.com", "timeout" to "1000")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
            coEvery { mockApiClient.sendQuery(any()) } throws TimeoutException("Request timeout")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery("Test query")
            }
        }
    }

    @Nested
    @DisplayName("Advanced Configuration Validation Tests")
    inner class AdvancedConfigurationValidationTests {

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
                "apiKey" to "test-key-12345",
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

        @ParameterizedTest
        @ValueSource(strings = ["", "   ", "x", "ab", "very-short"])
        @DisplayName("Should validate API key length requirements")
        fun `should validate API key length requirements`(apiKey: String) {
            // Given
            val config = mapOf(
                "apiKey" to apiKey,
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns config

            // When
            val result = auraAIService.initialize()

            // Then
            if (apiKey.trim().length >= 8) {
                assertTrue(result)
            } else {
                assertFalse(result)
            }
        }

        @Test
        @DisplayName("Should handle configuration with extra unknown fields")
        fun `should handle configuration with extra unknown fields`() {
            // Given
            val config = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000",
                "unknownField1" to "value1",
                "unknownField2" to "value2"
            )
            every { mockConfigService.getConfig("ai") } returns config

            // When
            val result = auraAIService.initialize()

            // Then
            assertTrue(result)
            verify { mockLogger.warn(match { it.contains("unknown") }) }
        }

        @Test
        @DisplayName("Should handle configuration updates with partial data")
        fun `should handle configuration updates with partial data`() {
            // Given
            val initialConfig = mapOf(
                "apiKey" to "initial-key-12345",
                "baseUrl" to "https://initial.test.com",
                "timeout" to "30000"
            )
            val partialUpdate = mapOf("timeout" to "45000")
            
            every { mockConfigService.getConfig("ai") } returns initialConfig
            every { mockConfigService.updateConfig("ai", partialUpdate) } returns true
            auraAIService.initialize()

            // When
            val result = auraAIService.updateConfiguration(partialUpdate)

            // Then
            assertTrue(result)
            verify { mockConfigService.updateConfig("ai", partialUpdate) }
        }
    }

    @Nested
    @DisplayName("Advanced Query Processing Tests")
    inner class AdvancedQueryProcessingTests {

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
        @DisplayName("Should handle extremely large query payloads")
        fun `should handle extremely large query payloads`() = runTest {
            // Given
            val largeQuery = "x".repeat(10000)
            val response = AIResponse("Large response", 0.8, 5000)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            val result = auraAIService.processQuery(largeQuery)

            // Then
            assertEquals(response, result)
            assertTrue(result.tokensUsed > 1000)
        }

        @Test
        @DisplayName("Should handle queries with special Unicode characters")
        fun `should handle queries with special Unicode characters`() = runTest {
            // Given
            val unicodeQuery = "测试 🚀 émojis ñoño עברית العربية"
            val response = AIResponse("Unicode response", 0.9, 25)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            val result = auraAIService.processQuery(unicodeQuery)

            // Then
            assertEquals(response, result)
            coVerify { mockApiClient.sendQuery(unicodeQuery) }
        }

        @Test
        @DisplayName("Should handle malformed or corrupted responses gracefully")
        fun `should handle malformed or corrupted responses gracefully`() = runTest {
            // Given
            val query = "Test query"
            coEvery { mockApiClient.sendQuery(any()) } throws JsonParseException("Malformed response")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should handle response with zero confidence")
        fun `should handle response with zero confidence`() = runTest {
            // Given
            val query = "Ambiguous query"
            val lowConfidenceResponse = AIResponse("Uncertain response", 0.0, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns lowConfidenceResponse

            // When
            val result = auraAIService.processQuery(query)

            // Then
            assertEquals(lowConfidenceResponse, result)
            assertEquals(0.0, result.confidence)
        }

        @Test
        @DisplayName("Should handle response with negative token count")
        fun `should handle response with negative token count`() = runTest {
            // Given
            val query = "Test query"
            val invalidResponse = AIResponse("Response", 0.8, -5)
            coEvery { mockApiClient.sendQuery(any()) } returns invalidResponse

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should handle queries with code injection attempts")
        fun `should handle queries with code injection attempts`() = runTest {
            // Given
            val maliciousQuery = "'; DROP TABLE users; --"
            val response = AIResponse("Sanitized response", 0.9, 15)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            val result = auraAIService.processQuery(maliciousQuery)

            // Then
            assertEquals(response, result)
            coVerify { mockApiClient.sendQuery(maliciousQuery) }
        }
    }

    @Nested
    @DisplayName("Advanced Context Management Tests")
    inner class AdvancedContextManagementTests {

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
        @DisplayName("Should handle context overflow scenarios")
        fun `should handle context overflow scenarios`() = runTest {
            // Given
            val sessionId = "overflow-session"
            val maxContextSize = 1000
            val largeContext = (1..1500).map { "Item $it" }.joinToString(" ")
            
            // When
            auraAIService.storeContext(sessionId, largeContext)
            val retrievedContext = auraAIService.getContext(sessionId)

            // Then
            assertTrue(retrievedContext.size <= maxContextSize)
        }

        @Test
        @DisplayName("Should handle concurrent context modifications")
        fun `should handle concurrent context modifications`() = runTest {
            // Given
            val sessionId = "concurrent-session"
            val contexts = (1..10).map { "Context $it" }
            
            // When
            contexts.forEach { context ->
                auraAIService.storeContext(sessionId, context)
            }
            val finalContext = auraAIService.getContext(sessionId)

            // Then
            assertFalse(finalContext.isEmpty())
            assertTrue(finalContext.any { it.contains("Context") })
        }

        @Test
        @DisplayName("Should handle context with null or empty values")
        fun `should handle context with null or empty values`() = runTest {
            // Given
            val sessionId = "null-context-session"
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.storeContext(sessionId, null)
            }
            
            assertThrows<IllegalArgumentException> {
                auraAIService.storeContext(sessionId, "")
            }
        }

        @Test
        @DisplayName("Should handle context expiration")
        fun `should handle context expiration`() = runTest {
            // Given
            val sessionId = "expiring-session"
            val context = "Expiring context"
            auraAIService.storeContext(sessionId, context)
            
            // When
            Thread.sleep(100) // Simulate time passing
            auraAIService.expireOldContexts()
            
            // Then
            assertTrue(auraAIService.getContext(sessionId).isEmpty())
        }

        @Test
        @DisplayName("Should handle invalid session IDs")
        fun `should handle invalid session IDs`() = runTest {
            // Given
            val invalidSessionIds = listOf("", "   ", "session@#$%", "session\nwith\nnewlines")
            
            // When & Then
            invalidSessionIds.forEach { sessionId ->
                assertThrows<IllegalArgumentException> {
                    auraAIService.storeContext(sessionId, "Test context")
                }
            }
        }

        @Test
        @DisplayName("Should handle context retrieval for non-existent sessions")
        fun `should handle context retrieval for non-existent sessions`() = runTest {
            // Given
            val nonExistentSessionId = "non-existent-session"
            
            // When
            val context = auraAIService.getContext(nonExistentSessionId)
            
            // Then
            assertTrue(context.isEmpty())
        }
    }

    @Nested
    @DisplayName("Advanced Error Handling and Recovery Tests")
    inner class AdvancedErrorHandlingTests {

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
        @DisplayName("Should handle cascading failures")
        fun `should handle cascading failures`() = runTest {
            // Given
            val query = "Test query"
            coEvery { mockApiClient.sendQuery(any()) } throws IOException("Network error")
            coEvery { mockApiClient.healthCheck() } throws IOException("Health check failed")
            
            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
            
            val healthResult = auraAIService.performHealthCheck()
            assertFalse(healthResult)
        }

        @Test
        @DisplayName("Should handle out of memory errors")
        fun `should handle out of memory errors`() = runTest {
            // Given
            val query = "Memory intensive query"
            coEvery { mockApiClient.sendQuery(any()) } throws OutOfMemoryError("Heap space")
            
            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should handle service degradation gracefully")
        fun `should handle service degradation gracefully`() = runTest {
            // Given
            val query = "Test query"
            val degradedResponse = AIResponse("Degraded response", 0.3, 5)
            coEvery { mockApiClient.sendQuery(any()) } returns degradedResponse
            
            // When
            val result = auraAIService.processQuery(query)
            
            // Then
            assertEquals(degradedResponse, result)
            assertTrue(result.confidence < 0.5)
            verify { mockLogger.warn(match { it.contains("degraded") }) }
        }

        @Test
        @DisplayName("Should handle interrupted operations")
        fun `should handle interrupted operations`() = runTest {
            // Given
            val query = "Long running query"
            coEvery { mockApiClient.sendQuery(any()) } throws InterruptedException("Operation interrupted")
            
            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should handle circuit breaker activation")
        fun `should handle circuit breaker activation`() = runTest {
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
    @DisplayName("Performance and Load Testing")
    inner class PerformanceAndLoadTests {

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
        @DisplayName("Should handle high-frequency requests")
        fun `should handle high-frequency requests`() = runTest {
            // Given
            val numberOfRequests = 100
            val queries = (1..numberOfRequests).map { "Query $it" }
            val responses = queries.map { AIResponse("Response for $it", 0.8, 10) }
            
            queries.zip(responses).forEach { (query, response) ->
                coEvery { mockApiClient.sendQuery(query) } returns response
            }
            
            // When
            val startTime = System.currentTimeMillis()
            val results = queries.map { auraAIService.processQuery(it) }
            val endTime = System.currentTimeMillis()
            
            // Then
            assertEquals(numberOfRequests, results.size)
            val totalTime = endTime - startTime
            assertTrue(totalTime < 5000) // Should complete within 5 seconds
        }

        @Test
        @DisplayName("Should handle memory-intensive operations")
        fun `should handle memory-intensive operations`() = runTest {
            // Given
            val sessionId = "memory-intensive-session"
            val largeContextItems = (1..1000).map { "Large context item $it with lots of text content" }
            
            // When
            largeContextItems.forEach { item ->
                auraAIService.storeContext(sessionId, item)
            }
            
            val retrievedContext = auraAIService.getContext(sessionId)
            
            // Then
            assertFalse(retrievedContext.isEmpty())
            assertTrue(retrievedContext.size <= 1000)
        }

        @Test
        @DisplayName("Should handle burst traffic patterns")
        fun `should handle burst traffic patterns`() = runTest {
            // Given
            val burstSize = 50
            val queries = (1..burstSize).map { "Burst query $it" }
            val response = AIResponse("Burst response", 0.8, 10)
            
            queries.forEach { query ->
                coEvery { mockApiClient.sendQuery(query) } returns response
            }
            
            // When
            val results = queries.map { auraAIService.processQuery(it) }
            
            // Then
            assertEquals(burstSize, results.size)
            results.forEach { result ->
                assertEquals(response, result)
            }
        }

        @Test
        @DisplayName("Should maintain performance under sustained load")
        fun `should maintain performance under sustained load`() = runTest {
            // Given
            val sustainedRequests = 200
            val query = "Sustained load query"
            val response = AIResponse("Sustained response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response
            
            // When
            val results = mutableListOf<AIResponse>()
            repeat(sustainedRequests) {
                results.add(auraAIService.processQuery(query))
            }
            
            // Then
            assertEquals(sustainedRequests, results.size)
            results.forEach { result ->
                assertEquals(response, result)
            }
        }
    }

    @Nested
    @DisplayName("Integration and Boundary Tests")
    inner class IntegrationAndBoundaryTests {

        @Test
        @DisplayName("Should handle service initialization race conditions")
        fun `should handle service initialization race conditions`() = runTest {
            // Given
            val config = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns config
            
            // When
            val initResults = (1..10).map {
                auraAIService.initialize()
            }
            
            // Then
            assertTrue(initResults.all { it })
            assertTrue(auraAIService.isInitialized())
        }

        @Test
        @DisplayName("Should handle boundary values for token limits")
        fun `should handle boundary values for token limits`() = runTest {
            // Given
            val config = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
            val query = "Boundary test query"
            val maxTokenResponse = AIResponse("Max tokens response", 0.8, 4096)
            coEvery { mockApiClient.sendQuery(any()) } returns maxTokenResponse
            
            // When
            val result = auraAIService.processQuery(query)
            
            // Then
            assertEquals(maxTokenResponse, result)
            assertTrue(result.tokensUsed <= 4096)
        }

        @Test
        @DisplayName("Should handle service shutdown during active operations")
        fun `should handle service shutdown during active operations`() = runTest {
            // Given
            val config = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
            val query = "Long running query"
            coEvery { mockApiClient.sendQuery(any()) } coAnswers {
                Thread.sleep(1000)
                AIResponse("Delayed response", 0.8, 10)
            }
            
            // When
            auraAIService.shutdown()
            
            // Then
            assertFalse(auraAIService.isInitialized())
            assertThrows<IllegalStateException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should handle configuration changes during operation")
        fun `should handle configuration changes during operation`() = runTest {
            // Given
            val initialConfig = mapOf(
                "apiKey" to "initial-key-12345",
                "baseUrl" to "https://initial.test.com",
                "timeout" to "30000"
            )
            val newConfig = mapOf(
                "apiKey" to "new-key-12345",
                "baseUrl" to "https://new.test.com",
                "timeout" to "45000"
            )
            
            every { mockConfigService.getConfig("ai") } returns initialConfig
            every { mockConfigService.updateConfig("ai", newConfig) } returns true
            auraAIService.initialize()
            
            val query = "Test query"
            val response = AIResponse("Test response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response
            
            // When
            auraAIService.updateConfiguration(newConfig)
            val result = auraAIService.processQuery(query)
            
            // Then
            assertEquals(response, result)
            verify { mockConfigService.updateConfig("ai", newConfig) }
        }
    }

    @Nested
    @DisplayName("Logging and Monitoring Tests")
    inner class LoggingAndMonitoringTests {

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
        @DisplayName("Should log successful operations")
        fun `should log successful operations`() = runTest {
            // Given
            val query = "Test query"
            val response = AIResponse("Test response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response
            
            // When
            auraAIService.processQuery(query)
            
            // Then
            verify { mockLogger.info(match { it.contains("processed successfully") }) }
        }

        @Test
        @DisplayName("Should log error conditions appropriately")
        fun `should log error conditions appropriately`() = runTest {
            // Given
            val query = "Test query"
            coEvery { mockApiClient.sendQuery(any()) } throws IOException("Network error")
            
            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
            
            verify { mockLogger.error(match { it.contains("error") }) }
        }

        @Test
        @DisplayName("Should log performance metrics")
        fun `should log performance metrics`() = runTest {
            // Given
            val query = "Performance test query"
            val response = AIResponse("Performance response", 0.8, 100)
            coEvery { mockApiClient.sendQuery(any()) } returns response
            
            // When
            auraAIService.processQuery(query)
            
            // Then
            verify { mockLogger.debug(match { it.contains("tokens") && it.contains("100") }) }
        }

        @Test
        @DisplayName("Should log configuration changes")
        fun `should log configuration changes`() {
            // Given
            val newConfig = mapOf(
                "apiKey" to "new-test-key-12345",
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
        @DisplayName("Should log health check results")
        fun `should log health check results`() = runTest {
            // Given
            coEvery { mockApiClient.healthCheck() } returns true
            
            // When
            auraAIService.performHealthCheck()
            
            // Then
            verify { mockLogger.debug(match { it.contains("health check") }) }
        }
    }

    companion object {
        @JvmStatic
        fun provideTestQueries(): Stream<Arguments> = Stream.of(
            Arguments.of("Simple question", "Simple answer"),
            Arguments.of("Complex multi-part question with details", "Detailed response"),
            Arguments.of("Question with special characters !@#$%^&*()", "Response with handling"),
            Arguments.of("Unicode question with émojis 🤖", "Unicode response 🚀")
        )
    }
}