package dev.aurakai.auraframefx.ai.services

import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.assertThrows
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.kotlin.*
import java.io.IOException

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
        whenever(mockConfigurationService.getApiKey()).thenReturn(testApiKey)
        whenever(mockConfigurationService.getBaseUrl()).thenReturn(testBaseUrl)
        whenever(mockConfigurationService.getTimeout()).thenReturn(testTimeout)
        auraAIService = AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
    }

    @AfterEach
    fun tearDown() {
        // no-op
    }

    @Nested
    @DisplayName("Initialization Tests")
    inner class InitializationTests {
        @Test
        @DisplayName("Should initialize with valid dependencies")
        fun shouldInitializeWithValidDependencies() {
            val service = AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            assertNotNull(service)
            verify(mockConfigurationService).getApiKey()
            verify(mockConfigurationService).getBaseUrl()
            verify(mockConfigurationService).getTimeout()
        }

        @Test
        @DisplayName("Should throw exception when API key is null")
        fun shouldThrowExceptionWhenApiKeyIsNull() {
            whenever(mockConfigurationService.getApiKey()).thenReturn(null)
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }

        @Test
        @DisplayName("Should throw exception when API key is empty")
        fun shouldThrowExceptionWhenApiKeyIsEmpty() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("")
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }

        @Test
        @DisplayName("Should throw exception when base URL is invalid")
        fun shouldThrowExceptionWhenBaseUrlIsInvalid() {
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("invalid-url")
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }
    }

    @Nested
    @DisplayName("Generate Response Tests")
    inner class GenerateResponseTests {
        @Test
        @DisplayName("Should return response body on success")
        fun shouldReturnBodyOnSuccess() = runTest {
            val prompt = "Hello"
            val response = mockHttpResponse(200, "OK")
            whenever(mockHttpClient.post(prompt)).thenReturn(response)

            val result = auraAIService.generateResponse(prompt)

            assertEquals("OK", result)
            verify(mockLogger).info("Generating AI response for prompt length: ${prompt.length}")
        }

        @Test
        @DisplayName("Should throw IOException on non-200 status")
        fun shouldThrowOnNon200() = runTest {
            val prompt = "Test"
            val response = mockHttpResponse(500, "Error")
            whenever(mockHttpClient.post(prompt)).thenReturn(response)

            assertThrows<IOException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("HTTP error response: ${response.statusCode} - ${response.body}")
        }

        @Test
        @DisplayName("Should throw exception on empty prompt")
        fun shouldThrowOnEmptyPrompt() = runTest {
            assertThrows<IllegalArgumentException> {
                auraAIService.generateResponse("")
            }
        }
    }

    @Nested
    @DisplayName("Generate Batch Responses Tests")
    inner class GenerateBatchResponsesTests {
        @Test
        @DisplayName("Should return empty list for empty prompts")
        fun shouldReturnEmptyForEmptyList() = runTest {
            val results = auraAIService.generateBatchResponses(emptyList())
            assertTrue(results.isEmpty())
            verify(mockLogger).info("No prompts provided for batch processing")
        }

        @Test
        @DisplayName("Should return list with response body")
        fun shouldReturnListOnSuccess() = runTest {
            val prompts = listOf("A", "B")
            val response = mockHttpResponse(200, "OK")
            whenever(mockHttpClient.post(prompts)).thenReturn(response)

            val results = auraAIService.generateBatchResponses(prompts)

            assertEquals(listOf("OK"), results)
            verify(mockLogger).info("Generating batch AI responses for ${prompts.size} prompts")
        }
    }

    @Nested
    @DisplayName("Generate Streaming Response Tests")
    inner class GenerateStreamingResponseTests {
        @Test
        @DisplayName("Should stream response on success")
        fun shouldStreamOnSuccess() = runTest {
            val prompt = "Stream"
            val flow = flowOf("a", "b")
            whenever(mockHttpClient.postStream(prompt)).thenReturn(flow)

            val results = auraAIService.generateStreamingResponse(prompt).toList()

            assertEquals(listOf("a", "b"), results)
            verify(mockLogger).info("Starting streaming response for prompt length: ${prompt.length}")
        }

        @Test
        @DisplayName("Should throw exception on empty prompt")
        fun shouldThrowOnEmptyPrompt() = runTest {
            assertThrows<IllegalArgumentException> {
                auraAIService.generateStreamingResponse("")
            }
        }
    }

    @Nested
    @DisplayName("Configuration Update Tests")
    inner class ConfigurationUpdateTests {
        @Test
        @DisplayName("Should update API key")
        fun shouldUpdateApiKey() {
            auraAIService.updateApiKey("new-key")
            verify(mockConfigurationService).updateApiKey("new-key")
            verify(mockLogger).info("API key updated successfully")
        }

        @Test
        @DisplayName("Should throw exception on empty API key")
        fun shouldThrowOnEmptyApiKey() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateApiKey("")
            }
        }

        @Test
        @DisplayName("Should update base URL")
        fun shouldUpdateBaseUrl() {
            auraAIService.updateBaseUrl("https://new.url")
            verify(mockConfigurationService).updateBaseUrl("https://new.url")
            verify(mockLogger).info("Base URL updated successfully")
        }

        @Test
        @DisplayName("Should throw exception on invalid base URL")
        fun shouldThrowOnInvalidBaseUrl() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateBaseUrl("http://bad")
            }
        }

        @Test
        @DisplayName("Should update timeout")
        fun shouldUpdateTimeout() {
            auraAIService.updateTimeout(1000L)
            verify(mockConfigurationService).updateTimeout(1000L)
            verify(mockLogger).info("Timeout updated to 1000 ms")
        }

        @Test
        @DisplayName("Should throw exception on non-positive timeout")
        fun shouldThrowOnInvalidTimeout() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateTimeout(0L)
            }
        }
    }

    @Nested
    @DisplayName("Health Check Tests")
    inner class HealthCheckTests {
        @Test
        @DisplayName("Should return healthy on 200 status")
        fun shouldReturnHealthy() = runTest {
            val response = mockHttpResponse(200, "OK")
            whenever(mockHttpClient.get("health")).thenReturn(response)

            val result = auraAIService.healthCheck()

            assertTrue(result.isHealthy)
            assertEquals("Service is healthy", result.message)
        }

        @Test
        @DisplayName("Should return unhealthy on non-200 status")
        fun shouldReturnUnhealthyOnError() = runTest {
            val response = mockHttpResponse(500, "Fail")
            whenever(mockHttpClient.get("health")).thenReturn(response)

            val result = auraAIService.healthCheck()

            assertFalse(result.isHealthy)
            assertTrue(result.message.contains("Service is unhealthy"))
        }

        @Test
        @DisplayName("Should return unhealthy on exception")
        fun shouldReturnUnhealthyOnException() = runTest {
            whenever(mockHttpClient.get("health")).thenThrow(RuntimeException("Down"))

            val result = auraAIService.healthCheck()

            assertFalse(result.isHealthy)
            assertTrue(result.message.contains("Service is unhealthy"))
        }
    }

    @Nested
    @DisplayName("Reload Configuration Tests")
    inner class ReloadConfigurationTests {
        @Test
        @DisplayName("Should reload configuration successfully")
        fun shouldReloadSuccessfully() {
            auraAIService.reloadConfiguration()
            verify(mockConfigurationService).getApiKey()
            verify(mockConfigurationService).getBaseUrl()
            verify(mockConfigurationService).getTimeout()
            verify(mockLogger).info("Configuration reloaded successfully")
        }

        @Test
        @DisplayName("Should throw ConfigurationException on invalid config")
        fun shouldThrowOnInvalidConfig() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("")
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            verify(mockLogger).error(startsWith("Failed to reload configuration"))
        }
    }

    @Nested
    @DisplayName("Model Parameters Tests")
    inner class ModelParametersTests {
        @Test
        @DisplayName("Should update valid parameters")
        fun shouldUpdateValid() {
            val params = mapOf("temperature" to 0.5, "max_tokens" to 10)
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
            verify(mockLogger).info("Model parameters updated: $params")
        }

        @Test
        @DisplayName("Should throw exception on invalid temperature")
        fun shouldThrowOnInvalidTemperature() {
            val params = mapOf("temperature" to 1.5)
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(params)
            }
            verify(mockLogger).error("Invalid model parameters: temperature must be between 0 and 1")
        }

        @Test
        @DisplayName("Should throw exception on invalid max_tokens")
        fun shouldThrowOnInvalidMaxTokens() {
            val params = mapOf("max_tokens" to 0)
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(params)
            }
            verify(mockLogger).error("Invalid model parameters: max_tokens must be positive")
        }
    }

    @Nested
    @DisplayName("Statistics Tests")
    inner class StatisticsTests {
        @Test
        @DisplayName("Should return default statistics")
        fun shouldReturnDefaults() {
            val stats = auraAIService.getServiceStatistics()
            assertEquals(0L, stats["totalRequests"])
            assertEquals(0L, stats["successfulRequests"])
            assertEquals(0L, stats["failedRequests"])
            assertEquals(0.0, stats["averageResponseTime"])
            verify(mockLogger).debug("Service statistics requested")
        }

        @Test
        @DisplayName("Should log on reset statistics")
        fun shouldLogOnReset() {
            auraAIService.resetStatistics()
            verify(mockLogger).info("Service statistics reset")
        }
    }

    @Nested
    @DisplayName("Cache Tests")
    inner class CacheTests {
        @Test
        @DisplayName("Should clear cache")
        fun shouldClearCache() {
            auraAIService.clearCache()
            verify(mockLogger).info("Response cache cleared")
        }

        @Test
        @DisplayName("Should expire cache")
        fun shouldExpireCache() {
            auraAIService.expireCache()
            verify(mockLogger).debug("Cache expired, making new request")
        }
    }

    private fun mockHttpResponse(statusCode: Int, body: String?): HttpResponse {
        val mockResponse = mock<HttpResponse>()
        whenever(mockResponse.statusCode).thenReturn(statusCode)
        whenever(mockResponse.body).thenReturn(body)
        return mockResponse
    }
}