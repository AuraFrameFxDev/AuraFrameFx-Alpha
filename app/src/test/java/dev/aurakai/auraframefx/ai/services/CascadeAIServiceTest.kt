package dev.aurakai.auraframefx.ai.services

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.extension.ExtendWith
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.junit.jupiter.params.provider.CsvSource
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.jupiter.MockitoExtension
import org.mockito.kotlin.*
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import com.fasterxml.jackson.core.JsonParseException
import kotlin.test.assertContains
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.test.assertFalse

@ExtendWith(MockitoExtension::class)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@DisplayName("CascadeAIService Tests")
class CascadeAIServiceTest {

    @Mock
    private lateinit var mockHttpClient: HttpClient

    @Mock
    private lateinit var mockConfigurationService: ConfigurationService

    @Mock
    private lateinit var mockLogger: Logger

    private lateinit var cascadeAIService: CascadeAIService

    @BeforeEach
    fun setUp() {
        cascadeAIService = CascadeAIService(mockHttpClient, mockConfigurationService, mockLogger)
    }

    @AfterEach
    fun tearDown() {
        reset(mockHttpClient, mockConfigurationService, mockLogger)
    }

    @Nested
    @DisplayName("Service Initialization Tests")
    inner class ServiceInitializationTests {

        @Test
        @DisplayName("Should initialize service with valid dependencies")
        fun shouldInitializeWithValidDependencies() {
            // Given
            val httpClient = mock<HttpClient>()
            val configService = mock<ConfigurationService>()
            val logger = mock<Logger>()

            // When
            val service = CascadeAIService(httpClient, configService, logger)

            // Then
            assertNotNull(service)
        }

        @Test
        @DisplayName("Should throw exception when initialized with null dependencies")
        fun shouldThrowExceptionWithNullDependencies() {
            // Given/When/Then
            assertThrows<IllegalArgumentException> {
                CascadeAIService(null, mockConfigurationService, mockLogger)
            }

            assertThrows<IllegalArgumentException> {
                CascadeAIService(mockHttpClient, null, mockLogger)
            }

            assertThrows<IllegalArgumentException> {
                CascadeAIService(mockHttpClient, mockConfigurationService, null)
            }
        }
    }

    @Nested
    @DisplayName("API Request Tests")
    inner class ApiRequestTests {

        @Test
        @DisplayName("Should make successful API request with valid parameters")
        fun shouldMakeSuccessfulApiRequest() {
            // Given
            val request = AIRequest("test prompt", "gpt-4")
            val expectedResponse = AIResponse("test response", "success")

            whenever(mockHttpClient.post(any(), any())).thenReturn(
                CompletableFuture.completedFuture(expectedResponse)
            )
            whenever(mockConfigurationService.getApiKey()).thenReturn("test-api-key")
            whenever(mockConfigurationService.getApiEndpoint()).thenReturn("https://api.test.com")

            // When
            val result = cascadeAIService.processRequest(request)

            // Then
            assertNotNull(result)
            assertEquals(expectedResponse, result.get())
            verify(mockHttpClient).post(any(), any())
            verify(mockConfigurationService).getApiKey()
        }

        @Test
        @DisplayName("Should handle API request timeout")
        fun shouldHandleApiRequestTimeout() {
            // Given
            val request = AIRequest("test prompt", "gpt-4")
            val timeoutFuture = CompletableFuture<AIResponse>()

            whenever(mockHttpClient.post(any(), any())).thenReturn(timeoutFuture)
            whenever(mockConfigurationService.getApiKey()).thenReturn("test-api-key")
            whenever(mockConfigurationService.getTimeout()).thenReturn(1000L)

            // When/Then
            assertThrows<TimeoutException> {
                cascadeAIService.processRequest(request).get(1, TimeUnit.SECONDS)
            }
        }

        @Test
        @DisplayName("Should handle API request failure")
        fun shouldHandleApiRequestFailure() {
            // Given
            val request = AIRequest("test prompt", "gpt-4")
            val failedFuture = CompletableFuture<AIResponse>()
            failedFuture.completeExceptionally(RuntimeException("API Error"))

            whenever(mockHttpClient.post(any(), any())).thenReturn(failedFuture)
            whenever(mockConfigurationService.getApiKey()).thenReturn("test-api-key")

            // When/Then
            assertThrows<RuntimeException> {
                cascadeAIService.processRequest(request).get()
            }
        }

        @ParameterizedTest
        @ValueSource(strings = ["", "   ", "\t", "\n"])
        @DisplayName("Should reject empty or whitespace-only prompts")
        fun shouldRejectEmptyPrompts(prompt: String) {
            // Given
            val request = AIRequest(prompt, "gpt-4")

            // When/Then
            assertThrows<IllegalArgumentException> {
                cascadeAIService.processRequest(request)
            }
        }

        @ParameterizedTest
        @CsvSource(
            "gpt-3.5-turbo,true",
            "gpt-4,true",
            "claude-v1,true",
            "invalid-model,false",
            "'',false"
        )
        @DisplayName("Should validate AI model names")
        fun shouldValidateAiModelNames(modelName: String, isValid: Boolean) {
            // Given
            val request = AIRequest("test prompt", modelName)

            // When/Then
            if (isValid) {
                assertDoesNotThrow {
                    cascadeAIService.validateModel(modelName)
                }
            } else {
                assertThrows<IllegalArgumentException> {
                    cascadeAIService.validateModel(modelName)
                }
            }
        }
    }

    @Nested
    @DisplayName("Configuration Tests")
    inner class ConfigurationTests {

        @Test
        @DisplayName("Should use configuration service for API settings")
        fun shouldUseConfigurationForApiSettings() {
            // Given
            val apiKey = "test-api-key"
            val endpoint = "https://api.test.com"
            val timeout = 5000L

            whenever(mockConfigurationService.getApiKey()).thenReturn(apiKey)
            whenever(mockConfigurationService.getApiEndpoint()).thenReturn(endpoint)
            whenever(mockConfigurationService.getTimeout()).thenReturn(timeout)

            // When
            val config = cascadeAIService.getConfiguration()

            // Then
            assertEquals(apiKey, config.apiKey)
            assertEquals(endpoint, config.endpoint)
            assertEquals(timeout, config.timeout)
        }

        @Test
        @DisplayName("Should handle missing API key gracefully")
        fun shouldHandleMissingApiKey() {
            // Given
            whenever(mockConfigurationService.getApiKey()).thenReturn(null)

            // When/Then
            assertThrows<IllegalStateException> {
                cascadeAIService.processRequest(AIRequest("test", "gpt-4"))
            }
        }

        @Test
        @DisplayName("Should handle invalid API endpoint")
        fun shouldHandleInvalidApiEndpoint() {
            // Given
            whenever(mockConfigurationService.getApiEndpoint()).thenReturn("invalid-url")
            whenever(mockConfigurationService.getApiKey()).thenReturn("test-key")

            // When/Then
            assertThrows<IllegalArgumentException> {
                cascadeAIService.processRequest(AIRequest("test", "gpt-4"))
            }
        }
    }

    @Nested
    @DisplayName("Cascade Logic Tests")
    inner class CascadeLogicTests {

        @Test
        @DisplayName("Should cascade to secondary service on primary failure")
        fun shouldCascadeToSecondaryServiceOnPrimaryFailure() {
            // Given
            val request = AIRequest("test prompt", "gpt-4")
            val primaryFailure = CompletableFuture<AIResponse>()
            primaryFailure.completeExceptionally(RuntimeException("Primary service failed"))

            val secondaryResponse = AIResponse("secondary response", "success")
            val secondarySuccess = CompletableFuture.completedFuture(secondaryResponse)

            whenever(mockHttpClient.post(contains("primary"), any())).thenReturn(primaryFailure)
            whenever(mockHttpClient.post(contains("secondary"), any())).thenReturn(secondarySuccess)
            whenever(mockConfigurationService.getApiKey()).thenReturn("test-key")
            whenever(mockConfigurationService.getPrimaryEndpoint()).thenReturn("https://primary.api.com")
            whenever(mockConfigurationService.getSecondaryEndpoint()).thenReturn("https://secondary.api.com")

            // When
            val result = cascadeAIService.processRequestWithCascade(request)

            // Then
            assertEquals(secondaryResponse, result.get())
            verify(mockHttpClient).post(contains("primary"), any())
            verify(mockHttpClient).post(contains("secondary"), any())
        }

        @Test
        @DisplayName("Should not cascade when primary service succeeds")
        fun shouldNotCascadeWhenPrimarySucceeds() {
            // Given
            val request = AIRequest("test prompt", "gpt-4")
            val primaryResponse = AIResponse("primary response", "success")
            val primarySuccess = CompletableFuture.completedFuture(primaryResponse)

            whenever(mockHttpClient.post(contains("primary"), any())).thenReturn(primarySuccess)
            whenever(mockConfigurationService.getApiKey()).thenReturn("test-key")
            whenever(mockConfigurationService.getPrimaryEndpoint()).thenReturn("https://primary.api.com")

            // When
            val result = cascadeAIService.processRequestWithCascade(request)

            // Then
            assertEquals(primaryResponse, result.get())
            verify(mockHttpClient, times(1)).post(contains("primary"), any())
            verify(mockHttpClient, never()).post(contains("secondary"), any())
        }

        @Test
        @DisplayName("Should fail when all cascade services fail")
        fun shouldFailWhenAllCascadeServicesFail() {
            // Given
            val request = AIRequest("test prompt", "gpt-4")
            val primaryFailure = CompletableFuture<AIResponse>()
            primaryFailure.completeExceptionally(RuntimeException("Primary failed"))

            val secondaryFailure = CompletableFuture<AIResponse>()
            secondaryFailure.completeExceptionally(RuntimeException("Secondary failed"))

            whenever(mockHttpClient.post(contains("primary"), any())).thenReturn(primaryFailure)
            whenever(mockHttpClient.post(contains("secondary"), any())).thenReturn(secondaryFailure)
            whenever(mockConfigurationService.getApiKey()).thenReturn("test-key")
            whenever(mockConfigurationService.getPrimaryEndpoint()).thenReturn("https://primary.api.com")
            whenever(mockConfigurationService.getSecondaryEndpoint()).thenReturn("https://secondary.api.com")

            // When/Then
            assertThrows<RuntimeException> {
                cascadeAIService.processRequestWithCascade(request).get()
            }
        }
    }

    @Nested
    @DisplayName("Logging Tests")
    inner class LoggingTests {

        @Test
        @DisplayName("Should log successful API requests")
        fun shouldLogSuccessfulApiRequests() {
            // Given
            val request = AIRequest("test prompt", "gpt-4")
            val response = AIResponse("test response", "success")

            whenever(mockHttpClient.post(any(), any())).thenReturn(
                CompletableFuture.completedFuture(response)
            )
            whenever(mockConfigurationService.getApiKey()).thenReturn("test-key")

            // When
            cascadeAIService.processRequest(request)

            // Then
            verify(mockLogger).info(contains("API request successful"))
        }

        @Test
        @DisplayName("Should log API request failures")
        fun shouldLogApiRequestFailures() {
            // Given
            val request = AIRequest("test prompt", "gpt-4")
            val failedFuture = CompletableFuture<AIResponse>()
            failedFuture.completeExceptionally(RuntimeException("API Error"))

            whenever(mockHttpClient.post(any(), any())).thenReturn(failedFuture)
            whenever(mockConfigurationService.getApiKey()).thenReturn("test-key")

            // When
            try {
                cascadeAIService.processRequest(request).get()
            } catch (e: Exception) {
                // Expected
            }

            // Then
            verify(mockLogger).error(contains("API request failed"), any())
        }

        @Test
        @DisplayName("Should log cascade attempts")
        fun shouldLogCascadeAttempts() {
            // Given
            val request = AIRequest("test prompt", "gpt-4")
            val primaryFailure = CompletableFuture<AIResponse>()
            primaryFailure.completeExceptionally(RuntimeException("Primary failed"))

            val secondaryResponse = AIResponse("secondary response", "success")
            val secondarySuccess = CompletableFuture.completedFuture(secondaryResponse)

            whenever(mockHttpClient.post(contains("primary"), any())).thenReturn(primaryFailure)
            whenever(mockHttpClient.post(contains("secondary"), any())).thenReturn(secondarySuccess)
            whenever(mockConfigurationService.getApiKey()).thenReturn("test-key")
            whenever(mockConfigurationService.getPrimaryEndpoint()).thenReturn("https://primary.api.com")
            whenever(mockConfigurationService.getSecondaryEndpoint()).thenReturn("https://secondary.api.com")

            // When
            cascadeAIService.processRequestWithCascade(request)

            // Then
            verify(mockLogger).warn(contains("Primary service failed, attempting cascade"))
            verify(mockLogger).info(contains("Cascade to secondary service successful"))
        }
    }

    @Nested
    @DisplayName("Edge Cases and Error Handling")
    inner class EdgeCasesAndErrorHandling {

        @Test
        @DisplayName("Should handle extremely large prompts")
        fun shouldHandleExtremelyLargePrompts() {
            // Given
            val largePrompt = "x".repeat(100000)
            val request = AIRequest(largePrompt, "gpt-4")

            whenever(mockConfigurationService.getMaxPromptLength()).thenReturn(50000)

            // When/Then
            assertThrows<IllegalArgumentException> {
                cascadeAIService.processRequest(request)
            }
        }

        @Test
        @DisplayName("Should handle special characters in prompts")
        fun shouldHandleSpecialCharactersInPrompts() {
            // Given
            val specialPrompt = "Hello 世界 🌍 \n\t\r special chars: @#$%^&*()"
            val request = AIRequest(specialPrompt, "gpt-4")
            val response = AIResponse("response", "success")

            whenever(mockHttpClient.post(any(), any())).thenReturn(
                CompletableFuture.completedFuture(response)
            )
            whenever(mockConfigurationService.getApiKey()).thenReturn("test-key")

            // When
            val result = cascadeAIService.processRequest(request)

            // Then
            assertEquals(response, result.get())
        }

        @Test
        @DisplayName("Should handle concurrent requests")
        fun shouldHandleConcurrentRequests() {
            // Given
            val request1 = AIRequest("prompt 1", "gpt-4")
            val request2 = AIRequest("prompt 2", "gpt-4")
            val response1 = AIResponse("response 1", "success")
            val response2 = AIResponse("response 2", "success")

            whenever(mockHttpClient.post(any(), any()))
                .thenReturn(CompletableFuture.completedFuture(response1))
                .thenReturn(CompletableFuture.completedFuture(response2))
            whenever(mockConfigurationService.getApiKey()).thenReturn("test-key")

            // When
            val future1 = cascadeAIService.processRequest(request1)
            val future2 = cascadeAIService.processRequest(request2)

            // Then
            assertEquals(response1, future1.get())
            assertEquals(response2, future2.get())
        }

        @Test
        @DisplayName("Should handle null responses from HTTP client")
        fun shouldHandleNullResponsesFromHttpClient() {
            // Given
            val request = AIRequest("test prompt", "gpt-4")

            whenever(mockHttpClient.post(any(), any())).thenReturn(
                CompletableFuture.completedFuture(null)
            )
            whenever(mockConfigurationService.getApiKey()).thenReturn("test-key")

            // When/Then
            assertThrows<IllegalStateException> {
                cascadeAIService.processRequest(request).get()
            }
        }

        @Test
        @DisplayName("Should handle malformed JSON responses")
        fun shouldHandleMalformedJsonResponses() {
            // Given
            val request = AIRequest("test prompt", "gpt-4")

            whenever(mockHttpClient.post(any(), any())).thenThrow(
                JsonParseException("Invalid JSON")
            )
            whenever(mockConfigurationService.getApiKey()).thenReturn("test-key")

            // When/Then
            assertThrows<JsonParseException> {
                cascadeAIService.processRequest(request).get()
            }
        }
    }

    @Nested
    @DisplayName("Performance Tests")
    inner class PerformanceTests {

        @Test
        @DisplayName("Should complete requests within timeout")
        fun shouldCompleteRequestsWithinTimeout() {
            // Given
            val request = AIRequest("test prompt", "gpt-4")
            val response = AIResponse("test response", "success")

            whenever(mockHttpClient.post(any(), any())).thenReturn(
                CompletableFuture.completedFuture(response)
            )
            whenever(mockConfigurationService.getApiKey()).thenReturn("test-key")
            whenever(mockConfigurationService.getTimeout()).thenReturn(5000L)

            // When
            val startTime = System.currentTimeMillis()
            val result = cascadeAIService.processRequest(request)
            val endTime = System.currentTimeMillis()

            // Then
            assertEquals(response, result.get())
            assertTrue(endTime - startTime < 5000)
        }

        @Test
        @DisplayName("Should handle high-frequency requests")
        fun shouldHandleHighFrequencyRequests() {
            // Given
            val requests = (1..100).map { AIRequest("prompt $it", "gpt-4") }
            val responses = requests.map { AIResponse("response for ${it.prompt}", "success") }

            whenever(mockHttpClient.post(any(), any())).thenReturn(
                *responses.map { CompletableFuture.completedFuture(it) }.toTypedArray()
            )
            whenever(mockConfigurationService.getApiKey()).thenReturn("test-key")

            // When
            val futures = requests.map { cascadeAIService.processRequest(it) }
            val results = futures.map { it.get() }

            // Then
            assertEquals(100, results.size)
            results.forEach { assertNotNull(it) }
        }
    }

    companion object {
        @JvmStatic
        fun provideInvalidInputs(): List<Arguments> {
            return listOf(
                Arguments.of(null, "gpt-4"),
                Arguments.of("", "gpt-4"),
                Arguments.of("valid prompt", null),
                Arguments.of("valid prompt", "")
            )
        }
    }
}