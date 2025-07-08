package dev.aurakai.auraframefx.ai.services

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
import org.mockito.kotlin.whenever
import org.mockito.kotlin.verify
import org.mockito.kotlin.verifyNoInteractions
import org.mockito.kotlin.any
import org.mockito.kotlin.eq
import org.mockito.kotlin.timeout
import org.mockito.kotlin.never
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.setMain
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.toList
import java.util.concurrent.TimeoutException

@ExtendWith(MockitoExtension::class)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class CascadeAIServiceTest {

    @Mock
    private lateinit var primaryAIService: AIService
    
    @Mock
    private lateinit var secondaryAIService: AIService
    
    @Mock
    private lateinit var tertiaryAIService: AIService
    
    @Mock
    private lateinit var configurationService: AIConfigurationService
    
    @Mock
    private lateinit var metricsService: AIMetricsService
    
    @Mock
    private lateinit var retryPolicy: RetryPolicy
    
    private lateinit var cascadeAIService: CascadeAIService
    
    private val testDispatcher = StandardTestDispatcher()
    
    @BeforeEach
    fun setUp() {
        MockitoAnnotations.openMocks(this)
        Dispatchers.setMain(testDispatcher)
        
        cascadeAIService = CascadeAIService(
            services = listOf(primaryAIService, secondaryAIService, tertiaryAIService),
            configurationService = configurationService,
            metricsService = metricsService,
            retryPolicy = retryPolicy
        )
    }
    
    @AfterEach
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Nested
    @DisplayName("Basic Functionality Tests")
    inner class BasicFunctionalityTests {
        
        @Test
        @DisplayName("Should successfully process request using primary service")
        fun shouldProcessRequestUsingPrimaryService() = runTest {
            // Given
            val request = AIRequest("test prompt", model = "gpt-4")
            val expectedResponse = AIResponse("test response", confidence = 0.95)
            whenever(primaryAIService.processRequest(request)).thenReturn(expectedResponse)
            
            // When
            val result = cascadeAIService.processRequest(request)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(primaryAIService).processRequest(request)
            verify(metricsService).recordServiceUsage(eq("primary"), any())
            verifyNoInteractions(secondaryAIService, tertiaryAIService)
        }
        
        @Test
        @DisplayName("Should cascade to secondary service when primary fails")
        fun shouldCascadeToSecondaryServiceWhenPrimaryFails() = runTest {
            // Given
            val request = AIRequest("test prompt", model = "gpt-4")
            val expectedResponse = AIResponse("test response", confidence = 0.85)
            whenever(primaryAIService.processRequest(request)).thenThrow(RuntimeException("Primary service failed"))
            whenever(secondaryAIService.processRequest(request)).thenReturn(expectedResponse)
            
            // When
            val result = cascadeAIService.processRequest(request)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(primaryAIService).processRequest(request)
            verify(secondaryAIService).processRequest(request)
            verify(metricsService).recordServiceFailure(eq("primary"), any())
            verify(metricsService).recordServiceUsage(eq("secondary"), any())
            verifyNoInteractions(tertiaryAIService)
        }
        
        @Test
        @DisplayName("Should cascade through all services when previous ones fail")
        fun shouldCascadeThroughAllServicesWhenPreviousOnesFail() = runTest {
            // Given
            val request = AIRequest("test prompt", model = "gpt-4")
            val expectedResponse = AIResponse("test response", confidence = 0.75)
            whenever(primaryAIService.processRequest(request)).thenThrow(RuntimeException("Primary failed"))
            whenever(secondaryAIService.processRequest(request)).thenThrow(RuntimeException("Secondary failed"))
            whenever(tertiaryAIService.processRequest(request)).thenReturn(expectedResponse)
            
            // When
            val result = cascadeAIService.processRequest(request)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(primaryAIService).processRequest(request)
            verify(secondaryAIService).processRequest(request)
            verify(tertiaryAIService).processRequest(request)
            verify(metricsService).recordServiceFailure(eq("primary"), any())
            verify(metricsService).recordServiceFailure(eq("secondary"), any())
            verify(metricsService).recordServiceUsage(eq("tertiary"), any())
        }
        
        @Test
        @DisplayName("Should throw exception when all services fail")
        fun shouldThrowExceptionWhenAllServicesFail() = runTest {
            // Given
            val request = AIRequest("test prompt", model = "gpt-4")
            val primaryException = RuntimeException("Primary failed")
            val secondaryException = RuntimeException("Secondary failed")
            val tertiaryException = RuntimeException("Tertiary failed")
            whenever(primaryAIService.processRequest(request)).thenThrow(primaryException)
            whenever(secondaryAIService.processRequest(request)).thenThrow(secondaryException)
            whenever(tertiaryAIService.processRequest(request)).thenThrow(tertiaryException)
            
            // When & Then
            val exception = assertThrows<CascadeServiceException> {
                cascadeAIService.processRequest(request)
            }
            
            assertTrue(exception.message?.contains("All cascade services failed") == true)
            verify(primaryAIService).processRequest(request)
            verify(secondaryAIService).processRequest(request)
            verify(tertiaryAIService).processRequest(request)
            verify(metricsService).recordServiceFailure(eq("primary"), any())
            verify(metricsService).recordServiceFailure(eq("secondary"), any())
            verify(metricsService).recordServiceFailure(eq("tertiary"), any())
        }
    }

    @Nested
    @DisplayName("Edge Cases and Error Handling")
    inner class EdgeCasesAndErrorHandling {
        
        @Test
        @DisplayName("Should handle null request gracefully")
        fun shouldHandleNullRequestGracefully() = runTest {
            // When & Then
            assertThrows<IllegalArgumentException> {
                cascadeAIService.processRequest(null)
            }
        }
        
        @Test
        @DisplayName("Should handle empty prompt in request")
        fun shouldHandleEmptyPromptInRequest() = runTest {
            // Given
            val request = AIRequest("", model = "gpt-4")
            val expectedResponse = AIResponse("default response", confidence = 0.5)
            whenever(primaryAIService.processRequest(request)).thenReturn(expectedResponse)
            
            // When
            val result = cascadeAIService.processRequest(request)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(primaryAIService).processRequest(request)
        }
        
        @Test
        @DisplayName("Should handle very long prompt")
        fun shouldHandleVeryLongPrompt() = runTest {
            // Given
            val longPrompt = "a".repeat(10000)
            val request = AIRequest(longPrompt, model = "gpt-4")
            val expectedResponse = AIResponse("processed long prompt", confidence = 0.8)
            whenever(primaryAIService.processRequest(request)).thenReturn(expectedResponse)
            
            // When
            val result = cascadeAIService.processRequest(request)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(primaryAIService).processRequest(request)
        }
        
        @Test
        @DisplayName("Should handle special characters in prompt")
        fun shouldHandleSpecialCharactersInPrompt() = runTest {
            // Given
            val specialPrompt = "Hello! @#$%^&*()_+{}|:<>?[]\\;'\",./"
            val request = AIRequest(specialPrompt, model = "gpt-4")
            val expectedResponse = AIResponse("processed special chars", confidence = 0.9)
            whenever(primaryAIService.processRequest(request)).thenReturn(expectedResponse)
            
            // When
            val result = cascadeAIService.processRequest(request)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(primaryAIService).processRequest(request)
        }
        
        @Test
        @DisplayName("Should handle timeout exceptions")
        fun shouldHandleTimeoutExceptions() = runTest {
            // Given
            val request = AIRequest("test prompt", model = "gpt-4")
            val expectedResponse = AIResponse("backup response", confidence = 0.7)
            whenever(primaryAIService.processRequest(request)).thenThrow(TimeoutException("Request timed out"))
            whenever(secondaryAIService.processRequest(request)).thenReturn(expectedResponse)
            
            // When
            val result = cascadeAIService.processRequest(request)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(primaryAIService).processRequest(request)
            verify(secondaryAIService).processRequest(request)
            verify(metricsService).recordServiceTimeout(eq("primary"), any())
        }
        
        @Test
        @DisplayName("Should handle interrupted exceptions")
        fun shouldHandleInterruptedExceptions() = runTest {
            // Given
            val request = AIRequest("test prompt", model = "gpt-4")
            val expectedResponse = AIResponse("backup response", confidence = 0.6)
            whenever(primaryAIService.processRequest(request)).thenThrow(InterruptedException("Thread interrupted"))
            whenever(secondaryAIService.processRequest(request)).thenReturn(expectedResponse)
            
            // When
            val result = cascadeAIService.processRequest(request)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(primaryAIService).processRequest(request)
            verify(secondaryAIService).processRequest(request)
        }
    }

    @Nested
    @DisplayName("Configuration and Retry Policy Tests")
    inner class ConfigurationAndRetryPolicyTests {
        
        @Test
        @DisplayName("Should respect retry policy for transient failures")
        fun shouldRespectRetryPolicyForTransientFailures() = runTest {
            // Given
            val request = AIRequest("test prompt", model = "gpt-4")
            val expectedResponse = AIResponse("retry success", confidence = 0.8)
            whenever(retryPolicy.shouldRetry(any(), any())).thenReturn(true, false)
            whenever(primaryAIService.processRequest(request))
                .thenThrow(RuntimeException("Transient error"))
                .thenReturn(expectedResponse)
            
            // When
            val result = cascadeAIService.processRequest(request)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(primaryAIService, timeout(1000).times(2)).processRequest(request)
            verify(retryPolicy, timeout(1000).times(2)).shouldRetry(any(), any())
        }
        
        @Test
        @DisplayName("Should use configuration service for service selection")
        fun shouldUseConfigurationServiceForServiceSelection() = runTest {
            // Given
            val request = AIRequest("test prompt", model = "gpt-4")
            val expectedResponse = AIResponse("configured response", confidence = 0.9)
            whenever(configurationService.getPreferredService(any())).thenReturn("secondary")
            whenever(secondaryAIService.processRequest(request)).thenReturn(expectedResponse)
            
            // When
            val result = cascadeAIService.processRequest(request)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(configurationService).getPreferredService(any())
            verify(secondaryAIService).processRequest(request)
            verify(primaryAIService, never()).processRequest(request)
        }
        
        @Test
        @DisplayName("Should handle configuration service failures gracefully")
        fun shouldHandleConfigurationServiceFailuresGracefully() = runTest {
            // Given
            val request = AIRequest("test prompt", model = "gpt-4")
            val expectedResponse = AIResponse("fallback response", confidence = 0.7)
            whenever(configurationService.getPreferredService(any())).thenThrow(RuntimeException("Config failed"))
            whenever(primaryAIService.processRequest(request)).thenReturn(expectedResponse)
            
            // When
            val result = cascadeAIService.processRequest(request)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(configurationService).getPreferredService(any())
            verify(primaryAIService).processRequest(request)
        }
    }

    @Nested
    @DisplayName("Streaming Operations")
    inner class StreamingOperations {
        
        @Test
        @DisplayName("Should handle streaming responses from primary service")
        fun shouldHandleStreamingResponsesFromPrimaryService() = runTest {
            // Given
            val request = AIRequest("stream prompt", model = "gpt-4", streaming = true)
            val streamResponse = flowOf(
                AIStreamChunk("Hello", isComplete = false),
                AIStreamChunk(" World", isComplete = false),
                AIStreamChunk("!", isComplete = true)
            )
            whenever(primaryAIService.processRequestStream(request)).thenReturn(streamResponse)
            
            // When
            val result = cascadeAIService.processRequestStream(request).toList()
            
            // Then
            assertEquals(3, result.size)
            assertEquals("Hello", result[0].content)
            assertEquals(" World", result[1].content)
            assertEquals("!", result[2].content)
            assertTrue(result[2].isComplete)
            verify(primaryAIService).processRequestStream(request)
        }
        
        @Test
        @DisplayName("Should cascade streaming when primary service fails")
        fun shouldCascadeStreamingWhenPrimaryServiceFails() = runTest {
            // Given
            val request = AIRequest("stream prompt", model = "gpt-4", streaming = true)
            val streamResponse = flowOf(
                AIStreamChunk("Backup", isComplete = false),
                AIStreamChunk(" Stream", isComplete = true)
            )
            whenever(primaryAIService.processRequestStream(request)).thenThrow(RuntimeException("Primary stream failed"))
            whenever(secondaryAIService.processRequestStream(request)).thenReturn(streamResponse)
            
            // When
            val result = cascadeAIService.processRequestStream(request).toList()
            
            // Then
            assertEquals(2, result.size)
            assertEquals("Backup", result[0].content)
            assertEquals(" Stream", result[1].content)
            verify(primaryAIService).processRequestStream(request)
            verify(secondaryAIService).processRequestStream(request)
        }
        
        @Test
        @DisplayName("Should handle streaming errors mid-stream")
        fun shouldHandleStreamingErrorsMidStream() = runTest {
            // Given
            val request = AIRequest("stream prompt", model = "gpt-4", streaming = true)
            val errorStream = flow {
                emit(AIStreamChunk("Start", isComplete = false))
                throw RuntimeException("Stream error")
            }
            val backupStream = flowOf(AIStreamChunk("Backup complete", isComplete = true))
            whenever(primaryAIService.processRequestStream(request)).thenReturn(errorStream)
            whenever(secondaryAIService.processRequestStream(request)).thenReturn(backupStream)
            
            // When
            val result = cascadeAIService.processRequestStream(request).toList()
            
            // Then
            assertEquals(1, result.size)
            assertEquals("Backup complete", result[0].content)
            verify(primaryAIService).processRequestStream(request)
            verify(secondaryAIService).processRequestStream(request)
        }
    }

    @Nested
    @DisplayName("Metrics and Monitoring")
    inner class MetricsAndMonitoring {
        
        @Test
        @DisplayName("Should record detailed metrics for successful requests")
        fun shouldRecordDetailedMetricsForSuccessfulRequests() = runTest {
            // Given
            val request = AIRequest("test prompt", model = "gpt-4")
            val response = AIResponse("test response", confidence = 0.95, processingTimeMs = 150)
            whenever(primaryAIService.processRequest(request)).thenReturn(response)
            
            // When
            cascadeAIService.processRequest(request)
            
            // Then
            verify(metricsService).recordServiceUsage(eq("primary"), any())
            verify(metricsService).recordLatency(eq("primary"), eq(150L))
            verify(metricsService).recordConfidenceScore(eq("primary"), eq(0.95))
        }
        
        @Test
        @DisplayName("Should record cascade metrics when services fail")
        fun shouldRecordCascadeMetricsWhenServicesFail() = runTest {
            // Given
            val request = AIRequest("test prompt", model = "gpt-4")
            val response = AIResponse("backup response", confidence = 0.8)
            whenever(primaryAIService.processRequest(request)).thenThrow(RuntimeException("Primary failed"))
            whenever(secondaryAIService.processRequest(request)).thenReturn(response)
            
            // When
            cascadeAIService.processRequest(request)
            
            // Then
            verify(metricsService).recordServiceFailure(eq("primary"), any())
            verify(metricsService).recordCascadeEvent(eq("primary"), eq("secondary"))
            verify(metricsService).recordServiceUsage(eq("secondary"), any())
        }
        
        @Test
        @DisplayName("Should record total cascade failure metrics")
        fun shouldRecordTotalCascadeFailureMetrics() = runTest {
            // Given
            val request = AIRequest("test prompt", model = "gpt-4")
            whenever(primaryAIService.processRequest(request)).thenThrow(RuntimeException("Primary failed"))
            whenever(secondaryAIService.processRequest(request)).thenThrow(RuntimeException("Secondary failed"))
            whenever(tertiaryAIService.processRequest(request)).thenThrow(RuntimeException("Tertiary failed"))
            
            // When & Then
            assertThrows<CascadeServiceException> {
                cascadeAIService.processRequest(request)
            }
            
            verify(metricsService).recordTotalCascadeFailure(any())
            verify(metricsService).recordServiceFailure(eq("primary"), any())
            verify(metricsService).recordServiceFailure(eq("secondary"), any())
            verify(metricsService).recordServiceFailure(eq("tertiary"), any())
        }
    }

    @Nested
    @DisplayName("Concurrent Operations")
    inner class ConcurrentOperations {
        
        @Test
        @DisplayName("Should handle concurrent requests safely")
        fun shouldHandleConcurrentRequestsSafely() = runTest {
            // Given
            val request1 = AIRequest("prompt 1", model = "gpt-4")
            val request2 = AIRequest("prompt 2", model = "gpt-4")
            val response1 = AIResponse("response 1", confidence = 0.9)
            val response2 = AIResponse("response 2", confidence = 0.85)
            whenever(primaryAIService.processRequest(request1)).thenReturn(response1)
            whenever(primaryAIService.processRequest(request2)).thenReturn(response2)
            
            // When
            val result1 = cascadeAIService.processRequest(request1)
            val result2 = cascadeAIService.processRequest(request2)
            
            // Then
            assertEquals(response1, result1)
            assertEquals(response2, result2)
            verify(primaryAIService).processRequest(request1)
            verify(primaryAIService).processRequest(request2)
        }
        
        @Test
        @DisplayName("Should maintain service state across concurrent operations")
        fun shouldMaintainServiceStateAcrossConcurrentOperations() = runTest {
            // Given
            val request = AIRequest("test prompt", model = "gpt-4")
            val response = AIResponse("test response", confidence = 0.9)
            whenever(primaryAIService.processRequest(request)).thenReturn(response)
            
            // When
            val results = (1..10).map { cascadeAIService.processRequest(request) }
            
            // Then
            results.forEach { assertEquals(response, it) }
            verify(primaryAIService, timeout(1000).times(10)).processRequest(request)
        }
    }

    @Nested
    @DisplayName("Circuit Breaker Pattern")
    inner class CircuitBreakerPattern {
        
        @Test
        @DisplayName("Should implement circuit breaker for failing services")
        fun shouldImplementCircuitBreakerForFailingServices() = runTest {
            // Given
            val request = AIRequest("test prompt", model = "gpt-4")
            val response = AIResponse("backup response", confidence = 0.8)
            whenever(primaryAIService.processRequest(request)).thenThrow(RuntimeException("Service down"))
            whenever(secondaryAIService.processRequest(request)).thenReturn(response)
            whenever(configurationService.isCircuitBreakerEnabled()).thenReturn(true)
            whenever(configurationService.getCircuitBreakerThreshold()).thenReturn(3)
            
            // When - Make multiple requests to trigger circuit breaker
            repeat(5) {
                cascadeAIService.processRequest(request)
            }
            
            // Then
            verify(primaryAIService, timeout(1000).times(3)).processRequest(request) // Should stop after threshold
            verify(secondaryAIService, timeout(1000).atLeast(3)).processRequest(request)
            verify(metricsService).recordCircuitBreakerTripped(eq("primary"))
        }
        
        @Test
        @DisplayName("Should reset circuit breaker after recovery period")
        fun shouldResetCircuitBreakerAfterRecoveryPeriod() = runTest {
            // Given
            val request = AIRequest("test prompt", model = "gpt-4")
            val response = AIResponse("recovered response", confidence = 0.9)
            whenever(primaryAIService.processRequest(request))
                .thenThrow(RuntimeException("Service down"))
                .thenThrow(RuntimeException("Service down"))
                .thenThrow(RuntimeException("Service down"))
                .thenReturn(response) // Service recovers
            whenever(configurationService.isCircuitBreakerEnabled()).thenReturn(true)
            whenever(configurationService.getCircuitBreakerRecoveryTimeMs()).thenReturn(1000L)
            
            // When
            repeat(3) {
                try {
                    cascadeAIService.processRequest(request)
                } catch (e: Exception) {
                    // Expected during circuit breaker period
                }
            }
            
            // Simulate recovery period
            delay(1100)
            val result = cascadeAIService.processRequest(request)
            
            // Then
            assertEquals(response, result)
            verify(metricsService).recordCircuitBreakerReset(eq("primary"))
        }
    }
}