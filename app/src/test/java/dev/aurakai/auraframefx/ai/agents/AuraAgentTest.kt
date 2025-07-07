package dev.aurakai.auraframefx.ai.agents

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.assertThrows
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.MockitoAnnotations
import org.mockito.junit.jupiter.MockitoExtension
import org.junit.jupiter.api.extension.ExtendWith
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import kotlin.test.assertFailsWith

@ExtendWith(MockitoExtension::class)
@DisplayName("AuraAgent Unit Tests")
class AuraAgentTest {

    @Mock
    private lateinit var mockAIService: AIService
    
    @Mock
    private lateinit var mockLogger: Logger
    
    @Mock
    private lateinit var mockEventBus: EventBus
    
    private lateinit var auraAgent: AuraAgent
    
    @BeforeEach
    fun setUp() {
        MockitoAnnotations.openMocks(this)
        auraAgent = AuraAgent(mockAIService, mockLogger, mockEventBus)
    }
    
    @AfterEach
    fun tearDown() {
        // Clean up resources if needed
        auraAgent.shutdown()
    }

    @Nested
    @DisplayName("Initialization Tests")
    inner class InitializationTests {
        
        @Test
        @DisplayName("Should initialize with valid dependencies")
        fun shouldInitializeWithValidDependencies() {
            // Given
            val aiService = mock(AIService::class.java)
            val logger = mock(Logger::class.java)
            val eventBus = mock(EventBus::class.java)
            
            // When
            val agent = AuraAgent(aiService, logger, eventBus)
            
            // Then
            assertNotNull(agent)
            assertFalse(agent.isRunning())
        }
        
        @Test
        @DisplayName("Should throw exception when initialized with null dependencies")
        fun shouldThrowExceptionWithNullDependencies() {
            // Given/When/Then
            assertThrows<IllegalArgumentException> {
                AuraAgent(null, mockLogger, mockEventBus)
            }
            
            assertThrows<IllegalArgumentException> {
                AuraAgent(mockAIService, null, mockEventBus)
            }
            
            assertThrows<IllegalArgumentException> {
                AuraAgent(mockAIService, mockLogger, null)
            }
        }
    }

    @Nested
    @DisplayName("Agent Lifecycle Tests")
    inner class LifecycleTests {
        
        @Test
        @DisplayName("Should start agent successfully")
        fun shouldStartAgentSuccessfully() {
            // Given
            assertFalse(auraAgent.isRunning())
            
            // When
            auraAgent.start()
            
            // Then
            assertTrue(auraAgent.isRunning())
            verify(mockLogger).info("AuraAgent started successfully")
        }
        
        @Test
        @DisplayName("Should not start agent twice")
        fun shouldNotStartAgentTwice() {
            // Given
            auraAgent.start()
            assertTrue(auraAgent.isRunning())
            
            // When
            auraAgent.start()
            
            // Then
            assertTrue(auraAgent.isRunning())
            verify(mockLogger).warn("AuraAgent is already running")
        }
        
        @Test
        @DisplayName("Should stop agent successfully")
        fun shouldStopAgentSuccessfully() {
            // Given
            auraAgent.start()
            assertTrue(auraAgent.isRunning())
            
            // When
            auraAgent.stop()
            
            // Then
            assertFalse(auraAgent.isRunning())
            verify(mockLogger).info("AuraAgent stopped successfully")
        }
        
        @Test
        @DisplayName("Should handle stop when not running")
        fun shouldHandleStopWhenNotRunning() {
            // Given
            assertFalse(auraAgent.isRunning())
            
            // When
            auraAgent.stop()
            
            // Then
            assertFalse(auraAgent.isRunning())
            verify(mockLogger).warn("AuraAgent is not running")
        }
        
        @Test
        @DisplayName("Should shutdown gracefully")
        fun shouldShutdownGracefully() {
            // Given
            auraAgent.start()
            
            // When
            auraAgent.shutdown()
            
            // Then
            assertFalse(auraAgent.isRunning())
            verify(mockLogger).info("AuraAgent shutdown completed")
        }
    }

    @Nested
    @DisplayName("Message Processing Tests")
    inner class MessageProcessingTests {
        
        @Test
        @DisplayName("Should process valid message successfully")
        fun shouldProcessValidMessageSuccessfully() {
            // Given
            val message = "Test message"
            val expectedResponse = "AI response"
            `when`(mockAIService.processMessage(message)).thenReturn(CompletableFuture.completedFuture(expectedResponse))
            
            // When
            val result = auraAgent.processMessage(message)
            
            // Then
            assertNotNull(result)
            assertEquals(expectedResponse, result.get(5, TimeUnit.SECONDS))
            verify(mockAIService).processMessage(message)
        }
        
        @Test
        @DisplayName("Should handle empty message")
        fun shouldHandleEmptyMessage() {
            // Given
            val emptyMessage = ""
            
            // When/Then
            assertThrows<IllegalArgumentException> {
                auraAgent.processMessage(emptyMessage)
            }
        }
        
        @Test
        @DisplayName("Should handle null message")
        fun shouldHandleNullMessage() {
            // Given
            val nullMessage: String? = null
            
            // When/Then
            assertThrows<IllegalArgumentException> {
                auraAgent.processMessage(nullMessage)
            }
        }
        
        @Test
        @DisplayName("Should handle AI service timeout")
        fun shouldHandleAIServiceTimeout() {
            // Given
            val message = "Test message"
            val timeoutFuture = CompletableFuture<String>()
            `when`(mockAIService.processMessage(message)).thenReturn(timeoutFuture)
            
            // When
            val result = auraAgent.processMessage(message)
            
            // Then
            assertThrows<TimeoutException> {
                result.get(1, TimeUnit.SECONDS)
            }
        }
        
        @Test
        @DisplayName("Should handle AI service exception")
        fun shouldHandleAIServiceException() {
            // Given
            val message = "Test message"
            val exception = RuntimeException("AI service error")
            `when`(mockAIService.processMessage(message)).thenThrow(exception)
            
            // When/Then
            assertThrows<RuntimeException> {
                auraAgent.processMessage(message)
            }
        }
    }

    @Nested
    @DisplayName("Event Handling Tests")
    inner class EventHandlingTests {
        
        @Test
        @DisplayName("Should handle agent started event")
        fun shouldHandleAgentStartedEvent() {
            // Given
            val startEvent = AgentStartedEvent(auraAgent.getId())
            
            // When
            auraAgent.handleEvent(startEvent)
            
            // Then
            verify(mockEventBus).publish(startEvent)
            verify(mockLogger).info("Handled AgentStartedEvent for agent: ${auraAgent.getId()}")
        }
        
        @Test
        @DisplayName("Should handle agent stopped event")
        fun shouldHandleAgentStoppedEvent() {
            // Given
            val stopEvent = AgentStoppedEvent(auraAgent.getId())
            
            // When
            auraAgent.handleEvent(stopEvent)
            
            // Then
            verify(mockEventBus).publish(stopEvent)
            verify(mockLogger).info("Handled AgentStoppedEvent for agent: ${auraAgent.getId()}")
        }
        
        @Test
        @DisplayName("Should handle message processed event")
        fun shouldHandleMessageProcessedEvent() {
            // Given
            val messageEvent = MessageProcessedEvent(auraAgent.getId(), "test message", "test response")
            
            // When
            auraAgent.handleEvent(messageEvent)
            
            // Then
            verify(mockEventBus).publish(messageEvent)
            verify(mockLogger).debug("Handled MessageProcessedEvent for agent: ${auraAgent.getId()}")
        }
        
        @Test
        @DisplayName("Should handle unknown event gracefully")
        fun shouldHandleUnknownEventGracefully() {
            // Given
            val unknownEvent = UnknownEvent("test")
            
            // When
            auraAgent.handleEvent(unknownEvent)
            
            // Then
            verify(mockLogger).warn("Received unknown event type: ${unknownEvent.javaClass.simpleName}")
        }
    }

    @Nested
    @DisplayName("Configuration Tests")
    inner class ConfigurationTests {
        
        @Test
        @DisplayName("Should get default configuration")
        fun shouldGetDefaultConfiguration() {
            // When
            val config = auraAgent.getConfiguration()
            
            // Then
            assertNotNull(config)
            assertFalse(config.isEmpty())
            assertTrue(config.containsKey("agentId"))
            assertTrue(config.containsKey("maxConcurrentTasks"))
        }
        
        @Test
        @DisplayName("Should update configuration successfully")
        fun shouldUpdateConfigurationSuccessfully() {
            // Given
            val newConfig = mapOf(
                "maxConcurrentTasks" to 10,
                "timeoutSeconds" to 30
            )
            
            // When
            auraAgent.updateConfiguration(newConfig)
            
            // Then
            val updatedConfig = auraAgent.getConfiguration()
            assertEquals(10, updatedConfig["maxConcurrentTasks"])
            assertEquals(30, updatedConfig["timeoutSeconds"])
            verify(mockLogger).info("Configuration updated successfully")
        }
        
        @Test
        @DisplayName("Should validate configuration parameters")
        fun shouldValidateConfigurationParameters() {
            // Given
            val invalidConfig = mapOf(
                "maxConcurrentTasks" to -1,
                "timeoutSeconds" to 0
            )
            
            // When/Then
            assertThrows<IllegalArgumentException> {
                auraAgent.updateConfiguration(invalidConfig)
            }
        }
    }

    @Nested
    @DisplayName("Concurrency Tests")
    inner class ConcurrencyTests {
        
        @Test
        @DisplayName("Should handle multiple concurrent message processing")
        fun shouldHandleMultipleConcurrentMessageProcessing() {
            // Given
            val messages = listOf("message1", "message2", "message3")
            messages.forEach { message ->
                `when`(mockAIService.processMessage(message))
                    .thenReturn(CompletableFuture.completedFuture("response_$message"))
            }
            
            // When
            val futures = messages.map { auraAgent.processMessage(it) }
            
            // Then
            futures.forEach { future ->
                assertNotNull(future.get(5, TimeUnit.SECONDS))
            }
            
            messages.forEach { message ->
                verify(mockAIService).processMessage(message)
            }
        }
        
        @Test
        @DisplayName("Should respect max concurrent tasks limit")
        fun shouldRespectMaxConcurrentTasksLimit() {
            // Given
            val config = mapOf("maxConcurrentTasks" to 2)
            auraAgent.updateConfiguration(config)
            
            val messages = listOf("message1", "message2", "message3", "message4")
            messages.forEach { message ->
                `when`(mockAIService.processMessage(message))
                    .thenReturn(CompletableFuture.completedFuture("response_$message"))
            }
            
            // When
            val futures = messages.map { auraAgent.processMessage(it) }
            
            // Then
            // Should only process 2 concurrent tasks at a time
            assertEquals(2, auraAgent.getActiveTasks())
            
            futures.forEach { future ->
                assertNotNull(future.get(10, TimeUnit.SECONDS))
            }
        }
    }

    @Nested
    @DisplayName("Error Recovery Tests")
    inner class ErrorRecoveryTests {
        
        @Test
        @DisplayName("Should recover from AI service failure")
        fun shouldRecoverFromAIServiceFailure() {
            // Given
            val message = "test message"
            `when`(mockAIService.processMessage(message))
                .thenThrow(RuntimeException("Service unavailable"))
                .thenReturn(CompletableFuture.completedFuture("recovered response"))
            
            // When
            assertThrows<RuntimeException> {
                auraAgent.processMessage(message)
            }
            
            // Retry should succeed
            val result = auraAgent.processMessage(message)
            
            // Then
            assertEquals("recovered response", result.get(5, TimeUnit.SECONDS))
            verify(mockAIService, times(2)).processMessage(message)
        }
        
        @Test
        @DisplayName("Should handle agent restart after failure")
        fun shouldHandleAgentRestartAfterFailure() {
            // Given
            auraAgent.start()
            assertTrue(auraAgent.isRunning())
            
            // Simulate failure
            auraAgent.handleFailure(RuntimeException("Critical error"))
            
            // When
            auraAgent.restart()
            
            // Then
            assertTrue(auraAgent.isRunning())
            verify(mockLogger).info("AuraAgent restarted after failure")
        }
    }

    @Nested
    @DisplayName("Resource Management Tests")
    inner class ResourceManagementTests {
        
        @Test
        @DisplayName("Should clean up resources on shutdown")
        fun shouldCleanUpResourcesOnShutdown() {
            // Given
            auraAgent.start()
            val message = "test message"
            `when`(mockAIService.processMessage(message))
                .thenReturn(CompletableFuture.completedFuture("response"))
            
            auraAgent.processMessage(message)
            
            // When
            auraAgent.shutdown()
            
            // Then
            assertFalse(auraAgent.isRunning())
            assertEquals(0, auraAgent.getActiveTasks())
            verify(mockLogger).info("Resources cleaned up successfully")
        }
        
        @Test
        @DisplayName("Should handle resource cleanup timeout")
        fun shouldHandleResourceCleanupTimeout() {
            // Given
            auraAgent.start()
            val config = mapOf("shutdownTimeoutSeconds" to 1)
            auraAgent.updateConfiguration(config)
            
            // Simulate long-running task
            val message = "long running task"
            val longRunningFuture = CompletableFuture<String>()
            `when`(mockAIService.processMessage(message)).thenReturn(longRunningFuture)
            
            auraAgent.processMessage(message)
            
            // When
            auraAgent.shutdown()
            
            // Then
            assertFalse(auraAgent.isRunning())
            verify(mockLogger).warn("Shutdown timeout reached, forcing cleanup")
        }
    }

    @Nested
    @DisplayName("Monitoring and Metrics Tests")
    inner class MonitoringTests {
        
        @Test
        @DisplayName("Should track processed message count")
        fun shouldTrackProcessedMessageCount() {
            // Given
            val messages = listOf("msg1", "msg2", "msg3")
            messages.forEach { message ->
                `when`(mockAIService.processMessage(message))
                    .thenReturn(CompletableFuture.completedFuture("response_$message"))
            }
            
            // When
            messages.forEach { auraAgent.processMessage(it) }
            
            // Then
            assertEquals(3, auraAgent.getProcessedMessageCount())
        }
        
        @Test
        @DisplayName("Should track error count")
        fun shouldTrackErrorCount() {
            // Given
            val message = "error message"
            `when`(mockAIService.processMessage(message))
                .thenThrow(RuntimeException("Service error"))
            
            // When
            assertThrows<RuntimeException> {
                auraAgent.processMessage(message)
            }
            
            // Then
            assertEquals(1, auraAgent.getErrorCount())
        }
        
        @Test
        @DisplayName("Should provide health status")
        fun shouldProvideHealthStatus() {
            // Given
            auraAgent.start()
            
            // When
            val health = auraAgent.getHealthStatus()
            
            // Then
            assertNotNull(health)
            assertTrue(health.isHealthy())
            assertEquals("RUNNING", health.getStatus())
        }
    }

    @Nested
    @DisplayName("Edge Case Tests")
    inner class EdgeCaseTests {
        
        @Test
        @DisplayName("Should handle very long messages")
        fun shouldHandleVeryLongMessages() {
            // Given
            val longMessage = "a".repeat(10000)
            `when`(mockAIService.processMessage(longMessage))
                .thenReturn(CompletableFuture.completedFuture("processed"))
            
            // When
            val result = auraAgent.processMessage(longMessage)
            
            // Then
            assertEquals("processed", result.get(5, TimeUnit.SECONDS))
            verify(mockAIService).processMessage(longMessage)
        }
        
        @Test
        @DisplayName("Should handle special characters in messages")
        fun shouldHandleSpecialCharactersInMessages() {
            // Given
            val specialMessage = "Hello! @#$%^&*()_+-=[]{}|;':\",./<>?"
            `when`(mockAIService.processMessage(specialMessage))
                .thenReturn(CompletableFuture.completedFuture("processed special"))
            
            // When
            val result = auraAgent.processMessage(specialMessage)
            
            // Then
            assertEquals("processed special", result.get(5, TimeUnit.SECONDS))
            verify(mockAIService).processMessage(specialMessage)
        }
        
        @Test
        @DisplayName("Should handle unicode messages")
        fun shouldHandleUnicodeMessages() {
            // Given
            val unicodeMessage = "Hello 世界! 🌍 Testing émojis and åccénts"
            `when`(mockAIService.processMessage(unicodeMessage))
                .thenReturn(CompletableFuture.completedFuture("processed unicode"))
            
            // When
            val result = auraAgent.processMessage(unicodeMessage)
            
            // Then
            assertEquals("processed unicode", result.get(5, TimeUnit.SECONDS))
            verify(mockAIService).processMessage(unicodeMessage)
        }
    }
}

// Mock classes for testing
class AIService {
    fun processMessage(message: String): CompletableFuture<String> {
        return CompletableFuture.completedFuture("mock response")
    }
}

class Logger {
    fun info(message: String) {}
    fun warn(message: String) {}
    fun debug(message: String) {}
    fun error(message: String) {}
}

class EventBus {
    fun publish(event: Any) {}
}

// Mock event classes
abstract class AgentEvent(val agentId: String)
class AgentStartedEvent(agentId: String) : AgentEvent(agentId)
class AgentStoppedEvent(agentId: String) : AgentEvent(agentId)
class MessageProcessedEvent(agentId: String, val message: String, val response: String) : AgentEvent(agentId)
class UnknownEvent(val data: String)

// Mock health status class
class HealthStatus {
    fun isHealthy(): Boolean = true
    fun getStatus(): String = "RUNNING"
}