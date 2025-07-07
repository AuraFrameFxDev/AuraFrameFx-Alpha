package dev.aurakai.auraframefx.ai.agents

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.extension.ExtendWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.jupiter.MockitoExtension
import org.mockito.kotlin.whenever
import org.mockito.kotlin.verify
import org.mockito.kotlin.any
import org.mockito.kotlin.never
import org.mockito.kotlin.times
import org.mockito.kotlin.eq
import org.mockito.kotlin.argumentCaptor
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestCoroutineScheduler
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.setMain
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.test.resetMain
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.test.assertFailsWith

@ExperimentalCoroutinesApi
@ExtendWith(MockitoExtension::class)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class AuraAgentTest {

    @Mock
    private lateinit var mockAgentContext: AgentContext
    
    @Mock
    private lateinit var mockMessageHandler: MessageHandler
    
    @Mock
    private lateinit var mockEventBus: EventBus
    
    @Mock
    private lateinit var mockConfigurationProvider: ConfigurationProvider
    
    private lateinit var auraAgent: AuraAgent
    private val testDispatcher = StandardTestDispatcher(TestCoroutineScheduler())

    @BeforeEach
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
        
        // Set up default mock behaviors
        whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(
            AgentConfiguration(
                name = "TestAgent",
                version = "1.0.0",
                capabilities = listOf("CHAT", "ANALYSIS"),
                maxConcurrentTasks = 5
            )
        )
        
        whenever(mockAgentContext.getMessageHandler()).thenReturn(mockMessageHandler)
        whenever(mockAgentContext.getEventBus()).thenReturn(mockEventBus)
        whenever(mockAgentContext.getConfigurationProvider()).thenReturn(mockConfigurationProvider)
        
        auraAgent = AuraAgent(mockAgentContext)
    }

    @AfterEach
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Nested
    @DisplayName("Agent Initialization Tests")
    inner class InitializationTests {

        @Test
        @DisplayName("Should initialize successfully with valid context")
        fun shouldInitializeSuccessfullyWithValidContext() {
            // Given
            val validContext = mock<AgentContext>()
            whenever(validContext.getConfigurationProvider()).thenReturn(mockConfigurationProvider)
            whenever(validContext.getMessageHandler()).thenReturn(mockMessageHandler)
            whenever(validContext.getEventBus()).thenReturn(mockEventBus)
            
            // When
            val agent = AuraAgent(validContext)
            
            // Then
            assertNotNull(agent)
            assertEquals("TestAgent", agent.getName())
            assertEquals("1.0.0", agent.getVersion())
            assertTrue(agent.isInitialized())
        }

        @Test
        @DisplayName("Should throw exception when context is null")
        fun shouldThrowExceptionWhenContextIsNull() {
            // When & Then
            assertFailsWith<IllegalArgumentException> {
                AuraAgent(null)
            }
        }

        @Test
        @DisplayName("Should throw exception when configuration provider is null")
        fun shouldThrowExceptionWhenConfigurationProviderIsNull() {
            // Given
            val invalidContext = mock<AgentContext>()
            whenever(invalidContext.getConfigurationProvider()).thenReturn(null)
            
            // When & Then
            assertFailsWith<IllegalStateException> {
                AuraAgent(invalidContext)
            }
        }

        @Test
        @DisplayName("Should initialize with default configuration when config is missing")
        fun shouldInitializeWithDefaultConfigurationWhenConfigIsMissing() {
            // Given
            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(null)
            
            // When
            val agent = AuraAgent(mockAgentContext)
            
            // Then
            assertNotNull(agent)
            assertEquals("AuraAgent", agent.getName())
            assertEquals("1.0.0", agent.getVersion())
        }
    }

    @Nested
    @DisplayName("Message Processing Tests")
    inner class MessageProcessingTests {

        @Test
        @DisplayName("Should process simple text message successfully")
        fun shouldProcessSimpleTextMessageSuccessfully() = runTest {
            // Given
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Hello, AuraAgent!",
                timestamp = System.currentTimeMillis()
            )
            
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Hello! How can I help you today?",
                    status = ResponseStatus.SUCCESS
                )
            )
            
            // When
            val response = auraAgent.processMessage(message)
            
            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            assertEquals("Hello! How can I help you today?", response.content)
            verify(mockMessageHandler).validateMessage(message)
            verify(mockMessageHandler).processMessage(message)
        }

        @Test
        @DisplayName("Should handle invalid message gracefully")
        fun shouldHandleInvalidMessageGracefully() = runTest {
            // Given
            val invalidMessage = AgentMessage(
                id = "",
                type = MessageType.TEXT,
                content = "",
                timestamp = 0
            )
            
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)
            
            // When
            val response = auraAgent.processMessage(invalidMessage)
            
            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
            assertTrue(response.content.contains("Invalid message"))
            verify(mockMessageHandler).validateMessage(invalidMessage)
            verify(mockMessageHandler, never()).processMessage(any())
        }

        @Test
        @DisplayName("Should handle message processing exceptions")
        fun shouldHandleMessageProcessingExceptions() = runTest {
            // Given
            val message = AgentMessage(
                id = "msg-002",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )
            
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenThrow(
                RuntimeException("Processing failed")
            )
            
            // When
            val response = auraAgent.processMessage(message)
            
            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.ERROR, response.status)
            assertTrue(response.content.contains("Processing failed"))
        }

        @Test
        @DisplayName("Should handle concurrent message processing")
        fun shouldHandleConcurrentMessageProcessing() = runTest {
            // Given
            val messages = (1..10).map { index ->
                AgentMessage(
                    id = "msg-$index",
                    type = MessageType.TEXT,
                    content = "Message $index",
                    timestamp = System.currentTimeMillis()
                )
            }
            
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = "test",
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            )
            
            // When
            val responses = messages.map { message ->
                auraAgent.processMessage(message)
            }
            
            // Then
            assertEquals(10, responses.size)
            responses.forEach { response ->
                assertEquals(ResponseStatus.SUCCESS, response.status)
            }
            verify(mockMessageHandler, times(10)).validateMessage(any())
            verify(mockMessageHandler, times(10)).processMessage(any())
        }

        @Test
        @DisplayName("Should respect maximum concurrent tasks limit")
        fun shouldRespectMaximumConcurrentTasksLimit() = runTest {
            // Given
            val config = AgentConfiguration(
                name = "TestAgent",
                version = "1.0.0",
                capabilities = listOf("CHAT"),
                maxConcurrentTasks = 2
            )
            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(config)
            
            val agent = AuraAgent(mockAgentContext)
            val latch = CountDownLatch(2)
            
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                latch.countDown()
                latch.await(1, TimeUnit.SECONDS)
                AgentResponse(
                    messageId = "test",
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            }
            
            // When
            val messages = (1..5).map { index ->
                AgentMessage(
                    id = "msg-$index",
                    type = MessageType.TEXT,
                    content = "Message $index",
                    timestamp = System.currentTimeMillis()
                )
            }
            
            // Then
            val responses = messages.map { message ->
                agent.processMessage(message)
            }
            
            assertEquals(5, responses.size)
            // Verify that not all messages were processed simultaneously
            assertTrue(agent.getActiveTaskCount() <= 2)
        }
    }

    @Nested
    @DisplayName("Event Handling Tests")
    inner class EventHandlingTests {

        @Test
        @DisplayName("Should publish event when message is processed")
        fun shouldPublishEventWhenMessageIsProcessed() = runTest {
            // Given
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )
            
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Response",
                    status = ResponseStatus.SUCCESS
                )
            )
            
            // When
            auraAgent.processMessage(message)
            
            // Then
            val eventCaptor = argumentCaptor<AgentEvent>()
            verify(mockEventBus).publish(eventCaptor.capture())
            
            val publishedEvent = eventCaptor.firstValue
            assertEquals(EventType.MESSAGE_PROCESSED, publishedEvent.type)
            assertEquals(message.id, publishedEvent.data["messageId"])
        }

        @Test
        @DisplayName("Should handle event publishing failures gracefully")
        fun shouldHandleEventPublishingFailuresGracefully() = runTest {
            // Given
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )
            
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Response",
                    status = ResponseStatus.SUCCESS
                )
            )
            
            doThrow(RuntimeException("Event publishing failed"))
                .whenever(mockEventBus).publish(any())
            
            // When
            val response = auraAgent.processMessage(message)
            
            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            // Processing should succeed even if event publishing fails
        }

        @Test
        @DisplayName("Should register event listeners during initialization")
        fun shouldRegisterEventListenersDuringInitialization() {
            // Given
            val context = mock<AgentContext>()
            whenever(context.getConfigurationProvider()).thenReturn(mockConfigurationProvider)
            whenever(context.getMessageHandler()).thenReturn(mockMessageHandler)
            whenever(context.getEventBus()).thenReturn(mockEventBus)
            
            // When
            AuraAgent(context)
            
            // Then
            verify(mockEventBus).subscribe(eq(EventType.SYSTEM_SHUTDOWN), any())
            verify(mockEventBus).subscribe(eq(EventType.CONFIGURATION_CHANGED), any())
        }
    }

    @Nested
    @DisplayName("Configuration Management Tests")
    inner class ConfigurationManagementTests {

        @Test
        @DisplayName("Should reload configuration when configuration changed event is received")
        fun shouldReloadConfigurationWhenConfigurationChangedEventIsReceived() = runTest {
            // Given
            val newConfig = AgentConfiguration(
                name = "UpdatedAgent",
                version = "2.0.0",
                capabilities = listOf("CHAT", "ANALYSIS", "TRANSLATION"),
                maxConcurrentTasks = 10
            )
            
            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(newConfig)
            
            // When
            auraAgent.handleConfigurationChanged()
            
            // Then
            assertEquals("UpdatedAgent", auraAgent.getName())
            assertEquals("2.0.0", auraAgent.getVersion())
            assertEquals(3, auraAgent.getCapabilities().size)
            assertEquals(10, auraAgent.getMaxConcurrentTasks())
        }

        @Test
        @DisplayName("Should handle configuration reload failures gracefully")
        fun shouldHandleConfigurationReloadFailuresGracefully() = runTest {
            // Given
            whenever(mockConfigurationProvider.getAgentConfiguration())
                .thenThrow(RuntimeException("Config load failed"))
            
            // When
            auraAgent.handleConfigurationChanged()
            
            // Then
            // Agent should continue with previous configuration
            assertEquals("TestAgent", auraAgent.getName())
            assertEquals("1.0.0", auraAgent.getVersion())
        }

        @Test
        @DisplayName("Should validate configuration before applying changes")
        fun shouldValidateConfigurationBeforeApplyingChanges() = runTest {
            // Given
            val invalidConfig = AgentConfiguration(
                name = "",
                version = "",
                capabilities = emptyList(),
                maxConcurrentTasks = -1
            )
            
            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(invalidConfig)
            
            // When
            auraAgent.handleConfigurationChanged()
            
            // Then
            // Should not apply invalid configuration
            assertEquals("TestAgent", auraAgent.getName())
            assertEquals("1.0.0", auraAgent.getVersion())
        }
    }

    @Nested
    @DisplayName("Capability Management Tests")
    inner class CapabilityManagementTests {

        @Test
        @DisplayName("Should return true when agent has requested capability")
        fun shouldReturnTrueWhenAgentHasRequestedCapability() {
            // Given
            val capability = "CHAT"
            
            // When
            val hasCapability = auraAgent.hasCapability(capability)
            
            // Then
            assertTrue(hasCapability)
        }

        @Test
        @DisplayName("Should return false when agent does not have requested capability")
        fun shouldReturnFalseWhenAgentDoesNotHaveRequestedCapability() {
            // Given
            val capability = "TRANSLATION"
            
            // When
            val hasCapability = auraAgent.hasCapability(capability)
            
            // Then
            assertFalse(hasCapability)
        }

        @Test
        @DisplayName("Should handle null capability gracefully")
        fun shouldHandleNullCapabilityGracefully() {
            // When
            val hasCapability = auraAgent.hasCapability(null)
            
            // Then
            assertFalse(hasCapability)
        }

        @Test
        @DisplayName("Should handle empty capability string gracefully")
        fun shouldHandleEmptyCapabilityStringGracefully() {
            // Given
            val capability = ""
            
            // When
            val hasCapability = auraAgent.hasCapability(capability)
            
            // Then
            assertFalse(hasCapability)
        }

        @Test
        @DisplayName("Should return all configured capabilities")
        fun shouldReturnAllConfiguredCapabilities() {
            // When
            val capabilities = auraAgent.getCapabilities()
            
            // Then
            assertEquals(2, capabilities.size)
            assertTrue(capabilities.contains("CHAT"))
            assertTrue(capabilities.contains("ANALYSIS"))
        }
    }

    @Nested
    @DisplayName("Lifecycle Management Tests")
    inner class LifecycleManagementTests {

        @Test
        @DisplayName("Should start agent successfully")
        fun shouldStartAgentSuccessfully() = runTest {
            // Given
            assertFalse(auraAgent.isRunning())
            
            // When
            auraAgent.start()
            
            // Then
            assertTrue(auraAgent.isRunning())
            verify(mockEventBus).publish(argThat { event ->
                event.type == EventType.AGENT_STARTED
            })
        }

        @Test
        @DisplayName("Should stop agent successfully")
        fun shouldStopAgentSuccessfully() = runTest {
            // Given
            auraAgent.start()
            assertTrue(auraAgent.isRunning())
            
            // When
            auraAgent.stop()
            
            // Then
            assertFalse(auraAgent.isRunning())
            verify(mockEventBus).publish(argThat { event ->
                event.type == EventType.AGENT_STOPPED
            })
        }

        @Test
        @DisplayName("Should handle multiple start calls gracefully")
        fun shouldHandleMultipleStartCallsGracefully() = runTest {
            // Given
            auraAgent.start()
            assertTrue(auraAgent.isRunning())
            
            // When
            auraAgent.start()
            
            // Then
            assertTrue(auraAgent.isRunning())
            // Should only publish one start event
            verify(mockEventBus, times(1)).publish(argThat { event ->
                event.type == EventType.AGENT_STARTED
            })
        }

        @Test
        @DisplayName("Should handle multiple stop calls gracefully")
        fun shouldHandleMultipleStopCallsGracefully() = runTest {
            // Given
            auraAgent.start()
            auraAgent.stop()
            assertFalse(auraAgent.isRunning())
            
            // When
            auraAgent.stop()
            
            // Then
            assertFalse(auraAgent.isRunning())
            // Should only publish one stop event
            verify(mockEventBus, times(1)).publish(argThat { event ->
                event.type == EventType.AGENT_STOPPED
            })
        }

        @Test
        @DisplayName("Should handle shutdown gracefully")
        fun shouldHandleShutdownGracefully() = runTest {
            // Given
            auraAgent.start()
            assertTrue(auraAgent.isRunning())
            
            // When
            auraAgent.shutdown()
            
            // Then
            assertFalse(auraAgent.isRunning())
            assertTrue(auraAgent.isShutdown())
            verify(mockEventBus).publish(argThat { event ->
                event.type == EventType.AGENT_SHUTDOWN
            })
        }

        @Test
        @DisplayName("Should reject new messages after shutdown")
        fun shouldRejectNewMessagesAfterShutdown() = runTest {
            // Given
            auraAgent.start()
            auraAgent.shutdown()
            
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )
            
            // When
            val response = auraAgent.processMessage(message)
            
            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.AGENT_SHUTDOWN, response.status)
            verify(mockMessageHandler, never()).processMessage(any())
        }
    }

    @Nested
    @DisplayName("Error Handling and Recovery Tests")
    inner class ErrorHandlingAndRecoveryTests {

        @Test
        @DisplayName("Should recover from temporary message handler failures")
        fun shouldRecoverFromTemporaryMessageHandlerFailures() = runTest {
            // Given
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )
            
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any()))
                .thenThrow(RuntimeException("Temporary failure"))
                .thenReturn(AgentResponse(
                    messageId = message.id,
                    content = "Recovered response",
                    status = ResponseStatus.SUCCESS
                ))
            
            // When
            val firstResponse = auraAgent.processMessage(message)
            val secondResponse = auraAgent.processMessage(message)
            
            // Then
            assertEquals(ResponseStatus.ERROR, firstResponse.status)
            assertEquals(ResponseStatus.SUCCESS, secondResponse.status)
            assertEquals("Recovered response", secondResponse.content)
        }

        @Test
        @DisplayName("Should handle out of memory errors gracefully")
        fun shouldHandleOutOfMemoryErrorsGracefully() = runTest {
            // Given
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )
            
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any()))
                .thenThrow(OutOfMemoryError("Out of memory"))
            
            // When
            val response = auraAgent.processMessage(message)
            
            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SYSTEM_ERROR, response.status)
            assertTrue(response.content.contains("System error"))
        }

        @Test
        @DisplayName("Should maintain state consistency during concurrent failures")
        fun shouldMaintainStateConsistencyDuringConcurrentFailures() = runTest {
            // Given
            val messages = (1..5).map { index ->
                AgentMessage(
                    id = "msg-$index",
                    type = MessageType.TEXT,
                    content = "Message $index",
                    timestamp = System.currentTimeMillis()
                )
            }
            
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any()))
                .thenThrow(RuntimeException("Concurrent failure"))
            
            // When
            val responses = messages.map { message ->
                auraAgent.processMessage(message)
            }
            
            // Then
            assertEquals(5, responses.size)
            responses.forEach { response ->
                assertEquals(ResponseStatus.ERROR, response.status)
            }
            
            // Agent should still be in a consistent state
            assertTrue(auraAgent.isInitialized())
            assertEquals(0, auraAgent.getActiveTaskCount())
        }
    }

    @Nested
    @DisplayName("Performance and Resource Management Tests")
    inner class PerformanceAndResourceManagementTests {

        @Test
        @DisplayName("Should not exceed memory limits during message processing")
        fun shouldNotExceedMemoryLimitsDuringMessageProcessing() = runTest {
            // Given
            val initialMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
            val largeMessage = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "x".repeat(1000000), // 1MB message
                timestamp = System.currentTimeMillis()
            )
            
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = largeMessage.id,
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            )
            
            // When
            repeat(10) {
                auraAgent.processMessage(largeMessage)
            }
            
            // Then
            System.gc()
            val finalMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
            val memoryIncrease = finalMemory - initialMemory
            
            // Memory increase should be reasonable (less than 10MB)
            assertTrue(memoryIncrease < 10 * 1024 * 1024, 
                "Memory increase of ${memoryIncrease / 1024 / 1024}MB exceeded acceptable limit")
        }

        @Test
        @DisplayName("Should handle message processing timeout gracefully")
        fun shouldHandleMessageProcessingTimeoutGracefully() = runTest {
            // Given
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )
            
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                Thread.sleep(10000) // Simulate long processing time
                AgentResponse(
                    messageId = message.id,
                    content = "Delayed response",
                    status = ResponseStatus.SUCCESS
                )
            }
            
            // When
            val startTime = System.currentTimeMillis()
            val response = auraAgent.processMessage(message)
            val endTime = System.currentTimeMillis()
            
            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.TIMEOUT, response.status)
            assertTrue(endTime - startTime < 5000, "Processing should timeout within 5 seconds")
        }

        @Test
        @DisplayName("Should clean up resources after processing")
        fun shouldCleanUpResourcesAfterProcessing() = runTest {
            // Given
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )
            
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Response",
                    status = ResponseStatus.SUCCESS
                )
            )
            
            // When
            auraAgent.processMessage(message)
            
            // Then
            assertEquals(0, auraAgent.getActiveTaskCount())
            assertTrue(auraAgent.getResourceUsage().memoryUsage < 1024 * 1024) // Less than 1MB
        }
    }
}