package dev.aurakai.auraframefx.ai.agents

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.async
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestCoroutineScheduler
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.RepeatedTest
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.extension.ExtendWith
import org.junit.jupiter.api.Timeout
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.junit.jupiter.params.provider.EnumSource
import org.junit.jupiter.params.provider.NullSource
import org.junit.jupiter.params.provider.EmptySource
import org.junit.jupiter.params.provider.CsvSource
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.jupiter.MockitoExtension
import org.mockito.kotlin.any
import org.mockito.kotlin.argumentCaptor
import org.mockito.kotlin.eq
import org.mockito.kotlin.never
import org.mockito.kotlin.times
import org.mockito.kotlin.verify
import org.mockito.kotlin.whenever
import org.mockito.kotlin.clearInvocations
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.assertFailsWith
import kotlin.test.assertContains
import kotlin.test.assertNotEquals

/**
 * Comprehensive test suite for AuraAgent using JUnit 5 and Mockito.
 * This test suite covers all public interfaces, edge cases, error conditions,
 * performance scenarios, thread safety, and integration scenarios.
 */
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
        clearInvocations(mockAgentContext, mockMessageHandler, mockEventBus, mockConfigurationProvider)
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
                @Suppress("CAST_NEVER_SUCCEEDS")
                AuraAgent(null as AgentContext)
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

        @ParameterizedTest
        @ValueSource(strings = ["", "   ", "\t\n"])
        @DisplayName("Should handle invalid agent names gracefully")
        fun shouldHandleInvalidAgentNamesGracefully(invalidName: String) {
            // Given
            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(
                AgentConfiguration(
                    name = invalidName,
                    version = "1.0.0",
                    capabilities = listOf("CHAT"),
                    maxConcurrentTasks = 5
                )
            )

            // When
            val agent = AuraAgent(mockAgentContext)

            // Then
            assertNotNull(agent)
            assertFalse(agent.getName().isBlank())
        }

        @Test
        @DisplayName("Should initialize with dependency injection verification")
        fun shouldInitializeWithDependencyInjectionVerification() {
            // Given
            val customContext = mock<AgentContext>()
            val customHandler = mock<MessageHandler>()
            val customEventBus = mock<EventBus>()
            val customConfigProvider = mock<ConfigurationProvider>()

            whenever(customContext.getConfigurationProvider()).thenReturn(customConfigProvider)
            whenever(customContext.getMessageHandler()).thenReturn(customHandler)
            whenever(customContext.getEventBus()).thenReturn(customEventBus)
            whenever(customConfigProvider.getAgentConfiguration()).thenReturn(
                AgentConfiguration(
                    name = "CustomAgent",
                    version = "2.0.0",
                    capabilities = listOf("CUSTOM"),
                    maxConcurrentTasks = 10
                )
            )

            // When
            val agent = AuraAgent(customContext)

            // Then
            verify(customContext).getConfigurationProvider()
            verify(customContext).getMessageHandler()
            verify(customContext).getEventBus()
            assertEquals("CustomAgent", agent.getName())
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

        @RepeatedTest(10)
        @DisplayName("Should handle concurrent message processing consistently")
        fun shouldHandleConcurrentMessageProcessingConsistently() = runTest {
            // Given
            val messages = (1..5).map { index ->
                AgentMessage(
                    id = "concurrent-$index",
                    type = MessageType.TEXT,
                    content = "Concurrent message $index",
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
            assertEquals(5, responses.size)
            responses.forEach { response ->
                assertEquals(ResponseStatus.SUCCESS, response.status)
            }
        }

        @Test
        @Timeout(value = 10, unit = TimeUnit.SECONDS)
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
            val barrier = CyclicBarrier(3) // 2 tasks + main thread
            val taskCounter = AtomicInteger(0)
            val maxConcurrent = AtomicInteger(0)

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                val current = taskCounter.incrementAndGet()
                maxConcurrent.updateAndGet { prev -> maxOf(prev, current) }
                barrier.await(5, TimeUnit.SECONDS)
                taskCounter.decrementAndGet()
                AgentResponse(
                    messageId = "test",
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            }

            // When
            val messages = (1..5).map { index ->
                AgentMessage(
                    id = "limit-test-$index",
                    type = MessageType.TEXT,
                    content = "Message $index",
                    timestamp = System.currentTimeMillis()
                )
            }

            val responses = messages.map { message ->
                agent.processMessage(message)
            }

            // Then
            assertEquals(5, responses.size)
            assertTrue(maxConcurrent.get() <= 2, "Max concurrent tasks exceeded: ${maxConcurrent.get()}")
        }

        @ParameterizedTest
        @EnumSource(MessageType::class)
        @DisplayName("Should handle all message types")
        fun shouldHandleAllMessageTypes(messageType: MessageType) = runTest {
            // Given
            val message = AgentMessage(
                id = "type-test-${messageType.name}",
                type = messageType,
                content = "Test content for $messageType",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Processed $messageType",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            verify(mockMessageHandler).validateMessage(message)
            verify(mockMessageHandler).processMessage(message)
        }

        @Test
        @DisplayName("Should maintain message processing order for sequential calls")
        fun shouldMaintainMessageProcessingOrderForSequentialCalls() = runTest {
            // Given
            val messageIds = mutableListOf<String>()
            val processedIds = mutableListOf<String>()

            val messages = (1..10).map { index ->
                val id = "order-test-$index"
                messageIds.add(id)
                AgentMessage(
                    id = id,
                    type = MessageType.TEXT,
                    content = "Order test message $index",
                    timestamp = System.currentTimeMillis()
                )
            }

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer { invocation ->
                val msg = invocation.getArgument<AgentMessage>(0)
                processedIds.add(msg.id)
                AgentResponse(
                    messageId = msg.id,
                    content = "Processed ${msg.id}",
                    status = ResponseStatus.SUCCESS
                )
            }

            // When
            messages.forEach { message ->
                auraAgent.processMessage(message)
            }

            // Then
            assertEquals(messageIds.size, processedIds.size)
            assertEquals(messageIds, processedIds)
        }

        @Test
        @DisplayName("Should handle message with metadata and context")
        fun shouldHandleMessageWithMetadataAndContext() = runTest {
            // Given
            val metadata = mapOf(
                "userId" to "user123",
                "sessionId" to "session456",
                "priority" to "high"
            )
            val message = AgentMessage(
                id = "metadata-test",
                type = MessageType.TEXT,
                content = "Message with metadata",
                timestamp = System.currentTimeMillis(),
                metadata = metadata
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer { invocation ->
                val msg = invocation.getArgument<AgentMessage>(0)
                assertEquals(metadata, msg.metadata)
                AgentResponse(
                    messageId = msg.id,
                    content = "Processed with metadata",
                    status = ResponseStatus.SUCCESS,
                    metadata = mapOf("processedBy" to "AuraAgent")
                )
            }

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            assertNotNull(response.metadata)
            assertEquals("AuraAgent", response.metadata!!["processedBy"])
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
                id = "event-test-001",
                type = MessageType.TEXT,
                content = "Test message for events",
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
                id = "event-failure-test",
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

        @Test
        @DisplayName("Should publish events in correct order")
        fun shouldPublishEventsInCorrectOrder() = runTest {
            // Given
            val messages = (1..3).map { index ->
                AgentMessage(
                    id = "order-event-$index",
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
            messages.forEach { message ->
                auraAgent.processMessage(message)
            }

            // Then
            val eventCaptor = argumentCaptor<AgentEvent>()
            verify(mockEventBus, times(3)).publish(eventCaptor.capture())

            val events = eventCaptor.allValues
            assertEquals(3, events.size)
            events.forEachIndexed { index, event ->
                assertEquals(EventType.MESSAGE_PROCESSED, event.type)
                assertEquals("order-event-${index + 1}", event.data["messageId"])
            }
        }

        @Test
        @DisplayName("Should handle event listener registration failures")
        fun shouldHandleEventListenerRegistrationFailures() {
            // Given
            val context = mock<AgentContext>()
            whenever(context.getConfigurationProvider()).thenReturn(mockConfigurationProvider)
            whenever(context.getMessageHandler()).thenReturn(mockMessageHandler)
            whenever(context.getEventBus()).thenReturn(mockEventBus)

            doThrow(RuntimeException("Registration failed"))
                .whenever(mockEventBus).subscribe(any(), any())

            // When & Then
            assertDoesNotThrow {
                AuraAgent(context)
            }
        }

        @Test
        @DisplayName("Should unsubscribe from events during shutdown")
        fun shouldUnsubscribeFromEventsDuringShutdown() = runTest {
            // Given
            auraAgent.start()

            // When
            auraAgent.shutdown()

            // Then
            verify(mockEventBus).unsubscribe(eq(EventType.SYSTEM_SHUTDOWN), any())
            verify(mockEventBus).unsubscribe(eq(EventType.CONFIGURATION_CHANGED), any())
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

        @ParameterizedTest
        @CsvSource(
            "0, 1",
            "-1, 1",
            "1000000, 1000",
            "500, 500"
        )
        @DisplayName("Should normalize max concurrent tasks to valid range")
        fun shouldNormalizeMaxConcurrentTasksToValidRange(input: Int, expected: Int) = runTest {
            // Given
            val config = AgentConfiguration(
                name = "TestAgent",
                version = "1.0.0",
                capabilities = listOf("CHAT"),
                maxConcurrentTasks = input
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(config)

            // When
            val agent = AuraAgent(mockAgentContext)

            // Then
            assertEquals(expected, agent.getMaxConcurrentTasks())
        }

        @Test
        @DisplayName("Should emit configuration changed event after successful reload")
        fun shouldEmitConfigurationChangedEventAfterSuccessfulReload() = runTest {
            // Given
            val newConfig = AgentConfiguration(
                name = "NewAgent",
                version = "3.0.0",
                capabilities = listOf("CHAT"),
                maxConcurrentTasks = 15
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(newConfig)

            // When
            auraAgent.handleConfigurationChanged()

            // Then
            verify(mockEventBus).publish(argThat { event ->
                event.type == EventType.CONFIGURATION_UPDATED &&
                event.data["agentName"] == "NewAgent" &&
                event.data["agentVersion"] == "3.0.0"
            })
        }

        @Test
        @DisplayName("Should handle hot configuration reloading during message processing")
        fun shouldHandleHotConfigurationReloadingDuringMessageProcessing() = runTest {
            // Given
            val message = AgentMessage(
                id = "hot-reload-test",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )

            val newConfig = AgentConfiguration(
                name = "HotReloadAgent",
                version = "4.0.0",
                capabilities = listOf("CHAT", "ANALYSIS"),
                maxConcurrentTasks = 20
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                // Simulate configuration change during processing
                whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(newConfig)
                auraAgent.handleConfigurationChanged()

                AgentResponse(
                    messageId = message.id,
                    content = "Processed with hot reload",
                    status = ResponseStatus.SUCCESS
                )
            }

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            assertEquals("HotReloadAgent", auraAgent.getName())
            assertEquals("4.0.0", auraAgent.getVersion())
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

        @ParameterizedTest
        @NullSource
        @EmptySource
        @ValueSource(strings = ["   ", "\t\n"])
        @DisplayName("Should handle invalid capability queries gracefully")
        fun shouldHandleInvalidCapabilityQueriesGracefully(capability: String?) {
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

        @ParameterizedTest
        @ValueSource(strings = ["CHAT", "chat", "Chat", "cHaT"])
        @DisplayName("Should handle case-insensitive capability checks")
        fun shouldHandleCaseInsensitiveCapabilityChecks(capability: String) {
            // When
            val hasCapability = auraAgent.hasCapability(capability)

            // Then
            assertTrue(hasCapability)
        }

        @Test
        @DisplayName("Should handle capability check with whitespace trimming")
        fun shouldHandleCapabilityCheckWithWhitespaceTrimming() {
            // Given
            val capabilityWithSpaces = "  CHAT  "

            // When
            val hasCapability = auraAgent.hasCapability(capabilityWithSpaces)

            // Then
            assertTrue(hasCapability)
        }

        @Test
        @DisplayName("Should return immutable capabilities list")
        fun shouldReturnImmutableCapabilitiesList() {
            // When
            val capabilities = auraAgent.getCapabilities()

            // Then
            assertFailsWith<UnsupportedOperationException> {
                @Suppress("UNCHECKED_CAST")
                (capabilities as MutableList<String>).add("NEW_CAPABILITY")
            }
        }

        @Test
        @DisplayName("Should handle capabilities update via configuration reload")
        fun shouldHandleCapabilitiesUpdateViaConfigurationReload() = runTest {
            // Given
            val originalCapabilities = auraAgent.getCapabilities()
            val newConfig = AgentConfiguration(
                name = "TestAgent",
                version = "1.0.0",
                capabilities = listOf("CHAT", "ANALYSIS", "TRANSLATION", "GENERATION"),
                maxConcurrentTasks = 5
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(newConfig)

            // When
            auraAgent.handleConfigurationChanged()
            val updatedCapabilities = auraAgent.getCapabilities()

            // Then
            assertNotEquals(originalCapabilities.size, updatedCapabilities.size)
            assertEquals(4, updatedCapabilities.size)
            assertTrue(updatedCapabilities.contains("TRANSLATION"))
            assertTrue(updatedCapabilities.contains("GENERATION"))
        }

        @Test
        @DisplayName("Should deduplicate capabilities from configuration")
        fun shouldDeduplicateCapabilitiesFromConfiguration() = runTest {
            // Given
            val configWithDuplicates = AgentConfiguration(
                name = "DuplicateAgent",
                version = "1.0.0",
                capabilities = listOf("CHAT", "CHAT", "ANALYSIS", "CHAT", "ANALYSIS"),
                maxConcurrentTasks = 5
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(configWithDuplicates)

            // When
            val agent = AuraAgent(mockAgentContext)
            val capabilities = agent.getCapabilities()

            // Then
            assertEquals(2, capabilities.size)
            assertTrue(capabilities.contains("CHAT"))
            assertTrue(capabilities.contains("ANALYSIS"))
        }

        @Test
        @DisplayName("Should handle special characters in capability names")
        fun shouldHandleSpecialCharactersInCapabilityNames() {
            // Given
            val specialCapability = "CHAT@#$%"

            // When
            val hasCapability = auraAgent.hasCapability(specialCapability)

            // Then
            assertFalse(hasCapability)
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
                id = "post-shutdown-test",
                type = MessageType.TEXT,
                content = "Test message after shutdown",
                timestamp = System.currentTimeMillis()
            )

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.AGENT_SHUTDOWN, response.status)
            verify(mockMessageHandler, never()).processMessage(any())
        }

        @Test
        @DisplayName("Should support agent restart after shutdown")
        fun shouldSupportAgentRestartAfterShutdown() = runTest {
            // Given
            auraAgent.start()
            auraAgent.shutdown()
            assertTrue(auraAgent.isShutdown())

            // When
            auraAgent.restart()

            // Then
            assertFalse(auraAgent.isShutdown())
            assertTrue(auraAgent.isRunning())
            verify(mockEventBus).publish(argThat { event ->
                event.type == EventType.AGENT_RESTARTED
            })
        }

        @Test
        @DisplayName("Should handle lifecycle state transitions correctly")
        fun shouldHandleLifecycleStateTransitionsCorrectly() = runTest {
            // Initial state
            assertFalse(auraAgent.isRunning())
            assertFalse(auraAgent.isShutdown())
            assertTrue(auraAgent.isInitialized())

            // Start
            auraAgent.start()
            assertTrue(auraAgent.isRunning())
            assertFalse(auraAgent.isShutdown())

            // Stop
            auraAgent.stop()
            assertFalse(auraAgent.isRunning())
            assertFalse(auraAgent.isShutdown())

            // Shutdown
            auraAgent.shutdown()
            assertFalse(auraAgent.isRunning())
            assertTrue(auraAgent.isShutdown())
        }

        @Test
        @DisplayName("Should handle concurrent lifecycle operations")
        fun shouldHandleConcurrentLifecycleOperations() = runTest {
            // Given
            val executor = Executors.newFixedThreadPool(10)
            val latch = CountDownLatch(20)
            val exceptions = mutableListOf<Exception>()

            // When
            repeat(20) { index ->
                executor.submit {
                    try {
                        when (index % 4) {
                            0 -> auraAgent.start()
                            1 -> auraAgent.stop()
                            2 -> if (!auraAgent.isShutdown()) auraAgent.start()
                            3 -> if (!auraAgent.isShutdown()) auraAgent.stop()
                        }
                    } catch (e: Exception) {
                        exceptions.add(e)
                    } finally {
                        latch.countDown()
                    }
                }
            }

            latch.await(10, TimeUnit.SECONDS)
            executor.shutdown()

            // Then
            assertTrue(exceptions.isEmpty(), "No exceptions should occur during concurrent operations")
            assertTrue(auraAgent.isInitialized())
        }

        @Test
        @DisplayName("Should handle graceful shutdown with pending tasks")
        fun shouldHandleGracefulShutdownWithPendingTasks() = runTest {
            // Given
            auraAgent.start()
            val pendingTaskLatch = CountDownLatch(1)
            val shutdownLatch = CountDownLatch(1)

            val message = AgentMessage(
                id = "pending-task-test",
                type = MessageType.TEXT,
                content = "Long running task",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                pendingTaskLatch.countDown()
                shutdownLatch.await(5, TimeUnit.SECONDS)
                AgentResponse(
                    messageId = message.id,
                    content = "Completed during shutdown",
                    status = ResponseStatus.SUCCESS
                )
            }

            // When
            val responsePromise = async { auraAgent.processMessage(message) }
            pendingTaskLatch.await(1, TimeUnit.SECONDS)
            auraAgent.shutdown()
            shutdownLatch.countDown()
            val response = responsePromise.await()

            // Then
            assertNotNull(response)
            assertTrue(auraAgent.isShutdown())
            assertEquals(0, auraAgent.getActiveTaskCount())
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
                id = "recovery-test",
                type = MessageType.TEXT,
                content = "Recovery test message",
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
                id = "oom-test",
                type = MessageType.TEXT,
                content = "OOM test message",
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
            val messages = (1..10).map { index ->
                AgentMessage(
                    id = "concurrent-failure-$index",
                    type = MessageType.TEXT,
                    content = "Concurrent failure test $index",
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
            assertEquals(10, responses.size)
            responses.forEach { response ->
                assertEquals(ResponseStatus.ERROR, response.status)
            }

            // Agent should still be in a consistent state
            assertTrue(auraAgent.isInitialized())
            assertEquals(0, auraAgent.getActiveTaskCount())
        }

        @Test
        @DisplayName("Should handle cascading failures appropriately")
        fun shouldHandleCascadingFailuresAppropriately() = runTest {
            // Given
            val message = AgentMessage(
                id = "cascade-test",
                type = MessageType.TEXT,
                content = "Cascade test",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any()))
                .thenThrow(RuntimeException("First failure"))

            doThrow(RuntimeException("Event bus failure"))
                .whenever(mockEventBus).publish(any())

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.ERROR, response.status)
            // Agent should remain functional despite cascading failures
            assertTrue(auraAgent.isInitialized())
        }

        @ParameterizedTest
        @ValueSource(classes = [
            RuntimeException::class,
            IllegalArgumentException::class,
            IllegalStateException::class,
            NullPointerException::class,
            IndexOutOfBoundsException::class
        ])
        @DisplayName("Should handle various exception types gracefully")
        fun shouldHandleVariousExceptionTypesGracefully(exceptionClass: Class<out Exception>) = runTest {
            // Given
            val message = AgentMessage(
                id = "exception-test-${exceptionClass.simpleName}",
                type = MessageType.TEXT,
                content = "Exception test",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any()))
                .thenThrow(exceptionClass.getDeclaredConstructor(String::class.java)
                    .newInstance("Test exception"))

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.ERROR, response.status)
            assertContains(response.content, "Test exception")
        }

        @Test
        @DisplayName("Should implement circuit breaker pattern for repeated failures")
        fun shouldImplementCircuitBreakerPatternForRepeatedFailures() = runTest {
            // Given
            val messages = (1..10).map { index ->
                AgentMessage(
                    id = "circuit-breaker-$index",
                    type = MessageType.TEXT,
                    content = "Circuit breaker test $index",
                    timestamp = System.currentTimeMillis()
                )
            }

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any()))
                .thenThrow(RuntimeException("Repeated failure"))

            // When
            val responses = messages.map { message ->
                auraAgent.processMessage(message)
            }

            // Then
            assertTrue(responses.take(5).all { it.status == ResponseStatus.ERROR })
            // After circuit breaker kicks in, should return circuit breaker response
            assertTrue(responses.drop(5).any { it.status == ResponseStatus.CIRCUIT_BREAKER_OPEN })
        }

        @Test
        @DisplayName("Should log detailed error information for debugging")
        fun shouldLogDetailedErrorInformationForDebugging() = runTest {
            // Given
            val message = AgentMessage(
                id = "debug-test",
                type = MessageType.TEXT,
                content = "Debug test",
                timestamp = System.currentTimeMillis()
            )

            val complexException = RuntimeException("Root cause",
                IllegalStateException("Intermediate cause",
                    NullPointerException("Original cause")))

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenThrow(complexException)

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.ERROR, response.status)
            // Response should contain diagnostic information
            assertTrue(response.content.contains("Root cause") ||
                      response.metadata?.containsKey("errorDetails") == true)
        }
    }

    @Nested
    @DisplayName("Performance and Resource Management Tests")
    inner class PerformanceAndResourceManagementTests {

        @Test
        @DisplayName("Should not exceed memory limits during message processing")
        fun shouldNotExceedMemoryLimitsDuringMessageProcessing() = runTest {
            // Given
            val runtime = Runtime.getRuntime()
            val initialMemory = runtime.totalMemory() - runtime.freeMemory()

            val largeMessage = AgentMessage(
                id = "memory-test",
                type = MessageType.TEXT,
                content = "x".repeat(1_000_000), // 1MB message
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = largeMessage.id,
                    content = "Processed large message",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            repeat(10) {
                auraAgent.processMessage(largeMessage)
            }

            // Then
            System.gc()
            Thread.sleep(100) // Allow GC to complete
            val finalMemory = runtime.totalMemory() - runtime.freeMemory()
            val memoryIncrease = finalMemory - initialMemory

            // Memory increase should be reasonable (less than 10MB)
            assertTrue(memoryIncrease < 10 * 1024 * 1024,
                "Memory increase of ${memoryIncrease / 1024 / 1024}MB exceeded acceptable limit")
        }

        @Test
        @Timeout(value = 5, unit = TimeUnit.SECONDS)
        @DisplayName("Should handle message processing timeout gracefully")
        fun shouldHandleMessageProcessingTimeoutGracefully() = runTest {
            // Given
            val message = AgentMessage(
                id = "timeout-test",
                type = MessageType.TEXT,
                content = "Timeout test message",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                Thread.sleep(10_000) // Simulate long processing time
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
                id = "cleanup-test",
                type = MessageType.TEXT,
                content = "Resource cleanup test",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            auraAgent.processMessage(message)

            // Then
            assertEquals(0, auraAgent.getActiveTaskCount())
            assertTrue(auraAgent.getResourceUsage().memoryUsage < 1024 * 1024) // Less than 1MB
        }

        @Test
        @DisplayName("Should throttle message processing under high load")
        fun shouldThrottleMessageProcessingUnderHighLoad() = runTest {
            // Given
            val executor = Executors.newFixedThreadPool(50)
            val processedCount = AtomicInteger(0)
            val rejectedCount = AtomicInteger(0)

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                Thread.sleep(100) // Simulate processing time
                processedCount.incrementAndGet()
                AgentResponse(
                    messageId = "test",
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            }

            // When
            val latch = CountDownLatch(100)
            repeat(100) { index ->
                executor.submit {
                    try {
                        val message = AgentMessage(
                            id = "load-test-$index",
                            type = MessageType.TEXT,
                            content = "Load test $index",
                            timestamp = System.currentTimeMillis()
                        )
                        val response = auraAgent.processMessage(message)
                        if (response.status == ResponseStatus.THROTTLED) {
                            rejectedCount.incrementAndGet()
                        }
                    } finally {
                        latch.countDown()
                    }
                }
            }

            latch.await(30, TimeUnit.SECONDS)
            executor.shutdown()

            // Then
            assertTrue(processedCount.get() > 0, "Some messages should be processed")
            // Under high load, some messages might be throttled
            val totalHandled = processedCount.get() + rejectedCount.get()
            assertEquals(100, totalHandled)
        }

        @RepeatedTest(5)
        @DisplayName("Should maintain consistent performance under repeated load")
        fun shouldMaintainConsistentPerformanceUnderRepeatedLoad() = runTest {
            // Given
            val messages = (1..50).map { index ->
                AgentMessage(
                    id = "perf-test-$index",
                    type = MessageType.TEXT,
                    content = "Performance test $index",
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
            val startTime = System.currentTimeMillis()
            messages.forEach { message ->
                auraAgent.processMessage(message)
            }
            val endTime = System.currentTimeMillis()
            val processingTime = endTime - startTime

            // Then
            assertTrue(processingTime < 10_000, "Processing 50 messages should take less than 10 seconds")
            assertEquals(0, auraAgent.getActiveTaskCount())
        }

        @Test
        @DisplayName("Should handle resource exhaustion gracefully")
        fun shouldHandleResourceExhaustionGracefully() = runTest {
            // Given
            val resourceExhaustionMessage = AgentMessage(
                id = "resource-exhaustion-test",
                type = MessageType.TEXT,
                content = "Resource exhaustion test",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                // Simulate resource exhaustion
                throw OutOfMemoryError("Java heap space")
            }

            // When
            val response = auraAgent.processMessage(resourceExhaustionMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SYSTEM_ERROR, response.status)
            assertTrue(auraAgent.isInitialized()) // Agent should remain functional
        }
    }
}