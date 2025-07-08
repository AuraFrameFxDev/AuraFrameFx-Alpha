package dev.aurakai.auraframefx.ai.pipeline

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
import org.mockito.kotlin.times
import org.mockito.kotlin.never
import org.mockito.kotlin.argumentCaptor
import org.mockito.kotlin.mock
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.setMain
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.Dispatchers
import java.io.IOException
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeoutException

@ExtendWith(MockitoExtension::class)
@ExperimentalCoroutinesApi
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@DisplayName("AIPipelineProcessor Tests")
class AIPipelineProcessorTest {

    @Mock
    private lateinit var mockPipelineStage: PipelineStage

    @Mock
    private lateinit var mockInputProcessor: InputProcessor

    @Mock
    private lateinit var mockOutputProcessor: OutputProcessor

    @Mock
    private lateinit var mockErrorHandler: ErrorHandler

    @Mock
    private lateinit var mockMetricsCollector: MetricsCollector

    private lateinit var processor: AIPipelineProcessor
    private val testDispatcher = StandardTestDispatcher()

    @BeforeEach
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
        processor = AIPipelineProcessor(
            inputProcessor = mockInputProcessor,
            outputProcessor = mockOutputProcessor,
            errorHandler = mockErrorHandler,
            metricsCollector = mockMetricsCollector
        )
    }

    @AfterEach
    fun tearDown() {
        Dispatchers.resetMain()
        reset(mockPipelineStage, mockInputProcessor, mockOutputProcessor, mockErrorHandler, mockMetricsCollector)
    }

    @Nested
    @DisplayName("Pipeline Initialization Tests")
    inner class PipelineInitializationTests {

        @Test
        @DisplayName("should initialize pipeline with valid configuration")
        fun shouldInitializePipelineWithValidConfiguration() {
            // Given
            val config = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = true,
                timeoutMs = 5000L
            )
            
            // When
            val result = processor.initialize(config)
            
            // Then
            assertTrue(result)
            assertTrue(processor.isInitialized())
        }
        
        @Test
        @DisplayName("should fail initialization with null configuration")
        fun shouldFailInitializationWithNullConfiguration() {
            // When & Then
            assertThrows<IllegalArgumentException> {
                processor.initialize(null)
            }
            assertFalse(processor.isInitialized())
        }
        
        @Test
        @DisplayName("should fail initialization with empty stages")
        fun shouldFailInitializationWithEmptyStages() {
            // Given
            val config = PipelineConfiguration(
                stages = emptyList(),
                parallelExecution = false,
                timeoutMs = 1000L
            )
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                processor.initialize(config)
            }
            assertFalse(processor.isInitialized())
        }
        
        @Test
        @DisplayName("should fail initialization with negative timeout")
        fun shouldFailInitializationWithNegativeTimeout() {
            // Given
            val config = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = false,
                timeoutMs = -1L
            )
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                processor.initialize(config)
            }
        }
    }

    @Nested
    @DisplayName("Pipeline Processing Tests")
    inner class PipelineProcessingTests {

        @BeforeEach
        fun setUpProcessor() {
            val config = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = false,
                timeoutMs = 5000L
            )
            processor.initialize(config)
        }

        @Test
        @DisplayName("should process valid input successfully")
        fun shouldProcessValidInputSuccessfully() = runTest {
            // Given
            val input = "test input"
            val expectedOutput = "processed output"
            whenever(mockInputProcessor.process(input)).thenReturn(input)
            whenever(mockPipelineStage.execute(any())).thenReturn(expectedOutput)
            whenever(mockOutputProcessor.process(expectedOutput)).thenReturn(expectedOutput)
            
            // When
            val result = processor.process(input)
            
            // Then
            assertEquals(expectedOutput, result)
            verify(mockInputProcessor).process(input)
            verify(mockPipelineStage).execute(any())
            verify(mockOutputProcessor).process(expectedOutput)
            verify(mockMetricsCollector).recordProcessingTime(any())
        }

        @Test
        @DisplayName("should handle null input gracefully")
        fun shouldHandleNullInputGracefully() = runTest {
            // When & Then
            assertThrows<IllegalArgumentException> {
                processor.process(null)
            }
            verify(mockInputProcessor, never()).process(any())
        }

        @Test
        @DisplayName("should handle empty input")
        fun shouldHandleEmptyInput() = runTest {
            // Given
            val input = ""
            whenever(mockInputProcessor.process(input)).thenReturn(input)
            whenever(mockPipelineStage.execute(any())).thenReturn("")
            whenever(mockOutputProcessor.process("")).thenReturn("")
            
            // When
            val result = processor.process(input)
            
            // Then
            assertEquals("", result)
            verify(mockInputProcessor).process(input)
        }

        @Test
        @DisplayName("should process without initialization should throw exception")
        fun shouldProcessWithoutInitializationShouldThrowException() = runTest {
            // Given
            val uninitializedProcessor = AIPipelineProcessor(
                mockInputProcessor, mockOutputProcessor, mockErrorHandler, mockMetricsCollector
            )
            
            // When & Then
            assertThrows<IllegalStateException> {
                uninitializedProcessor.process("test")
            }
        }

        @Test
        @DisplayName("should handle processing timeout")
        fun shouldHandleProcessingTimeout() = runTest {
            // Given
            val input = "test input"
            val config = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = false,
                timeoutMs = 100L
            )
            processor.initialize(config)
            
            whenever(mockInputProcessor.process(input)).thenReturn(input)
            whenever(mockPipelineStage.execute(any())).thenAnswer { 
                Thread.sleep(200) // Simulate slow processing
                "result"
            }
            
            // When & Then
            assertThrows<TimeoutException> {
                processor.process(input)
            }
            verify(mockErrorHandler).handleTimeout(any())
        }
    }

    @Nested
    @DisplayName("Error Handling Tests")
    inner class ErrorHandlingTests {

        @BeforeEach
        fun setUpProcessor() {
            val config = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = false,
                timeoutMs = 5000L
            )
            processor.initialize(config)
        }

        @Test
        @DisplayName("should handle input processing exception")
        fun shouldHandleInputProcessingException() = runTest {
            // Given
            val input = "test input"
            val exception = RuntimeException("Input processing failed")
            whenever(mockInputProcessor.process(input)).thenThrow(exception)
            
            // When & Then
            assertThrows<RuntimeException> {
                processor.process(input)
            }
            verify(mockErrorHandler).handleProcessingError(exception)
            verify(mockPipelineStage, never()).execute(any())
        }

        @Test
        @DisplayName("should handle pipeline stage exception")
        fun shouldHandlePipelineStageException() = runTest {
            // Given
            val input = "test input"
            val exception = IOException("Pipeline stage failed")
            whenever(mockInputProcessor.process(input)).thenReturn(input)
            whenever(mockPipelineStage.execute(any())).thenThrow(exception)
            
            // When & Then
            assertThrows<IOException> {
                processor.process(input)
            }
            verify(mockErrorHandler).handleProcessingError(exception)
            verify(mockOutputProcessor, never()).process(any())
        }

        @Test
        @DisplayName("should handle output processing exception")
        fun shouldHandleOutputProcessingException() = runTest {
            // Given
            val input = "test input"
            val stageOutput = "stage output"
            val exception = IllegalStateException("Output processing failed")
            whenever(mockInputProcessor.process(input)).thenReturn(input)
            whenever(mockPipelineStage.execute(any())).thenReturn(stageOutput)
            whenever(mockOutputProcessor.process(stageOutput)).thenThrow(exception)
            
            // When & Then
            assertThrows<IllegalStateException> {
                processor.process(input)
            }
            verify(mockErrorHandler).handleProcessingError(exception)
        }

        @Test
        @DisplayName("should retry on transient failures")
        fun shouldRetryOnTransientFailures() = runTest {
            // Given
            val input = "test input"
            val transientException = IOException("Temporary failure")
            whenever(mockInputProcessor.process(input)).thenReturn(input)
            whenever(mockPipelineStage.execute(any()))
                .thenThrow(transientException)
                .thenReturn("success")
            whenever(mockOutputProcessor.process("success")).thenReturn("success")
            
            processor.setRetryPolicy(RetryPolicy(maxRetries = 1, retryableExceptions = setOf(IOException::class.java)))
            
            // When
            val result = processor.process(input)
            
            // Then
            assertEquals("success", result)
            verify(mockPipelineStage, times(2)).execute(any())
            verify(mockErrorHandler).handleRetryableError(transientException)
        }
    }

    @Nested
    @DisplayName("Parallel Processing Tests")
    inner class ParallelProcessingTests {

        @Test
        @DisplayName("should process multiple stages in parallel")
        fun shouldProcessMultipleStagesInParallel() = runTest {
            // Given
            val stage1 = mock<PipelineStage>()
            val stage2 = mock<PipelineStage>()
            val config = PipelineConfiguration(
                stages = listOf(stage1, stage2),
                parallelExecution = true,
                timeoutMs = 5000L
            )
            processor.initialize(config)
            
            val input = "test input"
            whenever(mockInputProcessor.process(input)).thenReturn(input)
            whenever(stage1.execute(any())).thenReturn("output1")
            whenever(stage2.execute(any())).thenReturn("output2")
            whenever(mockOutputProcessor.process(any())).thenReturn("final output")
            
            // When
            val result = processor.process(input)
            
            // Then
            assertEquals("final output", result)
            verify(stage1).execute(any())
            verify(stage2).execute(any())
            verify(mockMetricsCollector).recordParallelExecution(2)
        }

        @Test
        @DisplayName("should handle partial parallel stage failures")
        fun shouldHandlePartialParallelStageFailures() = runTest {
            // Given
            val stage1 = mock<PipelineStage>()
            val stage2 = mock<PipelineStage>()
            val config = PipelineConfiguration(
                stages = listOf(stage1, stage2),
                parallelExecution = true,
                timeoutMs = 5000L,
                failFast = false
            )
            processor.initialize(config)
            
            val input = "test input"
            whenever(mockInputProcessor.process(input)).thenReturn(input)
            whenever(stage1.execute(any())).thenReturn("output1")
            whenever(stage2.execute(any())).thenThrow(RuntimeException("Stage 2 failed"))
            
            // When & Then
            assertThrows<RuntimeException> {
                processor.process(input)
            }
            verify(mockErrorHandler).handlePartialFailure(any(), any())
        }
    }

    @Nested
    @DisplayName("Metrics and Monitoring Tests")
    inner class MetricsAndMonitoringTests {

        @BeforeEach
        fun setUpProcessor() {
            val config = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = false,
                timeoutMs = 5000L
            )
            processor.initialize(config)
        }

        @Test
        @DisplayName("should collect processing metrics")
        fun shouldCollectProcessingMetrics() = runTest {
            // Given
            val input = "test input"
            whenever(mockInputProcessor.process(input)).thenReturn(input)
            whenever(mockPipelineStage.execute(any())).thenReturn("output")
            whenever(mockOutputProcessor.process("output")).thenReturn("output")
            
            // When
            processor.process(input)
            
            // Then
            verify(mockMetricsCollector).recordProcessingTime(any())
            verify(mockMetricsCollector).recordThroughput(1)
            verify(mockMetricsCollector).recordStageExecution(any())
        }

        @Test
        @DisplayName("should collect error metrics")
        fun shouldCollectErrorMetrics() = runTest {
            // Given
            val input = "test input"
            val exception = RuntimeException("Processing failed")
            whenever(mockInputProcessor.process(input)).thenThrow(exception)
            
            // When & Then
            assertThrows<RuntimeException> {
                processor.process(input)
            }
            verify(mockMetricsCollector).recordError(exception::class.java.simpleName)
            verify(mockMetricsCollector).recordFailureRate(any())
        }

        @Test
        @DisplayName("should provide processing statistics")
        fun shouldProvideProcessingStatistics() {
            // When
            val stats = processor.getProcessingStatistics()
            
            // Then
            assertNotNull(stats)
            verify(mockMetricsCollector).getStatistics()
        }
    }

    @Nested
    @DisplayName("Lifecycle Management Tests")
    inner class LifecycleManagementTests {

        @Test
        @DisplayName("should shutdown gracefully")
        fun shouldShutdownGracefully() = runTest {
            // Given
            val config = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = false,
                timeoutMs = 5000L
            )
            processor.initialize(config)
            
            // When
            processor.shutdown()
            
            // Then
            assertFalse(processor.isInitialized())
            verify(mockPipelineStage).cleanup()
            verify(mockMetricsCollector).shutdown()
        }

        @Test
        @DisplayName("should handle shutdown timeout")
        fun shouldHandleShutdownTimeout() = runTest {
            // Given
            val config = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = false,
                timeoutMs = 5000L
            )
            processor.initialize(config)
            
            whenever(mockPipelineStage.cleanup()).thenAnswer {
                Thread.sleep(6000) // Simulate slow cleanup
            }
            
            // When
            val shutdownResult = processor.shutdown(timeoutMs = 1000L)
            
            // Then
            assertFalse(shutdownResult)
            verify(mockErrorHandler).handleShutdownTimeout()
        }

        @Test
        @DisplayName("should restart processor after shutdown")
        fun shouldRestartProcessorAfterShutdown() {
            // Given
            val config = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = false,
                timeoutMs = 5000L
            )
            processor.initialize(config)
            processor.shutdown()
            
            // When
            val restartResult = processor.initialize(config)
            
            // Then
            assertTrue(restartResult)
            assertTrue(processor.isInitialized())
        }
    }

    @Nested
    @DisplayName("Configuration Validation Tests")
    inner class ConfigurationValidationTests {

        @Test
        @DisplayName("should validate stage dependencies")
        fun shouldValidateStageDependencies() {
            // Given
            val stage1 = mock<PipelineStage>()
            val stage2 = mock<PipelineStage>()
            whenever(stage2.getDependencies()).thenReturn(setOf("stage1"))
            whenever(stage1.getName()).thenReturn("stage1")
            whenever(stage2.getName()).thenReturn("stage2")
            
            val config = PipelineConfiguration(
                stages = listOf(stage1, stage2),
                parallelExecution = false,
                timeoutMs = 5000L
            )
            
            // When
            val result = processor.initialize(config)
            
            // Then
            assertTrue(result)
        }

        @Test
        @DisplayName("should fail on circular dependencies")
        fun shouldFailOnCircularDependencies() {
            // Given
            val stage1 = mock<PipelineStage>()
            val stage2 = mock<PipelineStage>()
            whenever(stage1.getDependencies()).thenReturn(setOf("stage2"))
            whenever(stage2.getDependencies()).thenReturn(setOf("stage1"))
            whenever(stage1.getName()).thenReturn("stage1")
            whenever(stage2.getName()).thenReturn("stage2")
            
            val config = PipelineConfiguration(
                stages = listOf(stage1, stage2),
                parallelExecution = false,
                timeoutMs = 5000L
            )
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                processor.initialize(config)
            }
        }

        @Test
        @DisplayName("should validate stage configuration")
        fun shouldValidateStageConfiguration() {
            // Given
            whenever(mockPipelineStage.isConfigurationValid()).thenReturn(false)
            
            val config = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = false,
                timeoutMs = 5000L
            )
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                processor.initialize(config)
            }
        }
    }

    @Nested
    @DisplayName("Resource Management Tests")
    inner class ResourceManagementTests {

        @Test
        @DisplayName("should manage memory usage effectively")
        fun shouldManageMemoryUsageEffectively() = runTest {
            // Given
            val config = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = false,
                timeoutMs = 5000L,
                maxMemoryUsageMB = 100
            )
            processor.initialize(config)
            
            val largeInput = "x".repeat(1000000) // 1MB string
            whenever(mockInputProcessor.process(largeInput)).thenReturn(largeInput)
            whenever(mockPipelineStage.execute(any())).thenReturn("processed")
            whenever(mockOutputProcessor.process("processed")).thenReturn("processed")
            
            // When
            val result = processor.process(largeInput)
            
            // Then
            assertEquals("processed", result)
            verify(mockMetricsCollector).recordMemoryUsage(any())
        }

        @Test
        @DisplayName("should handle memory pressure")
        fun shouldHandleMemoryPressure() = runTest {
            // Given
            val config = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = false,
                timeoutMs = 5000L,
                maxMemoryUsageMB = 1 // Very low limit
            )
            processor.initialize(config)
            
            val largeInput = "x".repeat(1000000) // 1MB string
            
            // When & Then
            assertThrows<OutOfMemoryError> {
                processor.process(largeInput)
            }
            verify(mockErrorHandler).handleMemoryPressure()
        }
    }

    @Nested
    @DisplayName("Concurrent Processing Tests")
    inner class ConcurrentProcessingTests {

        @Test
        @DisplayName("should handle concurrent processing requests")
        fun shouldHandleConcurrentProcessingRequests() = runTest {
            // Given
            val config = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = false,
                timeoutMs = 5000L,
                maxConcurrentRequests = 3
            )
            processor.initialize(config)
            
            whenever(mockInputProcessor.process(any())).thenReturn("processed input")
            whenever(mockPipelineStage.execute(any())).thenReturn("stage output")
            whenever(mockOutputProcessor.process(any())).thenReturn("final output")
            
            // When
            val futures = (1..3).map { i ->
                CompletableFuture.supplyAsync {
                    runTest { processor.process("input $i") }
                }
            }
            
            val results = futures.map { it.get() }
            
            // Then
            assertEquals(3, results.size)
            results.forEach { assertEquals("final output", it) }
            verify(mockMetricsCollector, times(3)).recordConcurrentRequest()
        }

        @Test
        @DisplayName("should reject requests when at capacity")
        fun shouldRejectRequestsWhenAtCapacity() = runTest {
            // Given
            val config = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = false,
                timeoutMs = 5000L,
                maxConcurrentRequests = 1
            )
            processor.initialize(config)
            
            whenever(mockInputProcessor.process(any())).thenAnswer {
                Thread.sleep(1000) // Simulate slow processing
                "processed"
            }
            
            // When
            val future1 = CompletableFuture.supplyAsync {
                runTest { processor.process("input1") }
            }
            
            Thread.sleep(100) // Ensure first request starts
            
            // Then
            assertThrows<IllegalStateException> {
                runTest { processor.process("input2") }
            }
            
            future1.get() // Clean up
        }
    }

    @Nested
    @DisplayName("Advanced Initialization Edge Cases")
    inner class AdvancedInitializationEdgeCases {

        @Test
        @DisplayName("should handle zero timeout configuration")
        fun shouldHandleZeroTimeoutConfiguration() {
            // Given
            val config = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = false,
                timeoutMs = 0L
            )
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                processor.initialize(config)
            }
        }

        @Test
        @DisplayName("should handle extremely large timeout values")
        fun shouldHandleExtremelyLargeTimeoutValues() {
            // Given
            val config = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = false,
                timeoutMs = Long.MAX_VALUE
            )
            
            // When
            val result = processor.initialize(config)
            
            // Then
            assertTrue(result)
            assertTrue(processor.isInitialized())
        }

        @Test
        @DisplayName("should detect duplicate stages by name")
        fun shouldDetectDuplicateStagesByName() {
            // Given
            val stage1 = mock<PipelineStage>()
            val stage2 = mock<PipelineStage>()
            whenever(stage1.getName()).thenReturn("duplicateStage")
            whenever(stage2.getName()).thenReturn("duplicateStage")
            
            val config = PipelineConfiguration(
                stages = listOf(stage1, stage2),
                parallelExecution = false,
                timeoutMs = 5000L
            )
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                processor.initialize(config)
            }
        }

        @Test
        @DisplayName("should validate stage ordering for dependencies")
        fun shouldValidateStageOrderingForDependencies() {
            // Given
            val stage1 = mock<PipelineStage>()
            val stage2 = mock<PipelineStage>()
            whenever(stage1.getName()).thenReturn("stage1")
            whenever(stage2.getName()).thenReturn("stage2")
            whenever(stage1.getDependencies()).thenReturn(setOf("stage2"))
            whenever(stage2.getDependencies()).thenReturn(emptySet())
            
            val config = PipelineConfiguration(
                stages = listOf(stage1, stage2), // Wrong order
                parallelExecution = false,
                timeoutMs = 5000L
            )
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                processor.initialize(config)
            }
        }

        @Test
        @DisplayName("should re-initialize with different configuration")
        fun shouldReinitializeWithDifferentConfiguration() {
            // Given
            val config1 = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = false,
                timeoutMs = 1000L
            )
            val config2 = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = true,
                timeoutMs = 2000L
            )
            
            processor.initialize(config1)
            
            // When
            val result = processor.initialize(config2)
            
            // Then
            assertTrue(result)
            assertTrue(processor.isInitialized())
            verify(mockPipelineStage, times(2)).initialize()
        }
    }

    @Nested
    @DisplayName("Advanced Input Processing Edge Cases")
    inner class AdvancedInputProcessingEdgeCases {

        @BeforeEach
        fun setUpProcessor() {
            val config = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = false,
                timeoutMs = 5000L
            )
            processor.initialize(config)
        }

        @Test
        @DisplayName("should handle unicode and special characters")
        fun shouldHandleUnicodeAndSpecialCharacters() = runTest {
            // Given
            val unicodeInput = "🚀 Test with émojis and spéciál chãracters 中文 العربية"
            whenever(mockInputProcessor.process(unicodeInput)).thenReturn(unicodeInput)
            whenever(mockPipelineStage.execute(any())).thenReturn(unicodeInput)
            whenever(mockOutputProcessor.process(unicodeInput)).thenReturn(unicodeInput)
            
            // When
            val result = processor.process(unicodeInput)
            
            // Then
            assertEquals(unicodeInput, result)
            verify(mockInputProcessor).process(unicodeInput)
        }

        @Test
        @DisplayName("should handle extremely large input strings")
        fun shouldHandleExtremelyLargeInputStrings() = runTest {
            // Given
            val largeInput = "A".repeat(10_000_000) // 10MB string
            whenever(mockInputProcessor.process(largeInput)).thenReturn("processed")
            whenever(mockPipelineStage.execute(any())).thenReturn("output")
            whenever(mockOutputProcessor.process("output")).thenReturn("final")
            
            // When
            val result = processor.process(largeInput)
            
            // Then
            assertEquals("final", result)
            verify(mockMetricsCollector).recordLargeInputProcessing(largeInput.length)
        }

        @Test
        @DisplayName("should handle input with only whitespace")
        fun shouldHandleInputWithOnlyWhitespace() = runTest {
            // Given
            val whitespaceInput = "   \t\n\r   "
            whenever(mockInputProcessor.process(whitespaceInput)).thenReturn(whitespaceInput.trim())
            whenever(mockPipelineStage.execute(any())).thenReturn("")
            whenever(mockOutputProcessor.process("")).thenReturn("")
            
            // When
            val result = processor.process(whitespaceInput)
            
            // Then
            assertEquals("", result)
            verify(mockInputProcessor).process(whitespaceInput)
        }

        @Test
        @DisplayName("should handle input with control characters")
        fun shouldHandleInputWithControlCharacters() = runTest {
            // Given
            val controlCharsInput = "Test\u0000\u0001\u0002\u0003String"
            whenever(mockInputProcessor.process(controlCharsInput)).thenReturn("sanitized")
            whenever(mockPipelineStage.execute(any())).thenReturn("processed")
            whenever(mockOutputProcessor.process("processed")).thenReturn("final")
            
            // When
            val result = processor.process(controlCharsInput)
            
            // Then
            assertEquals("final", result)
            verify(mockInputProcessor).process(controlCharsInput)
        }

        @Test
        @DisplayName("should handle malformed input gracefully")
        fun shouldHandleMalformedInputGracefully() = runTest {
            // Given
            val malformedInput = "\uFFFE\uFFFF" // Invalid Unicode characters
            whenever(mockInputProcessor.process(malformedInput)).thenThrow(IllegalArgumentException("Malformed input"))
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                processor.process(malformedInput)
            }
            verify(mockErrorHandler).handleMalformedInput(any())
        }
    }

    @Nested
    @DisplayName("Advanced Error Recovery and Resilience Tests")
    inner class AdvancedErrorRecoveryTests {

        @BeforeEach
        fun setUpProcessor() {
            val config = PipelineConfiguration(
                stages = listOf(mockPipelineStage),
                parallelExecution = false,
                timeoutMs = 5000L
            )
            processor.initialize(config)
        }

        @Test
        @DisplayName("should handle cascading failures across multiple stages")
        fun shouldHandleCascadingFailuresAcrossMultipleStages() = runTest {
            // Given
            val stage1 = mock<PipelineStage>()
            val stage2 = mock<PipelineStage>()
            val stage3 = mock<PipelineStage>()
            
            val config = PipelineConfiguration(
                stages = listOf(stage1, stage2, stage3),
                parallelExecution = false,
                timeoutMs = 5000L
            )
            processor.initialize(config)
            
            val input = "test"
            whenever(mockInputProcessor.process(input)).thenReturn(input)
            whenever(stage1.execute(any())).thenThrow(RuntimeException("Stage 1 failed"))
            whenever(stage2.execute(any())).thenThrow(RuntimeException("Stage 2 failed"))
            whenever(stage3.execute(any())).thenThrow(RuntimeException("Stage 3 failed"))
            
            // When & Then
            assertThrows<RuntimeException> {
                processor.process(input)
            }
            verify(mockErrorHandler, times(3)).handleProcessingError(any())
        }

        @Test
        @DisplayName("should recover from intermittent network failures")
        fun shouldRecoverFromIntermittentNetworkFailures() = runTest {
            // Given
            val input = "network test"
            val networkException = java.net.ConnectException("Network temporarily unavailable")
            whenever(mockInputProcessor.process(input)).thenReturn(input)
            whenever(mockPipelineStage.execute(any()))
                .thenThrow(networkException)
                .thenThrow(networkException)
                .thenReturn("success after retries")
            whenever(mockOutputProcessor.process("success after retries")).thenReturn("final success")
            
            processor.setRetryPolicy(RetryPolicy(
                maxRetries = 2,
                retryableExceptions = setOf(java.net.ConnectException::class.java),
                backoffStrategy = ExponentialBackoff(100, 2.0)
            ))
            
            // When
            val result = processor.process(input)
            
            // Then
            assertEquals("final success", result)
            verify(mockPipelineStage, times(3)).execute(any())
            verify(mockErrorHandler, times(2)).handleRetryableError(networkException)
        }

        @Test
        @DisplayName("should implement circuit breaker pattern for repeated failures")
        fun shouldImplementCircuitBreakerPatternForRepeatedFailures() = runTest {
            // Given
            val input = "circuit breaker test"
            val persistentException = RuntimeException("Persistent failure")
            whenever(mockInputProcessor.process(input)).thenReturn(input)
            whenever(mockPipelineStage.execute(any())).thenThrow(persistentException)
            
            processor.setCircuitBreakerPolicy(CircuitBreakerPolicy(
                failureThreshold = 3,
                recoveryTimeoutMs = 10000L
            ))
            
            // When & Then
            repeat(4) {
                assertThrows<RuntimeException> {
                    processor.process(input)
                }
            }
            
            verify(mockErrorHandler).handleCircuitBreakerOpen()
        }
    }
}