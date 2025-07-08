package dev.aurakai.auraframefx.ai.pipeline

import io.mockk.*
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.junit.jupiter.params.provider.MethodSource
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutionException
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

class AIPipelineProcessorTest {

    private lateinit var processor: AIPipelineProcessor
    private val mockPipelineStage1 = mockk<PipelineStage>()
    private val mockPipelineStage2 = mockk<PipelineStage>()
    private val mockPipelineStage3 = mockk<PipelineStage>()
    private val mockErrorHandler = mockk<ErrorHandler>()

    @BeforeEach
    fun setUp() {
        clearAllMocks()
        processor = AIPipelineProcessor()
    }

    @AfterEach
    fun tearDown() {
        clearAllMocks()
    }

    @Nested
    @DisplayName("Pipeline Configuration Tests")
    inner class PipelineConfigurationTests {

        @Test
        @DisplayName("Should create pipeline with single stage")
        fun `should create pipeline with single stage`() {
            // Given
            val stage = mockPipelineStage1
            
            // When
            processor.addStage(stage)
            
            // Then
            assertEquals(1, processor.getStageCount())
            assertTrue(processor.hasStage(stage))
        }

        @Test
        @DisplayName("Should create pipeline with multiple stages")
        fun `should create pipeline with multiple stages`() {
            // Given
            val stages = listOf(mockPipelineStage1, mockPipelineStage2, mockPipelineStage3)
            
            // When
            stages.forEach { processor.addStage(it) }
            
            // Then
            assertEquals(3, processor.getStageCount())
            stages.forEach { assertTrue(processor.hasStage(it)) }
        }

        @Test
        @DisplayName("Should maintain stage order")
        fun `should maintain stage order`() {
            // Given
            val stages = listOf(mockPipelineStage1, mockPipelineStage2, mockPipelineStage3)
            
            // When
            stages.forEach { processor.addStage(it) }
            
            // Then
            val retrievedStages = processor.getStages()
            assertEquals(stages, retrievedStages)
        }

        @Test
        @DisplayName("Should remove stage from pipeline")
        fun `should remove stage from pipeline`() {
            // Given
            processor.addStage(mockPipelineStage1)
            processor.addStage(mockPipelineStage2)
            
            // When
            processor.removeStage(mockPipelineStage1)
            
            // Then
            assertEquals(1, processor.getStageCount())
            assertFalse(processor.hasStage(mockPipelineStage1))
            assertTrue(processor.hasStage(mockPipelineStage2))
        }

        @Test
        @DisplayName("Should clear all stages")
        fun `should clear all stages`() {
            // Given
            processor.addStage(mockPipelineStage1)
            processor.addStage(mockPipelineStage2)
            
            // When
            processor.clearStages()
            
            // Then
            assertEquals(0, processor.getStageCount())
            assertFalse(processor.hasStage(mockPipelineStage1))
            assertFalse(processor.hasStage(mockPipelineStage2))
        }

        @Test
        @DisplayName("Should handle duplicate stage addition")
        fun `should handle duplicate stage addition`() {
            // Given
            processor.addStage(mockPipelineStage1)
            
            // When
            processor.addStage(mockPipelineStage1)
            
            // Then
            assertEquals(1, processor.getStageCount())
        }

        @Test
        @DisplayName("Should handle removal of non-existent stage")
        fun `should handle removal of non-existent stage`() {
            // Given
            processor.addStage(mockPipelineStage1)
            
            // When & Then
            assertDoesNotThrow { processor.removeStage(mockPipelineStage2) }
            assertEquals(1, processor.getStageCount())
        }
    }

    @Nested
    @DisplayName("Pipeline Execution Tests")
    inner class PipelineExecutionTests {

        @Test
        @DisplayName("Should execute single stage successfully")
        fun `should execute single stage successfully`() {
            // Given
            val input = "test input"
            val expectedOutput = "processed output"
            every { mockPipelineStage1.process(input) } returns expectedOutput
            processor.addStage(mockPipelineStage1)
            
            // When
            val result = processor.execute(input)
            
            // Then
            assertEquals(expectedOutput, result)
            verify { mockPipelineStage1.process(input) }
        }

        @Test
        @DisplayName("Should execute multiple stages in sequence")
        fun `should execute multiple stages in sequence`() {
            // Given
            val input = "initial input"
            val stage1Output = "stage1 output"
            val stage2Output = "stage2 output"
            val finalOutput = "final output"
            
            every { mockPipelineStage1.process(input) } returns stage1Output
            every { mockPipelineStage2.process(stage1Output) } returns stage2Output
            every { mockPipelineStage3.process(stage2Output) } returns finalOutput
            
            processor.addStage(mockPipelineStage1)
            processor.addStage(mockPipelineStage2)
            processor.addStage(mockPipelineStage3)
            
            // When
            val result = processor.execute(input)
            
            // Then
            assertEquals(finalOutput, result)
            verifyOrder {
                mockPipelineStage1.process(input)
                mockPipelineStage2.process(stage1Output)
                mockPipelineStage3.process(stage2Output)
            }
        }

        @Test
        @DisplayName("Should handle empty pipeline")
        fun `should handle empty pipeline`() {
            // Given
            val input = "test input"
            
            // When
            val result = processor.execute(input)
            
            // Then
            assertEquals(input, result)
        }

        @Test
        @DisplayName("Should execute asynchronously")
        fun `should execute asynchronously`() {
            // Given
            val input = "async input"
            val expectedOutput = "async output"
            every { mockPipelineStage1.process(input) } returns expectedOutput
            processor.addStage(mockPipelineStage1)
            
            // When
            val future = processor.executeAsync(input)
            val result = future.get(5, TimeUnit.SECONDS)
            
            // Then
            assertEquals(expectedOutput, result)
            verify { mockPipelineStage1.process(input) }
        }

        @Test
        @DisplayName("Should handle timeout in async execution")
        fun `should handle timeout in async execution`() {
            // Given
            val input = "slow input"
            every { mockPipelineStage1.process(input) } answers {
                Thread.sleep(10000) // Simulate slow processing
                "slow output"
            }
            processor.addStage(mockPipelineStage1)
            
            // When & Then
            val future = processor.executeAsync(input)
            assertThrows<TimeoutException> {
                future.get(1, TimeUnit.SECONDS)
            }
        }

        @ParameterizedTest
        @ValueSource(strings = ["", "single", "multiple words", "special!@#$%^&*()chars"])
        @DisplayName("Should handle various input types")
        fun `should handle various input types`(input: String) {
            // Given
            val expectedOutput = "processed: $input"
            every { mockPipelineStage1.process(input) } returns expectedOutput
            processor.addStage(mockPipelineStage1)
            
            // When
            val result = processor.execute(input)
            
            // Then
            assertEquals(expectedOutput, result)
        }

        @Test
        @DisplayName("Should handle null input gracefully")
        fun `should handle null input gracefully`() {
            // Given
            val input: String? = null
            every { mockPipelineStage1.process(any()) } returns "null handled"
            processor.addStage(mockPipelineStage1)
            
            // When & Then
            assertDoesNotThrow { processor.execute(input) }
        }
    }

    @Nested
    @DisplayName("Error Handling Tests")
    inner class ErrorHandlingTests {

        @Test
        @DisplayName("Should handle stage processing exception")
        fun `should handle stage processing exception`() {
            // Given
            val input = "error input"
            val exception = RuntimeException("Processing failed")
            every { mockPipelineStage1.process(input) } throws exception
            processor.addStage(mockPipelineStage1)
            
            // When & Then
            assertThrows<PipelineProcessingException> {
                processor.execute(input)
            }
        }

        @Test
        @DisplayName("Should handle error with custom error handler")
        fun `should handle error with custom error handler`() {
            // Given
            val input = "error input"
            val exception = RuntimeException("Processing failed")
            val errorOutput = "error handled"
            
            every { mockPipelineStage1.process(input) } throws exception
            every { mockErrorHandler.handle(exception, input) } returns errorOutput
            
            processor.addStage(mockPipelineStage1)
            processor.setErrorHandler(mockErrorHandler)
            
            // When
            val result = processor.execute(input)
            
            // Then
            assertEquals(errorOutput, result)
            verify { mockErrorHandler.handle(exception, input) }
        }

        @Test
        @DisplayName("Should propagate error in async execution")
        fun `should propagate error in async execution`() {
            // Given
            val input = "async error input"
            val exception = RuntimeException("Async processing failed")
            every { mockPipelineStage1.process(input) } throws exception
            processor.addStage(mockPipelineStage1)
            
            // When
            val future = processor.executeAsync(input)
            
            // Then
            assertThrows<ExecutionException> {
                future.get(5, TimeUnit.SECONDS)
            }
        }

        @Test
        @DisplayName("Should handle multiple consecutive errors")
        fun `should handle multiple consecutive errors`() {
            // Given
            val input = "multi error input"
            val exception1 = RuntimeException("First error")
            val exception2 = RuntimeException("Second error")
            
            every { mockPipelineStage1.process(input) } throws exception1
            every { mockPipelineStage2.process(any()) } throws exception2
            every { mockErrorHandler.handle(exception1, input) } returns "error1 handled"
            every { mockErrorHandler.handle(exception2, "error1 handled") } returns "error2 handled"
            
            processor.addStage(mockPipelineStage1)
            processor.addStage(mockPipelineStage2)
            processor.setErrorHandler(mockErrorHandler)
            
            // When
            val result = processor.execute(input)
            
            // Then
            assertEquals("error2 handled", result)
        }

        @Test
        @DisplayName("Should handle error in error handler")
        fun `should handle error in error handler`() {
            // Given
            val input = "error handler error input"
            val originalException = RuntimeException("Original error")
            val handlerException = RuntimeException("Handler error")
            
            every { mockPipelineStage1.process(input) } throws originalException
            every { mockErrorHandler.handle(originalException, input) } throws handlerException
            
            processor.addStage(mockPipelineStage1)
            processor.setErrorHandler(mockErrorHandler)
            
            // When & Then
            assertThrows<PipelineProcessingException> {
                processor.execute(input)
            }
        }
    }

    @Nested
    @DisplayName("Pipeline State Management Tests")
    inner class PipelineStateManagementTests {

        @Test
        @DisplayName("Should track pipeline execution state")
        fun `should track pipeline execution state`() {
            // Given
            val input = "state input"
            every { mockPipelineStage1.process(input) } returns "state output"
            processor.addStage(mockPipelineStage1)
            
            // When
            assertFalse(processor.isExecuting())
            val future = processor.executeAsync(input)
            
            // Then
            // Note: This test depends on the actual implementation timing
            val result = future.get(5, TimeUnit.SECONDS)
            assertFalse(processor.isExecuting())
            assertEquals("state output", result)
        }

        @Test
        @DisplayName("Should prevent concurrent executions")
        fun `should prevent concurrent executions`() {
            // Given
            val input = "concurrent input"
            every { mockPipelineStage1.process(input) } answers {
                Thread.sleep(100)
                "concurrent output"
            }
            processor.addStage(mockPipelineStage1)
            
            // When
            val future1 = processor.executeAsync(input)
            val future2 = processor.executeAsync(input)
            
            // Then
            assertThrows<PipelineExecutionException> {
                future2.get(5, TimeUnit.SECONDS)
            }
        }

        @Test
        @DisplayName("Should cancel pipeline execution")
        fun `should cancel pipeline execution`() {
            // Given
            val input = "cancel input"
            every { mockPipelineStage1.process(input) } answers {
                Thread.sleep(5000)
                "cancel output"
            }
            processor.addStage(mockPipelineStage1)
            
            // When
            val future = processor.executeAsync(input)
            Thread.sleep(100) // Give it time to start
            processor.cancel()
            
            // Then
            assertTrue(future.isCancelled())
        }

        @Test
        @DisplayName("Should reset pipeline state")
        fun `should reset pipeline state`() {
            // Given
            processor.addStage(mockPipelineStage1)
            processor.setErrorHandler(mockErrorHandler)
            
            // When
            processor.reset()
            
            // Then
            assertEquals(0, processor.getStageCount())
            assertNull(processor.getErrorHandler())
        }
    }

    @Nested
    @DisplayName("Pipeline Metrics Tests")
    inner class PipelineMetricsTests {

        @Test
        @DisplayName("Should track execution time")
        fun `should track execution time`() {
            // Given
            val input = "metrics input"
            every { mockPipelineStage1.process(input) } answers {
                Thread.sleep(50)
                "metrics output"
            }
            processor.addStage(mockPipelineStage1)
            
            // When
            processor.execute(input)
            
            // Then
            val executionTime = processor.getLastExecutionTime()
            assertTrue(executionTime > 0)
        }

        @Test
        @DisplayName("Should track stage execution counts")
        fun `should track stage execution counts`() {
            // Given
            val input = "count input"
            every { mockPipelineStage1.process(input) } returns "count output"
            processor.addStage(mockPipelineStage1)
            
            // When
            processor.execute(input)
            processor.execute(input)
            processor.execute(input)
            
            // Then
            assertEquals(3, processor.getStageExecutionCount(mockPipelineStage1))
        }

        @Test
        @DisplayName("Should track error counts")
        fun `should track error counts`() {
            // Given
            val input = "error count input"
            every { mockPipelineStage1.process(input) } throws RuntimeException("Test error")
            every { mockErrorHandler.handle(any(), any()) } returns "error handled"
            processor.addStage(mockPipelineStage1)
            processor.setErrorHandler(mockErrorHandler)
            
            // When
            processor.execute(input)
            processor.execute(input)
            
            // Then
            assertEquals(2, processor.getErrorCount())
        }

        @Test
        @DisplayName("Should provide execution statistics")
        fun `should provide execution statistics`() {
            // Given
            val input = "stats input"
            every { mockPipelineStage1.process(input) } returns "stats output"
            processor.addStage(mockPipelineStage1)
            
            // When
            repeat(5) { processor.execute(input) }
            
            // Then
            val stats = processor.getExecutionStatistics()
            assertEquals(5, stats.totalExecutions)
            assertEquals(0, stats.totalErrors)
            assertTrue(stats.averageExecutionTime > 0)
        }
    }

    @Nested
    @DisplayName("Pipeline Validation Tests")
    inner class PipelineValidationTests {

        @Test
        @DisplayName("Should validate pipeline configuration")
        fun `should validate pipeline configuration`() {
            // Given
            processor.addStage(mockPipelineStage1)
            processor.addStage(mockPipelineStage2)
            
            // When
            val isValid = processor.validatePipeline()
            
            // Then
            assertTrue(isValid)
        }

        @Test
        @DisplayName("Should detect invalid pipeline configuration")
        fun `should detect invalid pipeline configuration`() {
            // Given
            every { mockPipelineStage1.isValid() } returns false
            processor.addStage(mockPipelineStage1)
            
            // When
            val isValid = processor.validatePipeline()
            
            // Then
            assertFalse(isValid)
        }

        @Test
        @DisplayName("Should provide validation errors")
        fun `should provide validation errors`() {
            // Given
            val validationError = "Stage configuration invalid"
            every { mockPipelineStage1.isValid() } returns false
            every { mockPipelineStage1.getValidationErrors() } returns listOf(validationError)
            processor.addStage(mockPipelineStage1)
            
            // When
            val errors = processor.getValidationErrors()
            
            // Then
            assertTrue(errors.contains(validationError))
        }
    }

    companion object {
        @JvmStatic
        fun provideTestInputs(): List<String> {
            return listOf(
                "simple input",
                "input with spaces",
                "input-with-dashes",
                "input_with_underscores",
                "InputWithCamelCase",
                "123456789",
                "special!@#$%^&*()chars"
            )
        }
    }
}