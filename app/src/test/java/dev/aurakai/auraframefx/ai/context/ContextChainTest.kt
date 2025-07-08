package dev.aurakai.auraframefx.ai.context

import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.Assertions.*
import org.mockito.MockitoAnnotations
import org.mockito.kotlin.*
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutionException

@DisplayName("ContextChain Tests")
class ContextChainTest {

    private lateinit var contextChain: ContextChain
    private lateinit var mockContext1: Context
    private lateinit var mockContext2: Context
    private lateinit var mockContext3: Context

    @BeforeEach
    fun setup() {
        MockitoAnnotations.openMocks(this)
        contextChain = ContextChain()
        mockContext1 = mock()
        mockContext2 = mock()
        mockContext3 = mock()
    }

    @Nested
    @DisplayName("Constructor Tests")
    inner class ConstructorTests {

        @Test
        @DisplayName("should create empty chain when no arguments provided")
        fun testEmptyConstructor() {
            val chain = ContextChain()
            assertTrue(chain.isEmpty())
            assertEquals(0, chain.size())
        }

        @Test
        @DisplayName("should create chain with single context")
        fun testSingleContextConstructor() {
            val chain = ContextChain(mockContext1)
            assertFalse(chain.isEmpty())
            assertEquals(1, chain.size())
            assertTrue(chain.contains(mockContext1))
        }

        @Test
        @DisplayName("should create chain with multiple contexts")
        fun testMultipleContextsConstructor() {
            val chain = ContextChain(mockContext1, mockContext2, mockContext3)
            assertEquals(3, chain.size())
            assertTrue(chain.contains(mockContext1))
            assertTrue(chain.contains(mockContext2))
            assertTrue(chain.contains(mockContext3))
        }

        @Test
        @DisplayName("should handle null contexts gracefully")
        fun testNullContextsHandling() {
            val chain = ContextChain(mockContext1, null, mockContext2)
            assertEquals(2, chain.size())
            assertTrue(chain.contains(mockContext1))
            assertTrue(chain.contains(mockContext2))
        }
    }

    @Nested
    @DisplayName("Chain Operations Tests")
    inner class ChainOperationsTests {

        @Test
        @DisplayName("should add context to chain")
        fun testAddContext() {
            contextChain.add(mockContext1)
            assertEquals(1, contextChain.size())
            assertTrue(contextChain.contains(mockContext1))
        }

        @Test
        @DisplayName("should add multiple contexts")
        fun testAddMultipleContexts() {
            contextChain.add(mockContext1)
            contextChain.add(mockContext2)
            contextChain.add(mockContext3)
            assertEquals(3, contextChain.size())
        }

        @Test
        @DisplayName("should remove context from chain")
        fun testRemoveContext() {
            contextChain.add(mockContext1)
            contextChain.add(mockContext2)

            val removed = contextChain.remove(mockContext1)
            assertTrue(removed)
            assertEquals(1, contextChain.size())
            assertFalse(contextChain.contains(mockContext1))
            assertTrue(contextChain.contains(mockContext2))
        }

        @Test
        @DisplayName("should return false when removing non-existent context")
        fun testRemoveNonExistentContext() {
            contextChain.add(mockContext1)
            val removed = contextChain.remove(mockContext2)
            assertFalse(removed)
            assertEquals(1, contextChain.size())
        }

        @Test
        @DisplayName("should clear all contexts")
        fun testClearChain() {
            contextChain.add(mockContext1)
            contextChain.add(mockContext2)
            contextChain.clear()
            assertTrue(contextChain.isEmpty())
            assertEquals(0, contextChain.size())
        }

        @Test
        @DisplayName("should handle duplicate contexts")
        fun testDuplicateContexts() {
            contextChain.add(mockContext1)
            contextChain.add(mockContext1)
            assertEquals(1, contextChain.size()) // Should not add duplicates
        }
    }

    @Nested
    @DisplayName("Chain Execution Tests")
    inner class ChainExecutionTests {

        @Test
        @DisplayName("should execute empty chain successfully")
        fun testExecuteEmptyChain() {
            val result = contextChain.execute()
            assertNotNull(result)
            assertTrue(result.isDone)
            assertFalse(result.isCompletedExceptionally)
        }

        @Test
        @DisplayName("should execute single context chain")
        fun testExecuteSingleContext() {
            val expectedResult = "context1_result"
            whenever(mockContext1.execute()).thenReturn(CompletableFuture.completedFuture(expectedResult))

            contextChain.add(mockContext1)
            val result = contextChain.execute()

            assertEquals(expectedResult, result.get())
            verify(mockContext1).execute()
        }

        @Test
        @DisplayName("should execute multiple contexts in sequence")
        fun testExecuteMultipleContexts() {
            val result1 = "result1"
            val result2 = "result2"
            val result3 = "result3"

            whenever(mockContext1.execute()).thenReturn(CompletableFuture.completedFuture(result1))
            whenever(mockContext2.execute()).thenReturn(CompletableFuture.completedFuture(result2))
            whenever(mockContext3.execute()).thenReturn(CompletableFuture.completedFuture(result3))

            contextChain.add(mockContext1)
            contextChain.add(mockContext2)
            contextChain.add(mockContext3)

            val result = contextChain.execute()

            assertNotNull(result.get())
            verify(mockContext1).execute()
            verify(mockContext2).execute()
            verify(mockContext3).execute()
        }

        @Test
        @DisplayName("should handle context execution failure")
        fun testExecuteWithFailure() {
            val exception = RuntimeException("Context execution failed")
            whenever(mockContext1.execute()).thenReturn(CompletableFuture.failedFuture(exception))

            contextChain.add(mockContext1)
            val result = contextChain.execute()

            assertThrows<ExecutionException> {
                result.get()
            }
            assertTrue(result.isCompletedExceptionally)
        }

        @Test
        @DisplayName("should handle timeout in context execution")
        fun testExecuteWithTimeout() {
            val slowFuture = CompletableFuture<String>()
            whenever(mockContext1.execute()).thenReturn(slowFuture)

            contextChain.add(mockContext1)
            val result = contextChain.executeWithTimeout(100) // 100ms timeout

            val ex = assertThrows<ExecutionException> {
                result.get()
            }
            assertTrue(ex.cause is java.util.concurrent.TimeoutException)
        }

        @Test
        @DisplayName("should cancel execution gracefully")
        fun testCancelExecution() {
            val slowFuture = CompletableFuture<String>()
            whenever(mockContext1.execute()).thenReturn(slowFuture)

            contextChain.add(mockContext1)
            val result = contextChain.execute()

            result.cancel(true)
            assertTrue(result.isCancelled)
        }
    }

    @Nested
    @DisplayName("Chain State Tests")
    inner class ChainStateTests {

        @Test
        @DisplayName("should report correct chain size")
        fun testChainSize() {
            assertEquals(0, contextChain.size())

            contextChain.add(mockContext1)
            assertEquals(1, contextChain.size())

            contextChain.add(mockContext2)
            assertEquals(2, contextChain.size())

            contextChain.remove(mockContext1)
            assertEquals(1, contextChain.size())
        }

        @Test
        @DisplayName("should report empty state correctly")
        fun testIsEmpty() {
            assertTrue(contextChain.isEmpty())

            contextChain.add(mockContext1)
            assertFalse(contextChain.isEmpty())

            contextChain.clear()
            assertTrue(contextChain.isEmpty())
        }

        @Test
        @DisplayName("should check context containment")
        fun testContains() {
            assertFalse(contextChain.contains(mockContext1))

            contextChain.add(mockContext1)
            assertTrue(contextChain.contains(mockContext1))
            assertFalse(contextChain.contains(mockContext2))
        }

        @Test
        @DisplayName("should iterate through contexts")
        fun testIteration() {
            contextChain.add(mockContext1)
            contextChain.add(mockContext2)
            contextChain.add(mockContext3)

            val contexts = mutableListOf<Context>()
            for (context in contextChain) {
                contexts.add(context)
            }

            assertEquals(3, contexts.size)
            assertTrue(contexts.contains(mockContext1))
            assertTrue(contexts.contains(mockContext2))
            assertTrue(contexts.contains(mockContext3))
        }
    }

    @Nested
    @DisplayName("Chain Validation Tests")
    inner class ChainValidationTests {

        @Test
        @DisplayName("should validate chain integrity")
        fun testValidateChain() {
            contextChain.add(mockContext1)
            contextChain.add(mockContext2)

            whenever(mockContext1.isValid()).thenReturn(true)
            whenever(mockContext2.isValid()).thenReturn(true)

            assertTrue(contextChain.isValid())
        }

        @Test
        @DisplayName("should report invalid chain when context is invalid")
        fun testInvalidChainValidation() {
            contextChain.add(mockContext1)
            contextChain.add(mockContext2)

            whenever(mockContext1.isValid()).thenReturn(true)
            whenever(mockContext2.isValid()).thenReturn(false)

            assertFalse(contextChain.isValid())
        }

        @Test
        @DisplayName("should validate empty chain as valid")
        fun testEmptyChainValidation() {
            assertTrue(contextChain.isValid())
        }
    }

    @Nested
    @DisplayName("Context Priority Tests")
    inner class ContextPriorityTests {

        @Test
        @DisplayName("should respect context priority order")
        fun testContextPriorityOrder() {
            val highPriorityContext = mock<Context>()
            val mediumPriorityContext = mock<Context>()
            val lowPriorityContext = mock<Context>()

            whenever(highPriorityContext.priority).thenReturn(1)
            whenever(mediumPriorityContext.priority).thenReturn(2)
            whenever(lowPriorityContext.priority).thenReturn(3)

            contextChain.add(lowPriorityContext)
            contextChain.add(highPriorityContext)
            contextChain.add(mediumPriorityContext)

            val sortedContexts = contextChain.getSortedByPriority()
            assertEquals(highPriorityContext, sortedContexts[0])
            assertEquals(mediumPriorityContext, sortedContexts[1])
            assertEquals(lowPriorityContext, sortedContexts[2])
        }

        @Test
        @DisplayName("should handle contexts with same priority")
        fun testSamePriorityContexts() {
            val context1 = mock<Context>()
            val context2 = mock<Context>()

            whenever(context1.priority).thenReturn(1)
            whenever(context2.priority).thenReturn(1)

            contextChain.add(context1)
            contextChain.add(context2)

            val sortedContexts = contextChain.getSortedByPriority()
            assertEquals(2, sortedContexts.size)
            assertTrue(sortedContexts.contains(context1))
            assertTrue(sortedContexts.contains(context2))
        }
    }

    @Nested
    @DisplayName("Chain Serialization Tests")
    inner class ChainSerializationTests {

        @Test
        @DisplayName("should serialize chain to JSON")
        fun testSerializeToJson() {
            contextChain.add(mockContext1)
            contextChain.add(mockContext2)

            whenever(mockContext1.toJson()).thenReturn("""{"type":"context1"}""")
            whenever(mockContext2.toJson()).thenReturn("""{"type":"context2"}""")

            val json = contextChain.toJson()
            assertNotNull(json)
            assertTrue(json.contains("context1"))
            assertTrue(json.contains("context2"))
        }

        @Test
        @DisplayName("should deserialize chain from JSON")
        fun testDeserializeFromJson() {
            val json = """{"contexts":[{"type":"context1"},{"type":"context2"}]}"""

            val chain = ContextChain.fromJson(json)
            assertNotNull(chain)
            assertEquals(2, chain.size())
        }

        @Test
        @DisplayName("should handle malformed JSON gracefully")
        fun testMalformedJsonHandling() {
            val malformedJson = """{"contexts":[{"type":"context1"}"""
            assertThrows<IllegalArgumentException> {
                ContextChain.fromJson(malformedJson)
            }
        }
    }

    @Nested
    @DisplayName("Chain Performance Tests")
    inner class ChainPerformanceTests {

        @Test
        @DisplayName("should handle large number of contexts efficiently")
        fun testLargeChainPerformance() {
            val contexts = mutableListOf<Context>()
            repeat(1000) {
                val context = mock<Context>()
                whenever(context.execute()).thenReturn(CompletableFuture.completedFuture("result$it"))
                contexts.add(context)
                contextChain.add(context)
            }

            assertEquals(1000, contextChain.size())

            val startTime = System.currentTimeMillis()
            val result = contextChain.execute()
            val endTime = System.currentTimeMillis()

            assertNotNull(result.get())
            assertTrue(endTime - startTime < 5000) // Should complete within 5 seconds
        }

        @Test
        @DisplayName("should handle concurrent modifications safely")
        fun testConcurrentModifications() {
            val threads = mutableListOf<Thread>()

            repeat(10) { threadIndex ->
                val thread = Thread {
                    repeat(100) {
                        val context = mock<Context>()
                        contextChain.add(context)
                        if (contextChain.size() > 50) {
                            contextChain.remove(context)
                        }
                    }
                }
                threads.add(thread)
                thread.start()
            }

            threads.forEach { it.join() }

            assertTrue(contextChain.size() >= 0)
            assertTrue(contextChain.size() <= 500)
        }
    }

    @Nested
    @DisplayName("Edge Cases Tests")
    inner class EdgeCasesTests {

        @Test
        @DisplayName("should handle null context additions gracefully")
        fun testNullContextAddition() {
            val initialSize = contextChain.size()
            contextChain.add(null)
            assertEquals(initialSize, contextChain.size())
        }

        @Test
        @DisplayName("should handle context with null execution result")
        fun testContextWithNullResult() {
            whenever(mockContext1.execute()).thenReturn(CompletableFuture.completedFuture(null))

            contextChain.add(mockContext1)
            val result = contextChain.execute()

            assertNull(result.get())
        }

        @Test
        @DisplayName("should handle context throwing unchecked exception")
        fun testContextThrowingException() {
            whenever(mockContext1.execute()).thenThrow(RuntimeException("Unexpected error"))

            contextChain.add(mockContext1)
            val result = contextChain.execute()

            val ex = assertThrows<ExecutionException> {
                result.get()
            }
            assertTrue(ex.cause is RuntimeException)
        }

        @Test
        @DisplayName("should handle memory pressure gracefully")
        fun testMemoryPressure() {
            // Create contexts that consume significant memory
            repeat(100) {
                val context = mock<Context>()
                val largeData = ByteArray(1024 * 1024) // 1MB
                whenever(context.execute()).thenReturn(CompletableFuture.completedFuture(largeData))
                contextChain.add(context)
            }

            // Should not throw OutOfMemoryError
            assertDoesNotThrow {
                contextChain.execute()
            }
        }
    }
}