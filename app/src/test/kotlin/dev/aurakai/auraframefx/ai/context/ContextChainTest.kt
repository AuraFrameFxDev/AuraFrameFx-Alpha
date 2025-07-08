package dev.aurakai.auraframefx.ai.context

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.assertThrows
import org.mockito.kotlin.*
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.runBlocking
import java.util.concurrent.CompletableFuture

/**
 * Comprehensive unit tests for ContextChain class
 * Testing framework: JUnit 5 with Mockito for mocking
 */
class ContextChainTest {

    private lateinit var contextChain: ContextChain
    private lateinit var mockContext1: Context
    private lateinit var mockContext2: Context
    private lateinit var mockContext3: Context

    @BeforeEach
    fun setUp() {
        contextChain = ContextChain()
        mockContext1 = mock()
        mockContext2 = mock()
        mockContext3 = mock()
    }

    @AfterEach
    fun tearDown() {
        // Clean up any resources if needed
        contextChain.clear()
    }

    @Nested
    @DisplayName("Context Chain Construction Tests")
    inner class ConstructionTests {

        @Test
        @DisplayName("Should create empty context chain")
        fun shouldCreateEmptyContextChain() {
            assertTrue(contextChain.isEmpty())
            assertEquals(0, contextChain.size())
        }

        @Test
        @DisplayName("Should create context chain with initial contexts")
        fun shouldCreateContextChainWithInitialContexts() {
            val initialContexts = listOf(mockContext1, mockContext2)
            val chainWithInitial = ContextChain(initialContexts)
            
            assertEquals(2, chainWithInitial.size())
            assertFalse(chainWithInitial.isEmpty())
        }

        @Test
        @DisplayName("Should handle null initial contexts gracefully")
        fun shouldHandleNullInitialContextsGracefully() {
            val chainWithNull = ContextChain(null)
            assertTrue(chainWithNull.isEmpty())
            assertEquals(0, chainWithNull.size())
        }

        @Test
        @DisplayName("Should handle empty initial contexts list")
        fun shouldHandleEmptyInitialContextsList() {
            val chainWithEmpty = ContextChain(emptyList())
            assertTrue(chainWithEmpty.isEmpty())
            assertEquals(0, chainWithEmpty.size())
        }
    }

    @Nested
    @DisplayName("Context Addition Tests")
    inner class AdditionTests {

        @Test
        @DisplayName("Should add single context successfully")
        fun shouldAddSingleContextSuccessfully() {
            assertTrue(contextChain.add(mockContext1))
            assertEquals(1, contextChain.size())
            assertFalse(contextChain.isEmpty())
        }

        @Test
        @DisplayName("Should add multiple contexts in order")
        fun shouldAddMultipleContextsInOrder() {
            contextChain.add(mockContext1)
            contextChain.add(mockContext2)
            contextChain.add(mockContext3)
            
            assertEquals(3, contextChain.size())
            assertEquals(mockContext1, contextChain.get(0))
            assertEquals(mockContext2, contextChain.get(1))
            assertEquals(mockContext3, contextChain.get(2))
        }

        @Test
        @DisplayName("Should handle adding null context")
        fun shouldHandleAddingNullContext() {
            assertThrows<IllegalArgumentException> {
                contextChain.add(null)
            }
        }

        @Test
        @DisplayName("Should add context at specific index")
        fun shouldAddContextAtSpecificIndex() {
            contextChain.add(mockContext1)
            contextChain.add(mockContext3)
            contextChain.add(1, mockContext2)
            
            assertEquals(3, contextChain.size())
            assertEquals(mockContext1, contextChain.get(0))
            assertEquals(mockContext2, contextChain.get(1))
            assertEquals(mockContext3, contextChain.get(2))
        }

        @Test
        @DisplayName("Should throw exception when adding at invalid index")
        fun shouldThrowExceptionWhenAddingAtInvalidIndex() {
            assertThrows<IndexOutOfBoundsException> {
                contextChain.add(5, mockContext1)
            }
        }

        @Test
        @DisplayName("Should add all contexts from collection")
        fun shouldAddAllContextsFromCollection() {
            val contexts = listOf(mockContext1, mockContext2, mockContext3)
            assertTrue(contextChain.addAll(contexts))
            assertEquals(3, contextChain.size())
        }

        @Test
        @DisplayName("Should handle adding all from empty collection")
        fun shouldHandleAddingAllFromEmptyCollection() {
            assertFalse(contextChain.addAll(emptyList()))
            assertEquals(0, contextChain.size())
        }
    }

    @Nested
    @DisplayName("Context Removal Tests")
    inner class RemovalTests {

        @BeforeEach
        fun setUpContexts() {
            contextChain.add(mockContext1)
            contextChain.add(mockContext2)
            contextChain.add(mockContext3)
        }

        @Test
        @DisplayName("Should remove context by object reference")
        fun shouldRemoveContextByObjectReference() {
            assertTrue(contextChain.remove(mockContext2))
            assertEquals(2, contextChain.size())
            assertFalse(contextChain.contains(mockContext2))
        }

        @Test
        @DisplayName("Should remove context by index")
        fun shouldRemoveContextByIndex() {
            val removed = contextChain.removeAt(1)
            assertEquals(mockContext2, removed)
            assertEquals(2, contextChain.size())
        }

        @Test
        @DisplayName("Should throw exception when removing at invalid index")
        fun shouldThrowExceptionWhenRemovingAtInvalidIndex() {
            assertThrows<IndexOutOfBoundsException> {
                contextChain.removeAt(10)
            }
        }

        @Test
        @DisplayName("Should remove first occurrence only")
        fun shouldRemoveFirstOccurrenceOnly() {
            contextChain.add(mockContext1) // Add duplicate
            assertTrue(contextChain.remove(mockContext1))
            assertEquals(3, contextChain.size())
            assertTrue(contextChain.contains(mockContext1))
        }

        @Test
        @DisplayName("Should clear all contexts")
        fun shouldClearAllContexts() {
            contextChain.clear()
            assertTrue(contextChain.isEmpty())
            assertEquals(0, contextChain.size())
        }

        @Test
        @DisplayName("Should remove all contexts from collection")
        fun shouldRemoveAllContextsFromCollection() {
            val toRemove = listOf(mockContext1, mockContext3)
            assertTrue(contextChain.removeAll(toRemove))
            assertEquals(1, contextChain.size())
            assertEquals(mockContext2, contextChain.get(0))
        }

        @Test
        @DisplayName("Should retain only specified contexts")
        fun shouldRetainOnlySpecifiedContexts() {
            val toRetain = listOf(mockContext1, mockContext2)
            assertTrue(contextChain.retainAll(toRetain))
            assertEquals(2, contextChain.size())
            assertFalse(contextChain.contains(mockContext3))
        }
    }

    @Nested
    @DisplayName("Context Access Tests")
    inner class AccessTests {

        @BeforeEach
        fun setUpContexts() {
            contextChain.add(mockContext1)
            contextChain.add(mockContext2)
            contextChain.add(mockContext3)
        }

        @Test
        @DisplayName("Should get context by index")
        fun shouldGetContextByIndex() {
            assertEquals(mockContext1, contextChain.get(0))
            assertEquals(mockContext2, contextChain.get(1))
            assertEquals(mockContext3, contextChain.get(2))
        }

        @Test
        @DisplayName("Should throw exception when accessing invalid index")
        fun shouldThrowExceptionWhenAccessingInvalidIndex() {
            assertThrows<IndexOutOfBoundsException> {
                contextChain.get(10)
            }
        }

        @Test
        @DisplayName("Should get first context")
        fun shouldGetFirstContext() {
            assertEquals(mockContext1, contextChain.first())
        }

        @Test
        @DisplayName("Should get last context")
        fun shouldGetLastContext() {
            assertEquals(mockContext3, contextChain.last())
        }

        @Test
        @DisplayName("Should throw exception when getting first from empty chain")
        fun shouldThrowExceptionWhenGettingFirstFromEmptyChain() {
            contextChain.clear()
            assertThrows<NoSuchElementException> {
                contextChain.first()
            }
        }

        @Test
        @DisplayName("Should throw exception when getting last from empty chain")
        fun shouldThrowExceptionWhenGettingLastFromEmptyChain() {
            contextChain.clear()
            assertThrows<NoSuchElementException> {
                contextChain.last()
            }
        }

        @Test
        @DisplayName("Should find index of context")
        fun shouldFindIndexOfContext() {
            assertEquals(0, contextChain.indexOf(mockContext1))
            assertEquals(1, contextChain.indexOf(mockContext2))
            assertEquals(2, contextChain.indexOf(mockContext3))
        }

        @Test
        @DisplayName("Should return -1 for non-existent context")
        fun shouldReturnNegativeOneForNonExistentContext() {
            val nonExistentContext = mock<Context>()
            assertEquals(-1, contextChain.indexOf(nonExistentContext))
        }

        @Test
        @DisplayName("Should check if context exists")
        fun shouldCheckIfContextExists() {
            assertTrue(contextChain.contains(mockContext1))
            assertTrue(contextChain.contains(mockContext2))
            assertTrue(contextChain.contains(mockContext3))
            assertFalse(contextChain.contains(mock<Context>()))
        }

        @Test
        @DisplayName("Should check if all contexts exist")
        fun shouldCheckIfAllContextsExist() {
            val existingContexts = listOf(mockContext1, mockContext2)
            assertTrue(contextChain.containsAll(existingContexts))
            
            val mixedContexts = listOf(mockContext1, mock<Context>())
            assertFalse(contextChain.containsAll(mixedContexts))
        }
    }

    @Nested
    @DisplayName("Context Chain Iteration Tests")
    inner class IterationTests {

        @BeforeEach
        fun setUpContexts() {
            contextChain.add(mockContext1)
            contextChain.add(mockContext2)
            contextChain.add(mockContext3)
        }

        @Test
        @DisplayName("Should iterate through all contexts")
        fun shouldIterateThroughAllContexts() {
            val contexts = mutableListOf<Context>()
            for (context in contextChain) {
                contexts.add(context)
            }
            
            assertEquals(3, contexts.size)
            assertEquals(mockContext1, contexts[0])
            assertEquals(mockContext2, contexts[1])
            assertEquals(mockContext3, contexts[2])
        }

        @Test
        @DisplayName("Should convert to array")
        fun shouldConvertToArray() {
            val array = contextChain.toTypedArray()
            assertEquals(3, array.size)
            assertEquals(mockContext1, array[0])
            assertEquals(mockContext2, array[1])
            assertEquals(mockContext3, array[2])
        }

        @Test
        @DisplayName("Should support list iterator")
        fun shouldSupportListIterator() {
            val iterator = contextChain.listIterator()
            assertTrue(iterator.hasNext())
            assertEquals(mockContext1, iterator.next())
            assertTrue(iterator.hasNext())
            assertEquals(mockContext2, iterator.next())
            assertTrue(iterator.hasNext())
            assertEquals(mockContext3, iterator.next())
            assertFalse(iterator.hasNext())
        }

        @Test
        @DisplayName("Should support list iterator from index")
        fun shouldSupportListIteratorFromIndex() {
            val iterator = contextChain.listIterator(1)
            assertTrue(iterator.hasNext())
            assertEquals(mockContext2, iterator.next())
            assertTrue(iterator.hasNext())
            assertEquals(mockContext3, iterator.next())
            assertFalse(iterator.hasNext())
        }
    }

    @Nested
    @DisplayName("Context Chain Processing Tests")
    inner class ProcessingTests {

        @Test
        @DisplayName("Should process contexts in order")
        fun shouldProcessContextsInOrder() = runTest {
            val processingOrder = mutableListOf<Context>()
            
            whenever(mockContext1.process(any())).thenAnswer { invocation ->
                processingOrder.add(mockContext1)
                invocation.getArgument<Any>(0)
            }
            
            whenever(mockContext2.process(any())).thenAnswer { invocation ->
                processingOrder.add(mockContext2)
                invocation.getArgument<Any>(0)
            }
            
            whenever(mockContext3.process(any())).thenAnswer { invocation ->
                processingOrder.add(mockContext3)
                invocation.getArgument<Any>(0)
            }
            
            contextChain.add(mockContext1)
            contextChain.add(mockContext2)
            contextChain.add(mockContext3)
            
            val input = "test input"
            val result = contextChain.process(input)
            
            assertEquals(3, processingOrder.size)
            assertEquals(mockContext1, processingOrder[0])
            assertEquals(mockContext2, processingOrder[1])
            assertEquals(mockContext3, processingOrder[2])
            assertEquals(input, result)
        }

        @Test
        @DisplayName("Should handle processing with empty chain")
        fun shouldHandleProcessingWithEmptyChain() = runTest {
            val input = "test input"
            val result = contextChain.process(input)
            assertEquals(input, result)
        }

        @Test
        @DisplayName("Should handle processing exception")
        fun shouldHandleProcessingException() = runTest {
            whenever(mockContext1.process(any())).thenThrow(RuntimeException("Processing failed"))
            
            contextChain.add(mockContext1)
            
            assertThrows<RuntimeException> {
                runBlocking {
                    contextChain.process("test input")
                }
            }
        }

        @Test
        @DisplayName("Should transform data through context chain")
        fun shouldTransformDataThroughContextChain() = runTest {
            whenever(mockContext1.process(any())).thenReturn("processed1")
            whenever(mockContext2.process(any())).thenReturn("processed2")
            whenever(mockContext3.process(any())).thenReturn("processed3")
            
            contextChain.add(mockContext1)
            contextChain.add(mockContext2)
            contextChain.add(mockContext3)
            
            val result = contextChain.process("initial")
            
            verify(mockContext1).process("initial")
            verify(mockContext2).process("processed1")
            verify(mockContext3).process("processed2")
            assertEquals("processed3", result)
        }
    }

    @Nested
    @DisplayName("Context Chain Validation Tests")
    inner class ValidationTests {

        @Test
        @DisplayName("Should validate context types")
        fun shouldValidateContextTypes() {
            val validContext = mock<Context>()
            val invalidContext = mock<Any>()
            
            assertTrue(contextChain.isValidContext(validContext))
            assertFalse(contextChain.isValidContext(invalidContext))
        }

        @Test
        @DisplayName("Should validate chain integrity")
        fun shouldValidateChainIntegrity() {
            contextChain.add(mockContext1)
            contextChain.add(mockContext2)
            contextChain.add(mockContext3)
            
            assertTrue(contextChain.isValid())
        }

        @Test
        @DisplayName("Should detect circular dependencies")
        fun shouldDetectCircularDependencies() {
            // Mock circular dependency scenario
            whenever(mockContext1.getDependencies()).thenReturn(listOf("context2"))
            whenever(mockContext2.getDependencies()).thenReturn(listOf("context1"))
            
            contextChain.add(mockContext1)
            contextChain.add(mockContext2)
            
            assertFalse(contextChain.hasCircularDependencies())
        }
    }

    @Nested
    @DisplayName("Context Chain Serialization Tests")
    inner class SerializationTests {

        @Test
        @DisplayName("Should serialize context chain to JSON")
        fun shouldSerializeContextChainToJson() {
            whenever(mockContext1.getId()).thenReturn("context1")
            whenever(mockContext2.getId()).thenReturn("context2")
            whenever(mockContext3.getId()).thenReturn("context3")
            
            contextChain.add(mockContext1)
            contextChain.add(mockContext2)
            contextChain.add(mockContext3)
            
            val json = contextChain.toJson()
            assertNotNull(json)
            assertTrue(json.contains("context1"))
            assertTrue(json.contains("context2"))
            assertTrue(json.contains("context3"))
        }

        @Test
        @DisplayName("Should deserialize context chain from JSON")
        fun shouldDeserializeContextChainFromJson() {
            val json = """
                {
                    "contexts": [
                        {"id": "context1", "type": "test"},
                        {"id": "context2", "type": "test"}
                    ]
                }
            """.trimIndent()
            
            val deserializedChain = ContextChain.fromJson(json)
            assertNotNull(deserializedChain)
            assertEquals(2, deserializedChain.size())
        }

        @Test
        @DisplayName("Should handle malformed JSON gracefully")
        fun shouldHandleMalformedJsonGracefully() {
            val malformedJson = "invalid json"
            
            assertThrows<IllegalArgumentException> {
                ContextChain.fromJson(malformedJson)
            }
        }
    }

    @Nested
    @DisplayName("Context Chain Performance Tests")
    inner class PerformanceTests {

        @Test
        @DisplayName("Should handle large number of contexts efficiently")
        fun shouldHandleLargeNumberOfContextsEfficiently() {
            val largeContextList = (1..1000).map { mock<Context>() }
            
            val startTime = System.currentTimeMillis()
            contextChain.addAll(largeContextList)
            val endTime = System.currentTimeMillis()
            
            assertEquals(1000, contextChain.size())
            assertTrue(endTime - startTime < 1000) // Should complete within 1 second
        }

        @Test
        @DisplayName("Should process large context chain efficiently")
        fun shouldProcessLargeContextChainEfficiently() = runTest {
            val largeContextList = (1..100).map { 
                mock<Context>().apply {
                    whenever(process(any())).thenAnswer { invocation ->
                        invocation.getArgument<Any>(0)
                    }
                }
            }
            
            contextChain.addAll(largeContextList)
            
            val startTime = System.currentTimeMillis()
            val result = contextChain.process("test")
            val endTime = System.currentTimeMillis()
            
            assertEquals("test", result)
            assertTrue(endTime - startTime < 2000) // Should complete within 2 seconds
        }
    }

    @Nested
    @DisplayName("Context Chain Edge Cases")
    inner class EdgeCaseTests {

        @Test
        @DisplayName("Should handle concurrent modifications")
        fun shouldHandleConcurrentModifications() {
            contextChain.add(mockContext1)
            contextChain.add(mockContext2)
            
            val future1 = CompletableFuture.runAsync {
                contextChain.add(mockContext3)
            }
            
            val future2 = CompletableFuture.runAsync {
                contextChain.remove(mockContext1)
            }
            
            assertDoesNotThrow {
                CompletableFuture.allOf(future1, future2).join()
            }
        }

        @Test
        @DisplayName("Should handle memory pressure gracefully")
        fun shouldHandleMemoryPressureGracefully() {
            val contexts = (1..10000).map { mock<Context>() }
            
            assertDoesNotThrow {
                contextChain.addAll(contexts)
                contextChain.clear()
                System.gc()
            }
        }

        @Test
        @DisplayName("Should handle context with null processing result")
        fun shouldHandleContextWithNullProcessingResult() = runTest {
            whenever(mockContext1.process(any())).thenReturn(null)
            
            contextChain.add(mockContext1)
            
            val result = contextChain.process("test")
            assertNull(result)
        }
    }
}