package dev.aurakai.auraframefx.ai.context

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.assertThrows
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.kotlin.whenever
import org.mockito.kotlin.verify
import org.mockito.kotlin.any
import org.mockito.kotlin.never
import org.mockito.kotlin.times
import org.mockito.kotlin.reset
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutionException
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

@DisplayName("ContextChain Tests")
class ContextChainTest {

    private lateinit var contextChain: ContextChain
    private lateinit var autoCloseable: AutoCloseable
    
    @Mock
    private lateinit var mockContextProvider: ContextProvider
    
    @Mock
    private lateinit var mockContextProcessor: ContextProcessor
    
    @Mock
    private lateinit var mockCallback: (String) -> Unit

    @BeforeEach
    fun setUp() {
        autoCloseable = MockitoAnnotations.openMocks(this)
        contextChain = ContextChain()
    }

    @AfterEach
    fun tearDown() {
        autoCloseable.close()
    }

    @Nested
    @DisplayName("Basic Functionality Tests")
    inner class BasicFunctionalityTests {

        @Test
        @DisplayName("Should create empty context chain")
        fun shouldCreateEmptyContextChain() {
            assertTrue(contextChain.isEmpty())
            assertEquals(0, contextChain.size())
        }

        @Test
        @DisplayName("Should add single context provider")
        fun shouldAddSingleContextProvider() {
            contextChain.addProvider(mockContextProvider)
            
            assertFalse(contextChain.isEmpty())
            assertEquals(1, contextChain.size())
        }

        @Test
        @DisplayName("Should add multiple context providers")
        fun shouldAddMultipleContextProviders() {
            val provider1 = mockContextProvider
            val provider2 = mock<ContextProvider>()
            
            contextChain.addProvider(provider1)
            contextChain.addProvider(provider2)
            
            assertEquals(2, contextChain.size())
        }

        @Test
        @DisplayName("Should remove context provider")
        fun shouldRemoveContextProvider() {
            contextChain.addProvider(mockContextProvider)
            assertTrue(contextChain.removeProvider(mockContextProvider))
            
            assertTrue(contextChain.isEmpty())
            assertEquals(0, contextChain.size())
        }

        @Test
        @DisplayName("Should return false when removing non-existent provider")
        fun shouldReturnFalseWhenRemovingNonExistentProvider() {
            assertFalse(contextChain.removeProvider(mockContextProvider))
        }
    }

    @Nested
    @DisplayName("Context Processing Tests")
    inner class ContextProcessingTests {

        @Test
        @DisplayName("Should process context with single provider")
        fun shouldProcessContextWithSingleProvider() {
            val expectedContext = "test context"
            whenever(mockContextProvider.getContext()).thenReturn(expectedContext)
            
            contextChain.addProvider(mockContextProvider)
            val result = contextChain.processContext()
            
            assertEquals(expectedContext, result)
            verify(mockContextProvider).getContext()
        }

        @Test
        @DisplayName("Should process context with multiple providers")
        fun shouldProcessContextWithMultipleProviders() {
            val provider1 = mockContextProvider
            val provider2 = mock<ContextProvider>()
            val context1 = "context1"
            val context2 = "context2"
            
            whenever(provider1.getContext()).thenReturn(context1)
            whenever(provider2.getContext()).thenReturn(context2)
            
            contextChain.addProvider(provider1)
            contextChain.addProvider(provider2)
            
            val result = contextChain.processContext()
            
            assertTrue(result.contains(context1))
            assertTrue(result.contains(context2))
            verify(provider1).getContext()
            verify(provider2).getContext()
        }

        @Test
        @DisplayName("Should handle empty context from provider")
        fun shouldHandleEmptyContextFromProvider() {
            whenever(mockContextProvider.getContext()).thenReturn("")
            
            contextChain.addProvider(mockContextProvider)
            val result = contextChain.processContext()
            
            assertEquals("", result)
            verify(mockContextProvider).getContext()
        }

        @Test
        @DisplayName("Should handle null context from provider")
        fun shouldHandleNullContextFromProvider() {
            whenever(mockContextProvider.getContext()).thenReturn(null)
            
            contextChain.addProvider(mockContextProvider)
            val result = contextChain.processContext()
            
            assertNull(result)
            verify(mockContextProvider).getContext()
        }

        @Test
        @DisplayName("Should process context asynchronously")
        fun shouldProcessContextAsynchronously() {
            val expectedContext = "async context"
            whenever(mockContextProvider.getContext()).thenReturn(expectedContext)
            
            contextChain.addProvider(mockContextProvider)
            val future = contextChain.processContextAsync()
            
            val result = future.get(5, TimeUnit.SECONDS)
            assertEquals(expectedContext, result)
            verify(mockContextProvider).getContext()
        }

        @Test
        @DisplayName("Should handle timeout in async processing")
        fun shouldHandleTimeoutInAsyncProcessing() {
            whenever(mockContextProvider.getContext()).thenAnswer {
                Thread.sleep(6000) // Sleep longer than timeout
                "delayed context"
            }
            
            contextChain.addProvider(mockContextProvider)
            val future = contextChain.processContextAsync()
            
            assertThrows<TimeoutException> {
                future.get(1, TimeUnit.SECONDS)
            }
        }
    }

    @Nested
    @DisplayName("Error Handling Tests")
    inner class ErrorHandlingTests {

        @Test
        @DisplayName("Should handle provider exception gracefully")
        fun shouldHandleProviderExceptionGracefully() {
            val exception = RuntimeException("Provider error")
            whenever(mockContextProvider.getContext()).thenThrow(exception)
            
            contextChain.addProvider(mockContextProvider)
            
            assertThrows<RuntimeException> {
                contextChain.processContext()
            }
        }

        @Test
        @DisplayName("Should handle async processing exception")
        fun shouldHandleAsyncProcessingException() {
            val exception = RuntimeException("Async processing error")
            whenever(mockContextProvider.getContext()).thenThrow(exception)
            
            contextChain.addProvider(mockContextProvider)
            val future = contextChain.processContextAsync()
            
            assertThrows<ExecutionException> {
                future.get(5, TimeUnit.SECONDS)
            }
        }

        @Test
        @DisplayName("Should handle null provider addition")
        fun shouldHandleNullProviderAddition() {
            assertThrows<IllegalArgumentException> {
                contextChain.addProvider(null)
            }
        }

        @Test
        @DisplayName("Should handle duplicate provider addition")
        fun shouldHandleDuplicateProviderAddition() {
            contextChain.addProvider(mockContextProvider)
            contextChain.addProvider(mockContextProvider)
            
            // Should not add duplicate
            assertEquals(1, contextChain.size())
        }

        @Test
        @DisplayName("Should process context when empty chain")
        fun shouldProcessContextWhenEmptyChain() {
            val result = contextChain.processContext()
            
            assertTrue(result.isEmpty())
        }
    }

    @Nested
    @DisplayName("Callback and Listener Tests")
    inner class CallbackAndListenerTests {

        @Test
        @DisplayName("Should invoke callback on context processing")
        fun shouldInvokeCallbackOnContextProcessing() {
            val expectedContext = "callback context"
            whenever(mockContextProvider.getContext()).thenReturn(expectedContext)
            
            contextChain.addProvider(mockContextProvider)
            contextChain.setCallback(mockCallback)
            
            contextChain.processContext()
            
            verify(mockCallback).invoke(expectedContext)
        }

        @Test
        @DisplayName("Should not invoke callback when none set")
        fun shouldNotInvokeCallbackWhenNoneSet() {
            whenever(mockContextProvider.getContext()).thenReturn("context")
            
            contextChain.addProvider(mockContextProvider)
            contextChain.processContext()
            
            verify(mockCallback, never()).invoke(any())
        }

        @Test
        @DisplayName("Should handle callback exception")
        fun shouldHandleCallbackException() {
            whenever(mockContextProvider.getContext()).thenReturn("context")
            whenever(mockCallback.invoke(any())).thenThrow(RuntimeException("Callback error"))
            
            contextChain.addProvider(mockContextProvider)
            contextChain.setCallback(mockCallback)
            
            // Should not throw exception from callback
            assertDoesNotThrow {
                contextChain.processContext()
            }
        }
    }

    @Nested
    @DisplayName("Performance and Concurrency Tests")
    inner class PerformanceAndConcurrencyTests {

        @Test
        @DisplayName("Should handle concurrent context processing")
        fun shouldHandleConcurrentContextProcessing() {
            val provider1 = mockContextProvider
            val provider2 = mock<ContextProvider>()
            
            whenever(provider1.getContext()).thenReturn("context1")
            whenever(provider2.getContext()).thenReturn("context2")
            
            contextChain.addProvider(provider1)
            contextChain.addProvider(provider2)
            
            val futures = mutableListOf<CompletableFuture<String>>()
            
            repeat(10) {
                futures.add(contextChain.processContextAsync())
            }
            
            val results = futures.map { it.get(5, TimeUnit.SECONDS) }
            
            assertEquals(10, results.size)
            results.forEach { result ->
                assertTrue(result.contains("context1"))
                assertTrue(result.contains("context2"))
            }
        }

        @Test
        @DisplayName("Should handle large number of providers")
        fun shouldHandleLargeNumberOfProviders() {
            val providers = mutableListOf<ContextProvider>()
            
            repeat(100) { index ->
                val provider = mock<ContextProvider>()
                whenever(provider.getContext()).thenReturn("context$index")
                providers.add(provider)
                contextChain.addProvider(provider)
            }
            
            val result = contextChain.processContext()
            
            assertEquals(100, contextChain.size())
            providers.forEachIndexed { index, _ ->
                assertTrue(result.contains("context$index"))
            }
        }

        @Test
        @DisplayName("Should handle repeated processing efficiently")
        fun shouldHandleRepeatedProcessingEfficiently() {
            whenever(mockContextProvider.getContext()).thenReturn("repeated context")
            
            contextChain.addProvider(mockContextProvider)
            
            val startTime = System.currentTimeMillis()
            
            repeat(1000) {
                contextChain.processContext()
            }
            
            val endTime = System.currentTimeMillis()
            val duration = endTime - startTime
            
            // Should complete within reasonable time (less than 5 seconds)
            assertTrue(duration < 5000, "Processing took too long: ${duration}ms")
            verify(mockContextProvider, times(1000)).getContext()
        }
    }

    @Nested
    @DisplayName("State Management Tests")
    inner class StateManagementTests {

        @Test
        @DisplayName("Should maintain provider order")
        fun shouldMaintainProviderOrder() {
            val provider1 = mockContextProvider
            val provider2 = mock<ContextProvider>()
            val provider3 = mock<ContextProvider>()
            
            whenever(provider1.getContext()).thenReturn("first")
            whenever(provider2.getContext()).thenReturn("second")
            whenever(provider3.getContext()).thenReturn("third")
            
            contextChain.addProvider(provider1)
            contextChain.addProvider(provider2)
            contextChain.addProvider(provider3)
            
            val result = contextChain.processContext()
            
            val firstIndex = result.indexOf("first")
            val secondIndex = result.indexOf("second")
            val thirdIndex = result.indexOf("third")
            
            assertTrue(firstIndex < secondIndex)
            assertTrue(secondIndex < thirdIndex)
        }

        @Test
        @DisplayName("Should clear all providers")
        fun shouldClearAllProviders() {
            contextChain.addProvider(mockContextProvider)
            contextChain.addProvider(mock<ContextProvider>())
            
            contextChain.clear()
            
            assertTrue(contextChain.isEmpty())
            assertEquals(0, contextChain.size())
        }

        @Test
        @DisplayName("Should reset callback")
        fun shouldResetCallback() {
            contextChain.setCallback(mockCallback)
            contextChain.resetCallback()
            
            whenever(mockContextProvider.getContext()).thenReturn("context")
            contextChain.addProvider(mockContextProvider)
            contextChain.processContext()
            
            verify(mockCallback, never()).invoke(any())
        }
    }

    @Nested
    @DisplayName("Edge Cases and Boundary Tests")
    inner class EdgeCasesAndBoundaryTests {

        @Test
        @DisplayName("Should handle very long context strings")
        fun shouldHandleVeryLongContextStrings() {
            val longContext = "a".repeat(1000000) // 1MB string
            whenever(mockContextProvider.getContext()).thenReturn(longContext)
            
            contextChain.addProvider(mockContextProvider)
            val result = contextChain.processContext()
            
            assertEquals(longContext, result)
        }

        @Test
        @DisplayName("Should handle special characters in context")
        fun shouldHandleSpecialCharactersInContext() {
            val specialContext = "äöü@#$%^&*()[]{}|\\:;\"'<>,.?/~`"
            whenever(mockContextProvider.getContext()).thenReturn(specialContext)
            
            contextChain.addProvider(mockContextProvider)
            val result = contextChain.processContext()
            
            assertEquals(specialContext, result)
        }

        @Test
        @DisplayName("Should handle Unicode characters")
        fun shouldHandleUnicodeCharacters() {
            val unicodeContext = "测试 🚀 العربية हिंदी 🌟"
            whenever(mockContextProvider.getContext()).thenReturn(unicodeContext)
            
            contextChain.addProvider(mockContextProvider)
            val result = contextChain.processContext()
            
            assertEquals(unicodeContext, result)
        }

        @Test
        @DisplayName("Should handle mixed context types")
        fun shouldHandleMixedContextTypes() {
            val provider1 = mockContextProvider
            val provider2 = mock<ContextProvider>()
            val provider3 = mock<ContextProvider>()
            
            whenever(provider1.getContext()).thenReturn("string")
            whenever(provider2.getContext()).thenReturn("")
            whenever(provider3.getContext()).thenReturn(null)
            
            contextChain.addProvider(provider1)
            contextChain.addProvider(provider2)
            contextChain.addProvider(provider3)
            
            assertDoesNotThrow {
                contextChain.processContext()
            }
        }

        @Test
        @DisplayName("Should handle provider that returns same reference")
        fun shouldHandleProviderThatReturnsSameReference() {
            val sharedContext = "shared"
            whenever(mockContextProvider.getContext()).thenReturn(sharedContext)
            
            contextChain.addProvider(mockContextProvider)
            
            val result1 = contextChain.processContext()
            val result2 = contextChain.processContext()
            
            assertEquals(result1, result2)
        }
    }

    private inline fun <reified T : Any> mock(): T = org.mockito.kotlin.mock()
}