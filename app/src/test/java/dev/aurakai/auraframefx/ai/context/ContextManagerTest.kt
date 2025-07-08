package dev.aurakai.auraframefx.ai.context

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.Assertions.*
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.Mockito.*
import org.mockito.kotlin.whenever
import org.mockito.kotlin.verify
import org.mockito.kotlin.any
import org.mockito.kotlin.never
import org.mockito.kotlin.times
import java.util.concurrent.ConcurrentHashMap
import kotlin.test.assertNotNull

@DisplayName("ContextManager Tests")
class ContextManagerTest {

    private lateinit var contextManager: ContextManager
    private lateinit var mockCloseable: AutoCloseable

    @BeforeEach
    fun setUp() {
        MockitoAnnotations.openMocks(this)
        mockCloseable = mock(AutoCloseable::class.java)
        contextManager = ContextManager()
    }

    @AfterEach
    fun tearDown() {
        // Clean up any resources
        contextManager.clear()
    }

    @Nested
    @DisplayName("Context Storage Tests")
    inner class ContextStorageTests {

        @Test
        @DisplayName("Should store and retrieve string context successfully")
        fun shouldStoreAndRetrieveStringContext() {
            // Given
            val key = "test_key"
            val value = "test_value"

            // When
            contextManager.setContext(key, value)
            val retrieved = contextManager.getContext(key)

            // Then
            assertEquals(value, retrieved)
        }

        @Test
        @DisplayName("Should store and retrieve complex object context")
        fun shouldStoreAndRetrieveComplexObjectContext() {
            // Given
            val key = "complex_key"
            val value = mapOf("nested" to "value", "number" to 42)

            // When
            contextManager.setContext(key, value)
            val retrieved = contextManager.getContext(key)

            // Then
            assertEquals(value, retrieved)
        }

        @Test
        @DisplayName("Should return null for non-existent context key")
        fun shouldReturnNullForNonExistentKey() {
            // Given
            val nonExistentKey = "non_existent"

            // When
            val retrieved = contextManager.getContext(nonExistentKey)

            // Then
            assertNull(retrieved)
        }

        @Test
        @DisplayName("Should overwrite existing context value")
        fun shouldOverwriteExistingContextValue() {
            // Given
            val key = "overwrite_key"
            val originalValue = "original"
            val newValue = "new"

            // When
            contextManager.setContext(key, originalValue)
            contextManager.setContext(key, newValue)
            val retrieved = contextManager.getContext(key)

            // Then
            assertEquals(newValue, retrieved)
        }

        @Test
        @DisplayName("Should handle null values correctly")
        fun shouldHandleNullValues() {
            // Given
            val key = "null_key"

            // When
            contextManager.setContext(key, null)
            val retrieved = contextManager.getContext(key)

            // Then
            assertNull(retrieved)
        }

        @Test
        @DisplayName("Should handle empty string keys")
        fun shouldHandleEmptyStringKeys() {
            // Given
            val emptyKey = ""
            val value = "empty_key_value"

            // When
            contextManager.setContext(emptyKey, value)
            val retrieved = contextManager.getContext(emptyKey)

            // Then
            assertEquals(value, retrieved)
        }

        @Test
        @DisplayName("Should handle special characters in keys")
        fun shouldHandleSpecialCharactersInKeys() {
            // Given
            val specialKey = "key!@#$%^&*()_+-=[]{}|;':\",./<>?"
            val value = "special_chars_value"

            // When
            contextManager.setContext(specialKey, value)
            val retrieved = contextManager.getContext(specialKey)

            // Then
            assertEquals(value, retrieved)
        }
    }

    @Nested
    @DisplayName("Context Removal Tests")
    inner class ContextRemovalTests {

        @Test
        @DisplayName("Should remove existing context successfully")
        fun shouldRemoveExistingContext() {
            // Given
            val key = "remove_key"
            val value = "remove_value"
            contextManager.setContext(key, value)

            // When
            contextManager.removeContext(key)
            val retrieved = contextManager.getContext(key)

            // Then
            assertNull(retrieved)
        }

        @Test
        @DisplayName("Should handle removal of non-existent context gracefully")
        fun shouldHandleRemovalOfNonExistentContextGracefully() {
            // Given
            val nonExistentKey = "non_existent"

            // When & Then
            assertDoesNotThrow {
                contextManager.removeContext(nonExistentKey)
            }
        }

        @Test
        @DisplayName("Should clear all context successfully")
        fun shouldClearAllContext() {
            // Given
            contextManager.setContext("key1", "value1")
            contextManager.setContext("key2", "value2")
            contextManager.setContext("key3", "value3")

            // When
            contextManager.clear()

            // Then
            assertNull(contextManager.getContext("key1"))
            assertNull(contextManager.getContext("key2"))
            assertNull(contextManager.getContext("key3"))
        }
    }

    @Nested
    @DisplayName("Context Querying Tests")
    inner class ContextQueryingTests {

        @Test
        @DisplayName("Should check if context exists correctly")
        fun shouldCheckIfContextExistsCorrectly() {
            // Given
            val existingKey = "existing_key"
            val nonExistentKey = "non_existent_key"
            contextManager.setContext(existingKey, "value")

            // When & Then
            assertTrue(contextManager.hasContext(existingKey))
            assertFalse(contextManager.hasContext(nonExistentKey))
        }

        @Test
        @DisplayName("Should return all context keys")
        fun shouldReturnAllContextKeys() {
            // Given
            val keys = setOf("key1", "key2", "key3")
            keys.forEach { contextManager.setContext(it, "value") }

            // When
            val retrievedKeys = contextManager.getContextKeys()

            // Then
            assertEquals(keys, retrievedKeys.toSet())
        }

        @Test
        @DisplayName("Should return empty set when no context exists")
        fun shouldReturnEmptySetWhenNoContextExists() {
            // When
            val keys = contextManager.getContextKeys()

            // Then
            assertTrue(keys.isEmpty())
        }

        @Test
        @DisplayName("Should return context size correctly")
        fun shouldReturnContextSizeCorrectly() {
            // Given
            contextManager.setContext("key1", "value1")
            contextManager.setContext("key2", "value2")

            // When
            val size = contextManager.getContextSize()

            // Then
            assertEquals(2, size)
        }
    }

    @Nested
    @DisplayName("Context Typing Tests")
    inner class ContextTypingTests {

        @Test
        @DisplayName("Should retrieve typed context successfully")
        fun shouldRetrieveTypedContextSuccessfully() {
            // Given
            val key = "typed_key"
            val value: List<String> = listOf("item1", "item2", "item3")
            contextManager.setContext(key, value)

            // When
            val retrieved = contextManager.getTypedContext<List<String>>(key)

            // Then
            assertEquals(value, retrieved)
        }

        @Test
        @DisplayName("Should throw exception for incorrect type casting")
        fun shouldThrowExceptionForIncorrectTypeCasting() {
            // Given
            val key = "string_key"
            val value = "string_value"
            contextManager.setContext(key, value)

            // When & Then
            assertThrows<ClassCastException> {
                contextManager.getTypedContext<Int>(key)
            }
        }

        @Test
        @DisplayName("Should return null for non-existent typed context")
        fun shouldReturnNullForNonExistentTypedContext() {
            // Given
            val nonExistentKey = "non_existent_typed"

            // When
            val retrieved = contextManager.getTypedContext<String>(nonExistentKey)

            // Then
            assertNull(retrieved)
        }
    }

    @Nested
    @DisplayName("Context Scoping Tests")
    inner class ContextScopingTests {

        @Test
        @DisplayName("Should create and manage scoped context")
        fun shouldCreateAndManageScopedContext() {
            // Given
            val scopeName = "test_scope"
            val key = "scoped_key"
            val value = "scoped_value"

            // When
            contextManager.createScope(scopeName)
            contextManager.setScopedContext(scopeName, key, value)
            val retrieved = contextManager.getScopedContext(scopeName, key)

            // Then
            assertEquals(value, retrieved)
        }

        @Test
        @DisplayName("Should isolate scoped contexts")
        fun shouldIsolateScopedContexts() {
            // Given
            val scope1 = "scope1"
            val scope2 = "scope2"
            val key = "same_key"
            val value1 = "value1"
            val value2 = "value2"

            // When
            contextManager.createScope(scope1)
            contextManager.createScope(scope2)
            contextManager.setScopedContext(scope1, key, value1)
            contextManager.setScopedContext(scope2, key, value2)

            // Then
            assertEquals(value1, contextManager.getScopedContext(scope1, key))
            assertEquals(value2, contextManager.getScopedContext(scope2, key))
        }

        @Test
        @DisplayName("Should remove scoped context successfully")
        fun shouldRemoveScopedContextSuccessfully() {
            // Given
            val scopeName = "remove_scope"
            val key = "remove_key"
            val value = "remove_value"
            contextManager.createScope(scopeName)
            contextManager.setScopedContext(scopeName, key, value)

            // When
            contextManager.removeScope(scopeName)
            val retrieved = contextManager.getScopedContext(scopeName, key)

            // Then
            assertNull(retrieved)
        }

        @Test
        @DisplayName("Should throw exception for non-existent scope")
        fun shouldThrowExceptionForNonExistentScope() {
            // Given
            val nonExistentScope = "non_existent_scope"

            // When & Then
            assertThrows<IllegalArgumentException> {
                contextManager.setScopedContext(nonExistentScope, "key", "value")
            }
        }
    }

    @Nested
    @DisplayName("Thread Safety Tests")
    inner class ThreadSafetyTests {

        @Test
        @DisplayName("Should handle concurrent access safely")
        fun shouldHandleConcurrentAccessSafely() {
            // Given
            val threads = mutableListOf<Thread>()
            val results = ConcurrentHashMap<String, String>()

            // When
            for (i in 1..10) {
                val thread = Thread {
                    val key = "thread_key_$i"
                    val value = "thread_value_$i"
                    contextManager.setContext(key, value)
                    results[key] = contextManager.getContext(key) as String
                }
                threads.add(thread)
                thread.start()
            }

            threads.forEach { it.join() }

            // Then
            assertEquals(10, results.size)
            for (i in 1..10) {
                assertEquals("thread_value_$i", results["thread_key_$i"])
            }
        }

        @Test
        @DisplayName("Should handle concurrent modifications safely")
        fun shouldHandleConcurrentModificationsSafely() {
            // Given
            val key = "concurrent_key"
            val threads = mutableListOf<Thread>()

            // When
            for (i in 1..5) {
                val thread = Thread {
                    for (j in 1..100) {
                        contextManager.setContext(key, "value_${i}_$j")
                    }
                }
                threads.add(thread)
                thread.start()
            }

            threads.forEach { it.join() }

            // Then
            assertNotNull(contextManager.getContext(key))
            assertTrue(contextManager.hasContext(key))
        }
    }

    @Nested
    @DisplayName("Context Validation Tests")
    inner class ContextValidationTests {

        @Test
        @DisplayName("Should validate context with custom validator")
        fun shouldValidateContextWithCustomValidator() {
            // Given
            val key = "validated_key"
            val value = "valid_value"
            val validator: (Any?) -> Boolean = { it is String && it.isNotEmpty() }

            // When
            contextManager.setContextWithValidation(key, value, validator)
            val retrieved = contextManager.getContext(key)

            // Then
            assertEquals(value, retrieved)
        }

        @Test
        @DisplayName("Should reject invalid context with custom validator")
        fun shouldRejectInvalidContextWithCustomValidator() {
            // Given
            val key = "invalid_key"
            val value = ""
            val validator: (Any?) -> Boolean = { it is String && it.isNotEmpty() }

            // When & Then
            assertThrows<IllegalArgumentException> {
                contextManager.setContextWithValidation(key, value, validator)
            }
        }

        @Test
        @DisplayName("Should handle null validation gracefully")
        fun shouldHandleNullValidationGracefully() {
            // Given
            val key = "null_validation_key"
            val value = "some_value"
            val validator: (Any?) -> Boolean = { it != null }

            // When
            contextManager.setContextWithValidation(key, value, validator)
            val retrieved = contextManager.getContext(key)

            // Then
            assertEquals(value, retrieved)
        }
    }

    @Nested
    @DisplayName("Context Lifecycle Tests")
    inner class ContextLifecycleTests {

        @Test
        @DisplayName("Should handle context lifecycle events")
        fun shouldHandleContextLifecycleEvents() {
            // Given
            val key = "lifecycle_key"
            val value = "lifecycle_value"
            var onSetCalled = false
            var onGetCalled = false
            var onRemoveCalled = false

            val listener = object : ContextLifecycleListener {
                override fun onContextSet(key: String, value: Any?) {
                    onSetCalled = true
                }

                override fun onContextGet(key: String, value: Any?) {
                    onGetCalled = true
                }

                override fun onContextRemove(key: String) {
                    onRemoveCalled = true
                }
            }

            // When
            contextManager.addLifecycleListener(listener)
            contextManager.setContext(key, value)
            contextManager.getContext(key)
            contextManager.removeContext(key)

            // Then
            assertTrue(onSetCalled)
            assertTrue(onGetCalled)
            assertTrue(onRemoveCalled)
        }

        @Test
        @DisplayName("Should remove lifecycle listener successfully")
        fun shouldRemoveLifecycleListenerSuccessfully() {
            // Given
            val key = "listener_key"
            val value = "listener_value"
            var eventFired = false

            val listener = object : ContextLifecycleListener {
                override fun onContextSet(key: String, value: Any?) {
                    eventFired = true
                }

                override fun onContextGet(key: String, value: Any?) {}
                override fun onContextRemove(key: String) {}
            }

            // When
            contextManager.addLifecycleListener(listener)
            contextManager.removeLifecycleListener(listener)
            contextManager.setContext(key, value)

            // Then
            assertFalse(eventFired)
        }
    }

    @Nested
    @DisplayName("Context Persistence Tests")
    inner class ContextPersistenceTests {

        @Test
        @DisplayName("Should save context state to storage")
        fun shouldSaveContextStateToStorage() {
            // Given
            val key1 = "persist_key1"
            val value1 = "persist_value1"
            val key2 = "persist_key2"
            val value2 = "persist_value2"

            contextManager.setContext(key1, value1)
            contextManager.setContext(key2, value2)

            // When
            val snapshot = contextManager.createSnapshot()

            // Then
            assertNotNull(snapshot)
            assertEquals(2, snapshot.size)
            assertEquals(value1, snapshot[key1])
            assertEquals(value2, snapshot[key2])
        }

        @Test
        @DisplayName("Should restore context state from storage")
        fun shouldRestoreContextStateFromStorage() {
            // Given
            val snapshot = mapOf(
                "restore_key1" to "restore_value1",
                "restore_key2" to "restore_value2"
            )

            // When
            contextManager.restoreFromSnapshot(snapshot)

            // Then
            assertEquals("restore_value1", contextManager.getContext("restore_key1"))
            assertEquals("restore_value2", contextManager.getContext("restore_key2"))
        }

        @Test
        @DisplayName("Should handle empty snapshot gracefully")
        fun shouldHandleEmptySnapshotGracefully() {
            // Given
            val emptySnapshot = emptyMap<String, Any?>()

            // When & Then
            assertDoesNotThrow {
                contextManager.restoreFromSnapshot(emptySnapshot)
            }
        }
    }

    @Nested
    @DisplayName("Edge Cases and Error Handling")
    inner class EdgeCasesAndErrorHandling {

        @Test
        @DisplayName("Should handle extremely long key names")
        fun shouldHandleExtremelyLongKeyNames() {
            // Given
            val longKey = "a".repeat(10000)
            val value = "long_key_value"

            // When
            contextManager.setContext(longKey, value)
            val retrieved = contextManager.getContext(longKey)

            // Then
            assertEquals(value, retrieved)
        }

        @Test
        @DisplayName("Should handle large object storage")
        fun shouldHandleLargeObjectStorage() {
            // Given
            val largeObject = (1..1000).map { "item_$it" }
            val key = "large_object_key"

            // When
            contextManager.setContext(key, largeObject)
            val retrieved = contextManager.getContext(key)

            // Then
            assertEquals(largeObject, retrieved)
        }

        @Test
        @DisplayName("Should handle memory pressure gracefully")
        fun shouldHandleMemoryPressureGracefully() {
            // Given & When
            for (i in 1..10000) {
                contextManager.setContext("memory_key_$i", "memory_value_$i")
            }

            // Then
            assertTrue(contextManager.getContextSize() > 0)
            assertEquals("memory_value_1", contextManager.getContext("memory_key_1"))
            assertEquals("memory_value_10000", contextManager.getContext("memory_key_10000"))
        }

        @Test
        @DisplayName("Should handle recursive object references")
        fun shouldHandleRecursiveObjectReferences() {
            // Given
            val map = mutableMapOf<String, Any>()
            map["self"] = map
            val key = "recursive_key"

            // When & Then
            assertDoesNotThrow {
                contextManager.setContext(key, map)
                val retrieved = contextManager.getContext(key)
                assertNotNull(retrieved)
            }
        }
    }

    @Nested
    @DisplayName("Performance Tests")
    inner class PerformanceTests {

        @Test
        @DisplayName("Should perform context operations efficiently")
        fun shouldPerformContextOperationsEfficiently() {
            // Given
            val iterations = 1000
            val startTime = System.nanoTime()

            // When
            for (i in 1..iterations) {
                contextManager.setContext("perf_key_$i", "perf_value_$i")
                contextManager.getContext("perf_key_$i")
            }

            val endTime = System.nanoTime()
            val duration = endTime - startTime

            // Then
            assertTrue(duration < 1_000_000_000) // Less than 1 second
            assertEquals(iterations, contextManager.getContextSize())
        }

        @Test
        @DisplayName("Should handle bulk operations efficiently")
        fun shouldHandleBulkOperationsEfficiently() {
            // Given
            val bulkData = (1..100).associate { "bulk_key_$it" to "bulk_value_$it" }
            val startTime = System.nanoTime()

            // When
            contextManager.setBulkContext(bulkData)
            val retrieved = contextManager.getBulkContext(bulkData.keys)

            val endTime = System.nanoTime()
            val duration = endTime - startTime

            // Then
            assertTrue(duration < 100_000_000) // Less than 100ms
            assertEquals(bulkData.size, retrieved.size)
            assertEquals(bulkData, retrieved)
        }
    }
}

// Mock interfaces for testing
interface ContextLifecycleListener {
    fun onContextSet(key: String, value: Any?)
    fun onContextGet(key: String, value: Any?)
    fun onContextRemove(key: String)
}

// Extension methods for testing
fun ContextManager.setContextWithValidation(key: String, value: Any?, validator: (Any?) -> Boolean) {
    if (!validator(value)) {
        throw IllegalArgumentException("Context validation failed for key: $key")
    }
    setContext(key, value)
}

fun ContextManager.createSnapshot(): Map<String, Any?> {
    return getContextKeys().associateWith { getContext(it) }
}

fun ContextManager.restoreFromSnapshot(snapshot: Map<String, Any?>) {
    clear()
    snapshot.forEach { (key, value) -> setContext(key, value) }
}

fun ContextManager.setBulkContext(data: Map<String, Any?>) {
    data.forEach { (key, value) -> setContext(key, value) }
}

fun ContextManager.getBulkContext(keys: Set<String>): Map<String, Any?> {
    return keys.associateWith { getContext(it) }
}

fun ContextManager.addLifecycleListener(listener: ContextLifecycleListener) {
    // Implementation would be added to actual ContextManager
}

fun ContextManager.removeLifecycleListener(listener: ContextLifecycleListener) {
    // Implementation would be added to actual ContextManager
}

fun ContextManager.createScope(scopeName: String) {
    // Implementation would be added to actual ContextManager
}

fun ContextManager.setScopedContext(scopeName: String, key: String, value: Any?) {
    // Implementation would be added to actual ContextManager
}

fun ContextManager.getScopedContext(scopeName: String, key: String): Any? {
    // Implementation would be added to actual ContextManager
    return null
}

fun ContextManager.removeScope(scopeName: String) {
    // Implementation would be added to actual ContextManager
}

fun <T> ContextManager.getTypedContext(key: String): T? {
    @Suppress("UNCHECKED_CAST")
    return getContext(key) as? T
}