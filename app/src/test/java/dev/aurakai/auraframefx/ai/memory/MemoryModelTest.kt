package dev.aurakai.auraframefx.ai.memory

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.assertThrows
import org.mockito.MockitoAnnotations
import org.mockito.kotlin.mock
import org.mockito.kotlin.verify
import org.mockito.kotlin.whenever
import org.mockito.kotlin.any
import org.mockito.kotlin.eq
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.test.assertFalse

@DisplayName("MemoryModel Tests")
class MemoryModelTest {
    
    private lateinit var memoryModel: MemoryModel
    
    @BeforeEach
    fun setUp() {
        MockitoAnnotations.openMocks(this)
        memoryModel = MemoryModel()
    }
    
    @AfterEach
    fun tearDown() {
        memoryModel.clear()
    }
    
    @Nested
    @DisplayName("Memory Storage Tests")
    inner class MemoryStorageTests {
        
        @Test
        @DisplayName("Should store and retrieve simple memory item")
        fun shouldStoreAndRetrieveSimpleMemoryItem() {
            // Given
            val key = "test_key"
            val value = "test_value"
            
            // When
            memoryModel.store(key, value)
            val result = memoryModel.retrieve(key)
            
            // Then
            assertEquals(value, result)
        }
        
        @Test
        @DisplayName("Should store and retrieve complex memory item")
        fun shouldStoreAndRetrieveComplexMemoryItem() {
            // Given
            val key = "complex_key"
            val complexValue = mapOf(
                "name" to "John",
                "age" to 30,
                "skills" to listOf("Kotlin", "Java", "Python")
            )
            
            // When
            memoryModel.store(key, complexValue)
            val result = memoryModel.retrieve(key)
            
            // Then
            assertEquals(complexValue, result)
        }
        
        @Test
        @DisplayName("Should return null for non-existent key")
        fun shouldReturnNullForNonExistentKey() {
            // Given
            val nonExistentKey = "non_existent_key"
            
            // When
            val result = memoryModel.retrieve(nonExistentKey)
            
            // Then
            assertNull(result)
        }
        
        @Test
        @DisplayName("Should overwrite existing memory item")
        fun shouldOverwriteExistingMemoryItem() {
            // Given
            val key = "overwrite_key"
            val originalValue = "original_value"
            val newValue = "new_value"
            
            // When
            memoryModel.store(key, originalValue)
            memoryModel.store(key, newValue)
            val result = memoryModel.retrieve(key)
            
            // Then
            assertEquals(newValue, result)
        }
        
        @Test
        @DisplayName("Should handle null values")
        fun shouldHandleNullValues() {
            // Given
            val key = "null_key"
            val nullValue: String? = null
            
            // When
            memoryModel.store(key, nullValue)
            val result = memoryModel.retrieve(key)
            
            // Then
            assertNull(result)
        }
        
        @Test
        @DisplayName("Should handle empty string keys")
        fun shouldHandleEmptyStringKeys() {
            // Given
            val emptyKey = ""
            val value = "test_value"
            
            // When
            memoryModel.store(emptyKey, value)
            val result = memoryModel.retrieve(emptyKey)
            
            // Then
            assertEquals(value, result)
        }
        
        @Test
        @DisplayName("Should throw exception for null key")
        fun shouldThrowExceptionForNullKey() {
            // Given
            val nullKey: String? = null
            val value = "test_value"
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                memoryModel.store(nullKey, value)
            }
        }
    }
    
    @Nested
    @DisplayName("Memory Retrieval Tests")
    inner class MemoryRetrievalTests {
        
        @Test
        @DisplayName("Should retrieve all stored items")
        fun shouldRetrieveAllStoredItems() {
            // Given
            val items = mapOf(
                "key1" to "value1",
                "key2" to "value2",
                "key3" to "value3"
            )
            
            // When
            items.forEach { (key, value) ->
                memoryModel.store(key, value)
            }
            val allItems = memoryModel.getAllItems()
            
            // Then
            assertEquals(items.size, allItems.size)
            items.forEach { (key, value) ->
                assertEquals(value, allItems[key])
            }
        }
        
        @Test
        @DisplayName("Should return empty map when no items stored")
        fun shouldReturnEmptyMapWhenNoItemsStored() {
            // When
            val allItems = memoryModel.getAllItems()
            
            // Then
            assertTrue(allItems.isEmpty())
        }
        
        @Test
        @DisplayName("Should check if key exists")
        fun shouldCheckIfKeyExists() {
            // Given
            val existingKey = "existing_key"
            val nonExistentKey = "non_existent_key"
            
            // When
            memoryModel.store(existingKey, "value")
            
            // Then
            assertTrue(memoryModel.contains(existingKey))
            assertFalse(memoryModel.contains(nonExistentKey))
        }
        
        @Test
        @DisplayName("Should get all keys")
        fun shouldGetAllKeys() {
            // Given
            val keys = listOf("key1", "key2", "key3")
            
            // When
            keys.forEach { key ->
                memoryModel.store(key, "value_$key")
            }
            val retrievedKeys = memoryModel.getKeys()
            
            // Then
            assertEquals(keys.size, retrievedKeys.size)
            keys.forEach { key ->
                assertTrue(retrievedKeys.contains(key))
            }
        }
        
        @Test
        @DisplayName("Should get values by pattern")
        fun shouldGetValuesByPattern() {
            // Given
            val patternKey = "pattern_"
            val matchingKeys = listOf("pattern_1", "pattern_2", "pattern_3")
            val nonMatchingKeys = listOf("other_1", "other_2")
            
            // When
            matchingKeys.forEach { key ->
                memoryModel.store(key, "value_$key")
            }
            nonMatchingKeys.forEach { key ->
                memoryModel.store(key, "value_$key")
            }
            val matchingValues = memoryModel.getValuesByPattern(patternKey)
            
            // Then
            assertEquals(matchingKeys.size, matchingValues.size)
            matchingKeys.forEach { key ->
                assertTrue(matchingValues.containsKey(key))
            }
            nonMatchingKeys.forEach { key ->
                assertFalse(matchingValues.containsKey(key))
            }
        }
    }
    
    @Nested
    @DisplayName("Memory Management Tests")
    inner class MemoryManagementTests {
        
        @Test
        @DisplayName("Should remove single item")
        fun shouldRemoveSingleItem() {
            // Given
            val key = "remove_key"
            val value = "remove_value"
            
            // When
            memoryModel.store(key, value)
            assertTrue(memoryModel.contains(key))
            val removed = memoryModel.remove(key)
            
            // Then
            assertEquals(value, removed)
            assertFalse(memoryModel.contains(key))
        }
        
        @Test
        @DisplayName("Should return null when removing non-existent item")
        fun shouldReturnNullWhenRemovingNonExistentItem() {
            // Given
            val nonExistentKey = "non_existent_key"
            
            // When
            val removed = memoryModel.remove(nonExistentKey)
            
            // Then
            assertNull(removed)
        }
        
        @Test
        @DisplayName("Should clear all items")
        fun shouldClearAllItems() {
            // Given
            val items = mapOf(
                "key1" to "value1",
                "key2" to "value2",
                "key3" to "value3"
            )
            
            // When
            items.forEach { (key, value) ->
                memoryModel.store(key, value)
            }
            memoryModel.clear()
            
            // Then
            assertTrue(memoryModel.isEmpty())
            assertEquals(0, memoryModel.size())
        }
        
        @Test
        @DisplayName("Should get correct size")
        fun shouldGetCorrectSize() {
            // Given
            val items = mapOf(
                "key1" to "value1",
                "key2" to "value2",
                "key3" to "value3"
            )
            
            // When
            assertEquals(0, memoryModel.size())
            items.forEach { (key, value) ->
                memoryModel.store(key, value)
            }
            
            // Then
            assertEquals(items.size, memoryModel.size())
        }
        
        @Test
        @DisplayName("Should check if empty")
        fun shouldCheckIfEmpty() {
            // Given
            assertTrue(memoryModel.isEmpty())
            
            // When
            memoryModel.store("key", "value")
            
            // Then
            assertFalse(memoryModel.isEmpty())
            
            // When
            memoryModel.clear()
            
            // Then
            assertTrue(memoryModel.isEmpty())
        }
        
        @Test
        @DisplayName("Should remove items by pattern")
        fun shouldRemoveItemsByPattern() {
            // Given
            val patternKey = "temp_"
            val matchingKeys = listOf("temp_1", "temp_2", "temp_3")
            val nonMatchingKeys = listOf("perm_1", "perm_2")
            
            // When
            matchingKeys.forEach { key ->
                memoryModel.store(key, "value_$key")
            }
            nonMatchingKeys.forEach { key ->
                memoryModel.store(key, "value_$key")
            }
            
            val removedCount = memoryModel.removeByPattern(patternKey)
            
            // Then
            assertEquals(matchingKeys.size, removedCount)
            matchingKeys.forEach { key ->
                assertFalse(memoryModel.contains(key))
            }
            nonMatchingKeys.forEach { key ->
                assertTrue(memoryModel.contains(key))
            }
        }
    }
    
    @Nested
    @DisplayName("Memory Persistence Tests")
    inner class MemoryPersistenceTests {
        
        @Test
        @DisplayName("Should save and load memory state")
        fun shouldSaveAndLoadMemoryState() {
            // Given
            val items = mapOf(
                "key1" to "value1",
                "key2" to "value2",
                "key3" to "value3"
            )
            val filePath = "test_memory.json"
            
            // When
            items.forEach { (key, value) ->
                memoryModel.store(key, value)
            }
            memoryModel.saveToFile(filePath)
            memoryModel.clear()
            memoryModel.loadFromFile(filePath)
            
            // Then
            assertEquals(items.size, memoryModel.size())
            items.forEach { (key, value) ->
                assertEquals(value, memoryModel.retrieve(key))
            }
        }
        
        @Test
        @DisplayName("Should handle save to invalid file path")
        fun shouldHandleSaveToInvalidFilePath() {
            // Given
            val invalidPath = "/invalid/path/memory.json"
            
            // When & Then
            assertThrows<Exception> {
                memoryModel.saveToFile(invalidPath)
            }
        }
        
        @Test
        @DisplayName("Should handle load from non-existent file")
        fun shouldHandleLoadFromNonExistentFile() {
            // Given
            val nonExistentFile = "non_existent_memory.json"
            
            // When & Then
            assertThrows<Exception> {
                memoryModel.loadFromFile(nonExistentFile)
            }
        }
    }
    
    @Nested
    @DisplayName("Memory Search Tests")
    inner class MemorySearchTests {
        
        @Test
        @DisplayName("Should find items by value")
        fun shouldFindItemsByValue() {
            // Given
            val searchValue = "searchable_value"
            val items = mapOf(
                "key1" to searchValue,
                "key2" to "other_value",
                "key3" to searchValue,
                "key4" to "another_value"
            )
            
            // When
            items.forEach { (key, value) ->
                memoryModel.store(key, value)
            }
            val foundKeys = memoryModel.findKeysByValue(searchValue)
            
            // Then
            assertEquals(2, foundKeys.size)
            assertTrue(foundKeys.contains("key1"))
            assertTrue(foundKeys.contains("key3"))
        }
        
        @Test
        @DisplayName("Should search values containing substring")
        fun shouldSearchValuesContainingSubstring() {
            // Given
            val substring = "test"
            val items = mapOf(
                "key1" to "this is a test value",
                "key2" to "testing123",
                "key3" to "no match here",
                "key4" to "another test case"
            )
            
            // When
            items.forEach { (key, value) ->
                memoryModel.store(key, value)
            }
            val foundItems = memoryModel.searchBySubstring(substring)
            
            // Then
            assertEquals(3, foundItems.size)
            assertTrue(foundItems.containsKey("key1"))
            assertTrue(foundItems.containsKey("key2"))
            assertTrue(foundItems.containsKey("key4"))
        }
        
        @Test
        @DisplayName("Should perform case-insensitive search")
        fun shouldPerformCaseInsensitiveSearch() {
            // Given
            val searchTerm = "TEST"
            val items = mapOf(
                "key1" to "test value",
                "key2" to "Test Value",
                "key3" to "TEST VALUE",
                "key4" to "no match"
            )
            
            // When
            items.forEach { (key, value) ->
                memoryModel.store(key, value)
            }
            val foundItems = memoryModel.searchCaseInsensitive(searchTerm)
            
            // Then
            assertEquals(3, foundItems.size)
            assertTrue(foundItems.containsKey("key1"))
            assertTrue(foundItems.containsKey("key2"))
            assertTrue(foundItems.containsKey("key3"))
        }
    }
    
    @Nested
    @DisplayName("Memory Metadata Tests")
    inner class MemoryMetadataTests {
        
        @Test
        @DisplayName("Should track creation timestamp")
        fun shouldTrackCreationTimestamp() {
            // Given
            val key = "timestamp_key"
            val value = "timestamp_value"
            val beforeTime = System.currentTimeMillis()
            
            // When
            memoryModel.store(key, value)
            val afterTime = System.currentTimeMillis()
            val timestamp = memoryModel.getCreationTimestamp(key)
            
            // Then
            assertNotNull(timestamp)
            assertTrue(timestamp!! >= beforeTime)
            assertTrue(timestamp <= afterTime)
        }
        
        @Test
        @DisplayName("Should track access count")
        fun shouldTrackAccessCount() {
            // Given
            val key = "access_key"
            val value = "access_value"
            
            // When
            memoryModel.store(key, value)
            assertEquals(0, memoryModel.getAccessCount(key))
            
            memoryModel.retrieve(key)
            assertEquals(1, memoryModel.getAccessCount(key))
            
            memoryModel.retrieve(key)
            memoryModel.retrieve(key)
            assertEquals(3, memoryModel.getAccessCount(key))
        }
        
        @Test
        @DisplayName("Should track last access time")
        fun shouldTrackLastAccessTime() {
            // Given
            val key = "last_access_key"
            val value = "last_access_value"
            
            // When
            memoryModel.store(key, value)
            val beforeAccess = System.currentTimeMillis()
            Thread.sleep(10) // Small delay to ensure different timestamps
            memoryModel.retrieve(key)
            val afterAccess = System.currentTimeMillis()
            val lastAccessTime = memoryModel.getLastAccessTime(key)
            
            // Then
            assertNotNull(lastAccessTime)
            assertTrue(lastAccessTime!! >= beforeAccess)
            assertTrue(lastAccessTime <= afterAccess)
        }
    }
    
    @Nested
    @DisplayName("Memory Capacity Tests")
    inner class MemoryCapacityTests {
        
        @Test
        @DisplayName("Should handle maximum capacity")
        fun shouldHandleMaximumCapacity() {
            // Given
            val maxCapacity = 100
            val memoryModelWithLimit = MemoryModel(maxCapacity)
            
            // When
            for (i in 1..maxCapacity + 10) {
                memoryModelWithLimit.store("key$i", "value$i")
            }
            
            // Then
            assertEquals(maxCapacity, memoryModelWithLimit.size())
        }
        
        @Test
        @DisplayName("Should evict oldest items when capacity exceeded")
        fun shouldEvictOldestItemsWhenCapacityExceeded() {
            // Given
            val maxCapacity = 3
            val memoryModelWithLimit = MemoryModel(maxCapacity)
            
            // When
            memoryModelWithLimit.store("key1", "value1")
            memoryModelWithLimit.store("key2", "value2")
            memoryModelWithLimit.store("key3", "value3")
            memoryModelWithLimit.store("key4", "value4") // Should evict key1
            
            // Then
            assertEquals(maxCapacity, memoryModelWithLimit.size())
            assertFalse(memoryModelWithLimit.contains("key1"))
            assertTrue(memoryModelWithLimit.contains("key2"))
            assertTrue(memoryModelWithLimit.contains("key3"))
            assertTrue(memoryModelWithLimit.contains("key4"))
        }
    }
    
    @Nested
    @DisplayName("Memory Thread Safety Tests")
    inner class MemoryThreadSafetyTests {
        
        @Test
        @DisplayName("Should handle concurrent access safely")
        fun shouldHandleConcurrentAccessSafely() {
            // Given
            val threadCount = 10
            val operationsPerThread = 100
            val threads = mutableListOf<Thread>()
            
            // When
            repeat(threadCount) { threadIndex ->
                val thread = Thread {
                    repeat(operationsPerThread) { opIndex ->
                        val key = "thread${threadIndex}_op${opIndex}"
                        val value = "value${threadIndex}_${opIndex}"
                        memoryModel.store(key, value)
                        memoryModel.retrieve(key)
                    }
                }
                threads.add(thread)
                thread.start()
            }
            
            threads.forEach { it.join() }
            
            // Then
            assertEquals(threadCount * operationsPerThread, memoryModel.size())
        }
        
        @Test
        @DisplayName("Should handle concurrent modifications safely")
        fun shouldHandleConcurrentModificationsSafely() {
            // Given
            val key = "concurrent_key"
            val threadCount = 5
            val threads = mutableListOf<Thread>()
            
            // When
            repeat(threadCount) { threadIndex ->
                val thread = Thread {
                    repeat(10) { opIndex ->
                        memoryModel.store(key, "value_${threadIndex}_${opIndex}")
                        Thread.sleep(1)
                        memoryModel.retrieve(key)
                    }
                }
                threads.add(thread)
                thread.start()
            }
            
            threads.forEach { it.join() }
            
            // Then
            assertNotNull(memoryModel.retrieve(key))
            assertEquals(1, memoryModel.size())
        }
    }
    
    @Nested
    @DisplayName("Memory Error Handling Tests")
    inner class MemoryErrorHandlingTests {
        
        @Test
        @DisplayName("Should handle memory overflow gracefully")
        fun shouldHandleMemoryOverflowGracefully() {
            // Given
            val largeValue = "x".repeat(1000000) // 1MB string
            
            // When & Then
            assertDoesNotThrow {
                memoryModel.store("large_key", largeValue)
                val retrieved = memoryModel.retrieve("large_key")
                assertEquals(largeValue, retrieved)
            }
        }
        
        @Test
        @DisplayName("Should handle special characters in keys")
        fun shouldHandleSpecialCharactersInKeys() {
            // Given
            val specialKeys = listOf(
                "key with spaces",
                "key/with/slashes",
                "key.with.dots",
                "key-with-dashes",
                "key_with_underscores",
                "key@with@symbols",
                "key#with#hash"
            )
            
            // When & Then
            specialKeys.forEach { key ->
                assertDoesNotThrow {
                    memoryModel.store(key, "value_for_$key")
                    assertEquals("value_for_$key", memoryModel.retrieve(key))
                }
            }
        }
        
        @Test
        @DisplayName("Should handle Unicode characters")
        fun shouldHandleUnicodeCharacters() {
            // Given
            val unicodeKey = "🔑_unicode_key"
            val unicodeValue = "🌟_unicode_value_🎉"
            
            // When
            memoryModel.store(unicodeKey, unicodeValue)
            val retrieved = memoryModel.retrieve(unicodeKey)
            
            // Then
            assertEquals(unicodeValue, retrieved)
        }
    }
}