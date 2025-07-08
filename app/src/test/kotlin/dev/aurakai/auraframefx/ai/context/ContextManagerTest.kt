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
import org.mockito.kotlin.*
import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.CompletableFuture
import kotlin.test.assertContains
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.assertFalse

/**
 * Comprehensive unit tests for ContextManager class.
 * Testing framework: JUnit 5 with Mockito for mocking.
 * 
 * Tests cover:
 * - Context creation and management
 * - Context retrieval and searching
 * - Context persistence and loading
 * - Error handling and edge cases
 * - Concurrency scenarios
 * - Resource cleanup
 */
@DisplayName("ContextManager Tests")
class ContextManagerTest {

    private lateinit var contextManager: ContextManager
    private lateinit var tempDir: Path
    
    @Mock
    private lateinit var mockContextStore: ContextStore
    
    @Mock
    private lateinit var mockContextProcessor: ContextProcessor
    
    @BeforeEach
    fun setUp() {
        MockitoAnnotations.openMocks(this)
        tempDir = Files.createTempDirectory("context_manager_test")
        contextManager = ContextManager(tempDir.toString())
    }
    
    @AfterEach
    fun tearDown() {
        tempDir.toFile().deleteRecursively()
    }

    @Nested
    @DisplayName("Context Creation Tests")
    inner class ContextCreationTests {
        
        @Test
        @DisplayName("Should create context with valid parameters")
        fun shouldCreateContextWithValidParameters() {
            // Given
            val contextId = "test-context-1"
            val content = "Test content for context"
            val metadata = mapOf("type" to "test", "priority" to "high")
            
            // When
            val result = contextManager.createContext(contextId, content, metadata)
            
            // Then
            assertNotNull(result)
            assertEquals(contextId, result.id)
            assertEquals(content, result.content)
            assertEquals(metadata, result.metadata)
            assertTrue(result.timestamp > 0)
        }
        
        @Test
        @DisplayName("Should generate unique ID when none provided")
        fun shouldGenerateUniqueIdWhenNoneProvided() {
            // Given
            val content = "Test content"
            
            // When
            val result1 = contextManager.createContext(content = content)
            val result2 = contextManager.createContext(content = content)
            
            // Then
            assertNotNull(result1.id)
            assertNotNull(result2.id)
            assertNotEquals(result1.id, result2.id)
        }
        
        @Test
        @DisplayName("Should throw exception for empty content")
        fun shouldThrowExceptionForEmptyContent() {
            // Given
            val emptyContent = ""
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                contextManager.createContext(content = emptyContent)
            }
        }
        
        @Test
        @DisplayName("Should throw exception for null content")
        fun shouldThrowExceptionForNullContent() {
            // When & Then
            assertThrows<IllegalArgumentException> {
                contextManager.createContext(content = null)
            }
        }
        
        @Test
        @DisplayName("Should handle large content gracefully")
        fun shouldHandleLargeContentGracefully() {
            // Given
            val largeContent = "A".repeat(1_000_000) // 1MB of content
            
            // When
            val result = contextManager.createContext(content = largeContent)
            
            // Then
            assertNotNull(result)
            assertEquals(largeContent, result.content)
        }
        
        @Test
        @DisplayName("Should handle special characters in content")
        fun shouldHandleSpecialCharactersInContent() {
            // Given
            val specialContent = "Content with special chars: 🚀 \n\t\r \"quotes\" 'apostrophes' & < > symbols"
            
            // When
            val result = contextManager.createContext(content = specialContent)
            
            // Then
            assertNotNull(result)
            assertEquals(specialContent, result.content)
        }
    }

    @Nested
    @DisplayName("Context Retrieval Tests")
    inner class ContextRetrievalTests {
        
        @Test
        @DisplayName("Should retrieve existing context by ID")
        fun shouldRetrieveExistingContextById() {
            // Given
            val contextId = "test-context"
            val content = "Test content"
            val created = contextManager.createContext(contextId, content)
            
            // When
            val retrieved = contextManager.getContext(contextId)
            
            // Then
            assertNotNull(retrieved)
            assertEquals(created.id, retrieved.id)
            assertEquals(created.content, retrieved.content)
        }
        
        @Test
        @DisplayName("Should return null for non-existent context")
        fun shouldReturnNullForNonExistentContext() {
            // Given
            val nonExistentId = "non-existent-context"
            
            // When
            val result = contextManager.getContext(nonExistentId)
            
            // Then
            assertNull(result)
        }
        
        @Test
        @DisplayName("Should find contexts by content search")
        fun shouldFindContextsByContentSearch() {
            // Given
            contextManager.createContext("ctx1", "Java programming tutorial")
            contextManager.createContext("ctx2", "Kotlin programming guide")
            contextManager.createContext("ctx3", "Python basics")
            
            // When
            val results = contextManager.searchContexts("programming")
            
            // Then
            assertEquals(2, results.size)
            assertTrue(results.any { it.content.contains("Java programming") })
            assertTrue(results.any { it.content.contains("Kotlin programming") })
        }
        
        @Test
        @DisplayName("Should find contexts by metadata search")
        fun shouldFindContextsByMetadataSearch() {
            // Given
            val metadata1 = mapOf("category" to "tutorial", "language" to "java")
            val metadata2 = mapOf("category" to "guide", "language" to "kotlin")
            val metadata3 = mapOf("category" to "tutorial", "language" to "python")
            
            contextManager.createContext("ctx1", "Content 1", metadata1)
            contextManager.createContext("ctx2", "Content 2", metadata2)
            contextManager.createContext("ctx3", "Content 3", metadata3)
            
            // When
            val results = contextManager.searchContextsByMetadata("category", "tutorial")
            
            // Then
            assertEquals(2, results.size)
            assertTrue(results.all { it.metadata["category"] == "tutorial" })
        }
        
        @Test
        @DisplayName("Should return empty list for search with no matches")
        fun shouldReturnEmptyListForSearchWithNoMatches() {
            // Given
            contextManager.createContext("ctx1", "Java content")
            contextManager.createContext("ctx2", "Kotlin content")
            
            // When
            val results = contextManager.searchContexts("nonexistent")
            
            // Then
            assertTrue(results.isEmpty())
        }
        
        @Test
        @DisplayName("Should handle case-insensitive search")
        fun shouldHandleCaseInsensitiveSearch() {
            // Given
            contextManager.createContext("ctx1", "Java Programming")
            
            // When
            val results = contextManager.searchContexts("java programming")
            
            // Then
            assertEquals(1, results.size)
            assertContains(results[0].content, "Java Programming")
        }
    }

    @Nested
    @DisplayName("Context Persistence Tests")
    inner class ContextPersistenceTests {
        
        @Test
        @DisplayName("Should save context to persistent storage")
        fun shouldSaveContextToPersistentStorage() {
            // Given
            val context = contextManager.createContext("persistent-ctx", "Content to persist")
            
            // When
            contextManager.saveContext(context)
            
            // Then
            val savedFile = File(tempDir.toFile(), "${context.id}.json")
            assertTrue(savedFile.exists())
            assertTrue(savedFile.readText().contains(context.content))
        }
        
        @Test
        @DisplayName("Should load context from persistent storage")
        fun shouldLoadContextFromPersistentStorage() {
            // Given
            val originalContext = contextManager.createContext("load-ctx", "Content to load")
            contextManager.saveContext(originalContext)
            
            // When
            val loadedContext = contextManager.loadContext(originalContext.id)
            
            // Then
            assertNotNull(loadedContext)
            assertEquals(originalContext.id, loadedContext.id)
            assertEquals(originalContext.content, loadedContext.content)
        }
        
        @Test
        @DisplayName("Should handle corrupted storage gracefully")
        fun shouldHandleCorruptedStorageGracefully() {
            // Given
            val corruptedFile = File(tempDir.toFile(), "corrupted.json")
            corruptedFile.writeText("invalid json content")
            
            // When & Then
            assertThrows<RuntimeException> {
                contextManager.loadContext("corrupted")
            }
        }
        
        @Test
        @DisplayName("Should backup contexts before overwriting")
        fun shouldBackupContextsBeforeOverwriting() {
            // Given
            val context = contextManager.createContext("backup-ctx", "Original content")
            contextManager.saveContext(context)
            
            val updatedContext = context.copy(content = "Updated content")
            
            // When
            contextManager.saveContext(updatedContext)
            
            // Then
            val backupFile = File(tempDir.toFile(), "${context.id}.backup")
            assertTrue(backupFile.exists())
            assertTrue(backupFile.readText().contains("Original content"))
        }
    }

    @Nested
    @DisplayName("Context Management Tests")
    inner class ContextManagementTests {
        
        @Test
        @DisplayName("Should delete context successfully")
        fun shouldDeleteContextSuccessfully() {
            // Given
            val context = contextManager.createContext("delete-ctx", "Content to delete")
            contextManager.saveContext(context)
            
            // When
            val deleted = contextManager.deleteContext(context.id)
            
            // Then
            assertTrue(deleted)
            assertNull(contextManager.getContext(context.id))
            assertFalse(File(tempDir.toFile(), "${context.id}.json").exists())
        }
        
        @Test
        @DisplayName("Should return false when deleting non-existent context")
        fun shouldReturnFalseWhenDeletingNonExistentContext() {
            // Given
            val nonExistentId = "non-existent"
            
            // When
            val deleted = contextManager.deleteContext(nonExistentId)
            
            // Then
            assertFalse(deleted)
        }
        
        @Test
        @DisplayName("Should update existing context")
        fun shouldUpdateExistingContext() {
            // Given
            val originalContext = contextManager.createContext("update-ctx", "Original content")
            val updatedContent = "Updated content"
            val updatedMetadata = mapOf("updated" to "true")
            
            // When
            val updated = contextManager.updateContext(
                originalContext.id, 
                updatedContent, 
                updatedMetadata
            )
            
            // Then
            assertNotNull(updated)
            assertEquals(updatedContent, updated.content)
            assertEquals(updatedMetadata, updated.metadata)
        }
        
        @Test
        @DisplayName("Should maintain context ordering by timestamp")
        fun shouldMaintainContextOrderingByTimestamp() {
            // Given
            val context1 = contextManager.createContext("ctx1", "Content 1")
            Thread.sleep(10) // Ensure different timestamps
            val context2 = contextManager.createContext("ctx2", "Content 2")
            Thread.sleep(10)
            val context3 = contextManager.createContext("ctx3", "Content 3")
            
            // When
            val allContexts = contextManager.getAllContexts()
            
            // Then
            assertEquals(3, allContexts.size)
            assertTrue(allContexts[0].timestamp <= allContexts[1].timestamp)
            assertTrue(allContexts[1].timestamp <= allContexts[2].timestamp)
        }
        
        @Test
        @DisplayName("Should limit context count when specified")
        fun shouldLimitContextCountWhenSpecified() {
            // Given
            val maxContexts = 5
            contextManager.setMaxContexts(maxContexts)
            
            // When
            for (i in 1..10) {
                contextManager.createContext("ctx$i", "Content $i")
            }
            
            // Then
            val allContexts = contextManager.getAllContexts()
            assertEquals(maxContexts, allContexts.size)
        }
    }

    @Nested
    @DisplayName("Error Handling Tests")
    inner class ErrorHandlingTests {
        
        @Test
        @DisplayName("Should handle file system errors gracefully")
        fun shouldHandleFileSystemErrorsGracefully() {
            // Given
            val readOnlyDir = Files.createTempDirectory("readonly")
            readOnlyDir.toFile().setWritable(false)
            val readOnlyContextManager = ContextManager(readOnlyDir.toString())
            val context = Context("test", "content", emptyMap(), System.currentTimeMillis())
            
            // When & Then
            assertThrows<RuntimeException> {
                readOnlyContextManager.saveContext(context)
            }
            
            // Cleanup
            readOnlyDir.toFile().setWritable(true)
            readOnlyDir.toFile().deleteRecursively()
        }
        
        @Test
        @DisplayName("Should handle invalid context ID characters")
        fun shouldHandleInvalidContextIdCharacters() {
            // Given
            val invalidIds = listOf("ctx/invalid", "ctx\\invalid", "ctx:invalid", "ctx*invalid")
            
            // When & Then
            invalidIds.forEach { invalidId ->
                assertThrows<IllegalArgumentException> {
                    contextManager.createContext(invalidId, "content")
                }
            }
        }
        
        @Test
        @DisplayName("Should handle concurrent access gracefully")
        fun shouldHandleConcurrentAccessGracefully() {
            // Given
            val futures = mutableListOf<CompletableFuture<Context>>()
            
            // When
            for (i in 1..50) {
                val future = CompletableFuture.supplyAsync {
                    contextManager.createContext("concurrent-ctx-$i", "Content $i")
                }
                futures.add(future)
            }
            
            // Then
            val results = futures.map { it.get() }
            assertEquals(50, results.size)
            assertEquals(50, results.map { it.id }.distinct().size) // All IDs should be unique
        }
        
        @Test
        @DisplayName("Should handle memory pressure gracefully")
        fun shouldHandleMemoryPressureGracefully() {
            // Given
            val largeContent = "A".repeat(10_000)
            
            // When
            val contexts = (1..100).map { i ->
                contextManager.createContext("memory-ctx-$i", largeContent)
            }
            
            // Then
            assertEquals(100, contexts.size)
            contexts.forEach { context ->
                assertEquals(largeContent, context.content)
            }
        }
    }

    @Nested
    @DisplayName("Performance Tests")
    inner class PerformanceTests {
        
        @Test
        @DisplayName("Should handle bulk operations efficiently")
        fun shouldHandleBulkOperationsEfficiently() {
            // Given
            val contextCount = 1000
            val startTime = System.currentTimeMillis()
            
            // When
            for (i in 1..contextCount) {
                contextManager.createContext("bulk-ctx-$i", "Content $i")
            }
            
            val endTime = System.currentTimeMillis()
            val duration = endTime - startTime
            
            // Then
            assertTrue(duration < 5000) // Should complete within 5 seconds
            assertEquals(contextCount, contextManager.getAllContexts().size)
        }
        
        @Test
        @DisplayName("Should perform search operations efficiently")
        fun shouldPerformSearchOperationsEfficiently() {
            // Given
            for (i in 1..100) {
                contextManager.createContext("search-ctx-$i", "Content $i with keyword")
            }
            
            val startTime = System.currentTimeMillis()
            
            // When
            val results = contextManager.searchContexts("keyword")
            
            val endTime = System.currentTimeMillis()
            val duration = endTime - startTime
            
            // Then
            assertTrue(duration < 1000) // Should complete within 1 second
            assertEquals(100, results.size)
        }
    }

    @Nested
    @DisplayName("Integration Tests")
    inner class IntegrationTests {
        
        @Test
        @DisplayName("Should maintain data integrity across operations")
        fun shouldMaintainDataIntegrityAcrossOperations() {
            // Given
            val contexts = (1..10).map { i ->
                contextManager.createContext("integrity-ctx-$i", "Content $i")
            }
            
            // When
            contexts.forEach { contextManager.saveContext(it) }
            
            // Update some contexts
            val updatedContexts = contexts.take(5).map { context ->
                contextManager.updateContext(context.id, "Updated ${context.content}")
            }
            
            // Delete some contexts
            contexts.drop(7).forEach { context ->
                contextManager.deleteContext(context.id)
            }
            
            // Then
            val remainingContexts = contextManager.getAllContexts()
            assertEquals(7, remainingContexts.size)
            
            // Verify updated contexts
            updatedContexts.forEach { updated ->
                val retrieved = contextManager.getContext(updated!!.id)
                assertNotNull(retrieved)
                assertTrue(retrieved.content.startsWith("Updated"))
            }
        }
        
        @Test
        @DisplayName("Should handle mixed content types correctly")
        fun shouldHandleMixedContentTypesCorrectly() {
            // Given
            val jsonContent = """{"key": "value", "number": 42}"""
            val xmlContent = """<root><item>value</item></root>"""
            val plainContent = "Simple plain text content"
            
            // When
            val jsonContext = contextManager.createContext("json-ctx", jsonContent)
            val xmlContext = contextManager.createContext("xml-ctx", xmlContent)
            val plainContext = contextManager.createContext("plain-ctx", plainContent)
            
            // Then
            assertEquals(jsonContent, jsonContext.content)
            assertEquals(xmlContent, xmlContext.content)
            assertEquals(plainContent, plainContext.content)
            
            // Verify search works across content types
            val searchResults = contextManager.searchContexts("value")
            assertEquals(3, searchResults.size)
        }
    }

    @Nested
    @DisplayName("Edge Case Tests")
    inner class EdgeCaseTests {
        
        @Test
        @DisplayName("Should handle empty metadata gracefully")
        fun shouldHandleEmptyMetadataGracefully() {
            // Given
            val emptyMetadata = emptyMap<String, String>()
            
            // When
            val context = contextManager.createContext("empty-meta-ctx", "Content", emptyMetadata)
            
            // Then
            assertNotNull(context)
            assertTrue(context.metadata.isEmpty())
        }
        
        @Test
        @DisplayName("Should handle very long context IDs")
        fun shouldHandleVeryLongContextIds() {
            // Given
            val longId = "a".repeat(255)
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                contextManager.createContext(longId, "content")
            }
        }
        
        @Test
        @DisplayName("Should handle null metadata values")
        fun shouldHandleNullMetadataValues() {
            // Given
            val metadataWithNulls = mapOf("key1" to "value1", "key2" to null)
            
            // When
            val context = contextManager.createContext("null-meta-ctx", "Content", metadataWithNulls)
            
            // Then
            assertNotNull(context)
            assertEquals("value1", context.metadata["key1"])
            assertNull(context.metadata["key2"])
        }
        
        @Test
        @DisplayName("Should handle context with only whitespace content")
        fun shouldHandleContextWithOnlyWhitespaceContent() {
            // Given
            val whitespaceContent = "   \t\n\r   "
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                contextManager.createContext("whitespace-ctx", whitespaceContent)
            }
        }
    }
}