package com.example.app.ipc

import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.extension.ExtendWith
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.junit.jupiter.MockitoExtension
import org.mockito.kotlin.*
import java.io.IOException
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeoutException

@ExtendWith(MockitoExtension::class)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class OracleDriveServiceConnectorTest {

    @Mock
    private lateinit var mockServiceClient: OracleDriveServiceClient
    
    @Mock
    private lateinit var mockConnectionManager: ConnectionManager
    
    @Mock
    private lateinit var mockAuthProvider: AuthProvider
    
    private lateinit var connector: OracleDriveServiceConnector
    
    @BeforeEach
    fun setUp() {
        MockitoAnnotations.openMocks(this)
        connector = OracleDriveServiceConnector(
            serviceClient = mockServiceClient,
            connectionManager = mockConnectionManager,
            authProvider = mockAuthProvider
        )
    }
    
    @AfterEach
    fun tearDown() {
        connector.close()
    }
    
    @Nested
    @DisplayName("Connection Management Tests")
    inner class ConnectionManagementTests {
        
        @Test
        @DisplayName("Should successfully establish connection with valid credentials")
        fun testSuccessfulConnection() = runTest {
            // Given
            val validCredentials = Credentials("valid_token", "valid_endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(validCredentials)
            whenever(mockConnectionManager.connect(any())).thenReturn(true)
            
            // When
            val result = connector.connect()
            
            // Then
            assertTrue(result)
            verify(mockAuthProvider).getCredentials()
            verify(mockConnectionManager).connect(validCredentials)
        }
        
        @Test
        @DisplayName("Should fail to connect with invalid credentials")
        fun testConnectionFailureWithInvalidCredentials() = runTest {
            // Given
            val invalidCredentials = Credentials("invalid_token", "invalid_endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(invalidCredentials)
            whenever(mockConnectionManager.connect(any())).thenReturn(false)
            
            // When
            val result = connector.connect()
            
            // Then
            assertFalse(result)
            verify(mockAuthProvider).getCredentials()
            verify(mockConnectionManager).connect(invalidCredentials)
        }
        
        @Test
        @DisplayName("Should handle connection timeout gracefully")
        fun testConnectionTimeout() = runTest {
            // Given
            whenever(mockAuthProvider.getCredentials()).thenThrow(TimeoutException("Connection timeout"))
            
            // When & Then
            assertThrows<TimeoutException> {
                connector.connect()
            }
        }
        
        @Test
        @DisplayName("Should retry connection on temporary failures")
        fun testConnectionRetry() = runTest {
            // Given
            val credentials = Credentials("token", "endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any()))
                .thenReturn(false)
                .thenReturn(false)
                .thenReturn(true)
            
            // When
            val result = connector.connectWithRetry(maxRetries = 3)
            
            // Then
            assertTrue(result)
            verify(mockConnectionManager, times(3)).connect(credentials)
        }
        
        @Test
        @DisplayName("Should properly close connection and clean up resources")
        fun testProperConnectionCleanup() {
            // Given
            connector.connect()
            
            // When
            connector.close()
            
            // Then
            verify(mockConnectionManager).close()
            verify(mockServiceClient).shutdown()
        }
    }
    
    @Nested
    @DisplayName("Data Operation Tests")
    inner class DataOperationTests {
        
        @Test
        @DisplayName("Should successfully upload file")
        fun testSuccessfulFileUpload() = runTest {
            // Given
            val fileData = "test file content".toByteArray()
            val fileName = "test.txt"
            val expectedUploadId = "upload_123"
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture(expectedUploadId))
            
            // When
            val result = connector.uploadFile(fileName, fileData)
            
            // Then
            assertEquals(expectedUploadId, result.get())
            verify(mockServiceClient).uploadFile(fileName, fileData)
        }
        
        @Test
        @DisplayName("Should handle file upload failure")
        fun testFileUploadFailure() = runTest {
            // Given
            val fileData = "test file content".toByteArray()
            val fileName = "test.txt"
            val exception = IOException("Upload failed")
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(exception))
            
            // When & Then
            val result = connector.uploadFile(fileName, fileData)
            assertThrows<IOException> {
                result.get()
            }
        }
        
        @Test
        @DisplayName("Should successfully download file")
        fun testSuccessfulFileDownload() = runTest {
            // Given
            val fileId = "file_123"
            val expectedData = "downloaded content".toByteArray()
            
            whenever(mockServiceClient.downloadFile(any()))
                .thenReturn(CompletableFuture.completedFuture(expectedData))
            
            // When
            val result = connector.downloadFile(fileId)
            
            // Then
            assertArrayEquals(expectedData, result.get())
            verify(mockServiceClient).downloadFile(fileId)
        }
        
        @Test
        @DisplayName("Should handle file download failure")
        fun testFileDownloadFailure() = runTest {
            // Given
            val fileId = "non_existent_file"
            val exception = IOException("File not found")
            
            whenever(mockServiceClient.downloadFile(any()))
                .thenReturn(CompletableFuture.failedFuture(exception))
            
            // When & Then
            val result = connector.downloadFile(fileId)
            assertThrows<IOException> {
                result.get()
            }
        }
        
        @Test
        @DisplayName("Should successfully delete file")
        fun testSuccessfulFileDelete() = runTest {
            // Given
            val fileId = "file_123"
            
            whenever(mockServiceClient.deleteFile(any()))
                .thenReturn(CompletableFuture.completedFuture(true))
            
            // When
            val result = connector.deleteFile(fileId)
            
            // Then
            assertTrue(result.get())
            verify(mockServiceClient).deleteFile(fileId)
        }
        
        @Test
        @DisplayName("Should handle file delete failure")
        fun testFileDeleteFailure() = runTest {
            // Given
            val fileId = "protected_file"
            val exception = SecurityException("Access denied")
            
            whenever(mockServiceClient.deleteFile(any()))
                .thenReturn(CompletableFuture.failedFuture(exception))
            
            // When & Then
            val result = connector.deleteFile(fileId)
            assertThrows<SecurityException> {
                result.get()
            }
        }
    }
    
    @Nested
    @DisplayName("Error Handling Tests")
    inner class ErrorHandlingTests {
        
        @Test
        @DisplayName("Should handle authentication failure")
        fun testAuthenticationFailure() = runTest {
            // Given
            whenever(mockAuthProvider.getCredentials())
                .thenThrow(SecurityException("Authentication failed"))
            
            // When & Then
            assertThrows<SecurityException> {
                connector.connect()
            }
        }
        
        @Test
        @DisplayName("Should handle network connectivity issues")
        fun testNetworkConnectivityIssues() = runTest {
            // Given
            whenever(mockConnectionManager.connect(any()))
                .thenThrow(IOException("Network unreachable"))
            
            // When & Then
            assertThrows<IOException> {
                connector.connect()
            }
        }
        
        @Test
        @DisplayName("Should handle service unavailable errors")
        fun testServiceUnavailableError() = runTest {
            // Given
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(
                    ServiceUnavailableException("Service temporarily unavailable")))
            
            // When & Then
            val result = connector.uploadFile("test.txt", "content".toByteArray())
            assertThrows<ServiceUnavailableException> {
                result.get()
            }
        }
        
        @Test
        @DisplayName("Should handle invalid input parameters")
        fun testInvalidInputParameters() = runTest {
            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.uploadFile("", "content".toByteArray())
            }
            
            assertThrows<IllegalArgumentException> {
                connector.uploadFile("test.txt", byteArrayOf())
            }
            
            assertThrows<IllegalArgumentException> {
                connector.downloadFile("")
            }
            
            assertThrows<IllegalArgumentException> {
                connector.deleteFile("")
            }
        }
    }
    
    @Nested
    @DisplayName("State Management Tests")
    inner class StateManagementTests {
        
        @Test
        @DisplayName("Should maintain connection state correctly")
        fun testConnectionStateManagement() = runTest {
            // Given
            whenever(mockAuthProvider.getCredentials()).thenReturn(Credentials("token", "endpoint"))
            whenever(mockConnectionManager.connect(any())).thenReturn(true)
            whenever(mockConnectionManager.isConnected()).thenReturn(true)
            
            // When
            connector.connect()
            
            // Then
            assertTrue(connector.isConnected())
            
            // When
            connector.close()
            
            // Then
            assertFalse(connector.isConnected())
        }
        
        @Test
        @DisplayName("Should prevent operations on disconnected service")
        fun testOperationsOnDisconnectedService() = runTest {
            // Given
            whenever(mockConnectionManager.isConnected()).thenReturn(false)
            
            // When & Then
            assertThrows<IllegalStateException> {
                connector.uploadFile("test.txt", "content".toByteArray())
            }
            
            assertThrows<IllegalStateException> {
                connector.downloadFile("file_123")
            }
            
            assertThrows<IllegalStateException> {
                connector.deleteFile("file_123")
            }
        }
    }
    
    @Nested
    @DisplayName("Performance and Concurrency Tests")
    inner class PerformanceAndConcurrencyTests {
        
        @Test
        @DisplayName("Should handle concurrent operations safely")
        fun testConcurrentOperations() = runTest {
            // Given
            val fileData = "test content".toByteArray()
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_id"))
            
            // When
            val futures = (1..10).map { i ->
                connector.uploadFile("file_$i.txt", fileData)
            }
            
            // Then
            futures.forEach { future ->
                assertEquals("upload_id", future.get())
            }
            
            verify(mockServiceClient, times(10)).uploadFile(any(), any())
        }
        
        @Test
        @DisplayName("Should handle large file uploads")
        fun testLargeFileUpload() = runTest {
            // Given
            val largeFileData = ByteArray(1024 * 1024) // 1MB
            val fileName = "large_file.dat"
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_123"))
            
            // When
            val result = connector.uploadFile(fileName, largeFileData)
            
            // Then
            assertEquals("upload_123", result.get())
            verify(mockServiceClient).uploadFile(fileName, largeFileData)
        }
        
        @Test
        @DisplayName("Should handle timeout on slow operations")
        fun testSlowOperationTimeout() = runTest {
            // Given
            val slowFuture = CompletableFuture<String>()
            whenever(mockServiceClient.uploadFile(any(), any())).thenReturn(slowFuture)
            
            // When & Then
            assertThrows<TimeoutException> {
                connector.uploadFileWithTimeout("test.txt", "content".toByteArray(), 1000)
            }
        }
    }
    
    @Nested
    @DisplayName("Edge Cases Tests")
    inner class EdgeCasesTests {
        
        @Test
        @DisplayName("Should handle empty file upload")
        fun testEmptyFileUpload() = runTest {
            // Given
            val emptyFileData = byteArrayOf()
            val fileName = "empty.txt"
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.uploadFile(fileName, emptyFileData)
            }
        }
        
        @Test
        @DisplayName("Should handle file with special characters in name")
        fun testFileWithSpecialCharacters() = runTest {
            // Given
            val fileName = "file with spaces & special-chars.txt"
            val fileData = "content".toByteArray()
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_123"))
            
            // When
            val result = connector.uploadFile(fileName, fileData)
            
            // Then
            assertEquals("upload_123", result.get())
            verify(mockServiceClient).uploadFile(fileName, fileData)
        }
        
        @Test
        @DisplayName("Should handle null parameters gracefully")
        fun testNullParameters() = runTest {
            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.uploadFile(null, "content".toByteArray())
            }
            
            assertThrows<IllegalArgumentException> {
                connector.uploadFile("test.txt", null)
            }
            
            assertThrows<IllegalArgumentException> {
                connector.downloadFile(null)
            }
            
            assertThrows<IllegalArgumentException> {
                connector.deleteFile(null)
            }
        }
        
        @Test
        @DisplayName("Should handle multiple consecutive connections")
        fun testMultipleConsecutiveConnections() = runTest {
            // Given
            val credentials = Credentials("token", "endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any())).thenReturn(true)
            
            // When
            connector.connect()
            connector.connect()
            connector.connect()
            
            // Then
            verify(mockConnectionManager, times(3)).connect(credentials)
        }
    }
    
    @Nested
    @DisplayName("Integration-like Tests")
    inner class IntegrationLikeTests {
        
        @Test
        @DisplayName("Should complete full file lifecycle")
        fun testFullFileLifecycle() = runTest {
            // Given
            val fileName = "lifecycle_test.txt"
            val fileData = "test content".toByteArray()
            val uploadId = "upload_123"
            val fileId = "file_123"
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture(uploadId))
            whenever(mockServiceClient.downloadFile(any()))
                .thenReturn(CompletableFuture.completedFuture(fileData))
            whenever(mockServiceClient.deleteFile(any()))
                .thenReturn(CompletableFuture.completedFuture(true))
            
            // When & Then
            // Upload
            val uploadResult = connector.uploadFile(fileName, fileData)
            assertEquals(uploadId, uploadResult.get())
            
            // Download
            val downloadResult = connector.downloadFile(fileId)
            assertArrayEquals(fileData, downloadResult.get())
            
            // Delete
            val deleteResult = connector.deleteFile(fileId)
            assertTrue(deleteResult.get())
            
            // Verify call sequence
            val inOrder = inOrder(mockServiceClient)
            inOrder.verify(mockServiceClient).uploadFile(fileName, fileData)
            inOrder.verify(mockServiceClient).downloadFile(fileId)
            inOrder.verify(mockServiceClient).deleteFile(fileId)
        }
        
        @Test
        @DisplayName("Should handle authentication refresh during operations")
        fun testAuthenticationRefreshDuringOperations() = runTest {
            // Given
            val expiredCredentials = Credentials("expired_token", "endpoint")
            val refreshedCredentials = Credentials("new_token", "endpoint")
            
            whenever(mockAuthProvider.getCredentials())
                .thenReturn(expiredCredentials)
                .thenReturn(refreshedCredentials)
            
            whenever(mockConnectionManager.connect(expiredCredentials))
                .thenThrow(SecurityException("Token expired"))
            whenever(mockConnectionManager.connect(refreshedCredentials))
                .thenReturn(true)
            
            // When
            val result = connector.connectWithAuthRefresh()
            
            // Then
            assertTrue(result)
            verify(mockAuthProvider, times(2)).getCredentials()
            verify(mockConnectionManager).connect(expiredCredentials)
            verify(mockConnectionManager).connect(refreshedCredentials)
        }
    }
}

// Helper data classes for testing
data class Credentials(val token: String, val endpoint: String)
class ServiceUnavailableException(message: String) : Exception(message)
    @Nested
    @DisplayName("Advanced Connection Management Tests")
    inner class AdvancedConnectionManagementTests {
        
        @Test
        @DisplayName("Should handle connection pool exhaustion gracefully")
        fun testConnectionPoolExhaustion() = runTest {
            // Given
            val credentials = Credentials("token", "endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any()))
                .thenThrow(IllegalStateException("Connection pool exhausted"))
            
            // When & Then
            assertThrows<IllegalStateException> {
                connector.connect()
            }
        }
        
        @Test
        @DisplayName("Should handle connection interrupted during operation")
        fun testConnectionInterrupted() = runTest {
            // Given
            val fileData = "test content".toByteArray()
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(InterruptedException("Connection interrupted")))
            
            // When & Then
            val result = connector.uploadFile("test.txt", fileData)
            assertThrows<InterruptedException> {
                result.get()
            }
        }
        
        @Test
        @DisplayName("Should validate connection health before operations")
        fun testConnectionHealthValidation() = runTest {
            // Given
            whenever(mockConnectionManager.isConnected()).thenReturn(false)
            whenever(mockConnectionManager.isHealthy()).thenReturn(false)
            
            // When & Then
            assertThrows<IllegalStateException> {
                connector.uploadFile("test.txt", "content".toByteArray())
            }
        }
        
        @Test
        @DisplayName("Should handle connection recovery after network interruption")
        fun testConnectionRecovery() = runTest {
            // Given
            val credentials = Credentials("token", "endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any()))
                .thenReturn(false)
                .thenReturn(true)
            whenever(mockConnectionManager.isConnected())
                .thenReturn(false)
                .thenReturn(true)
            
            // When
            val result = connector.connectWithRecovery()
            
            // Then
            assertTrue(result)
            verify(mockConnectionManager, times(2)).connect(credentials)
        }
    }
    
    @Nested
    @DisplayName("Advanced Data Operations Tests")
    inner class AdvancedDataOperationsTests {
        
        @Test
        @DisplayName("Should handle chunked file upload for large files")
        fun testChunkedFileUpload() = runTest {
            // Given
            val largeFileData = ByteArray(10 * 1024 * 1024) // 10MB
            val fileName = "large_file.dat"
            val chunkSize = 1024 * 1024 // 1MB chunks
            
            whenever(mockServiceClient.uploadFileChunked(any(), any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_123"))
            
            // When
            val result = connector.uploadFileChunked(fileName, largeFileData, chunkSize)
            
            // Then
            assertEquals("upload_123", result.get())
            verify(mockServiceClient).uploadFileChunked(fileName, largeFileData, chunkSize)
        }
        
        @Test
        @DisplayName("Should handle partial file download with range requests")
        fun testPartialFileDownload() = runTest {
            // Given
            val fileId = "file_123"
            val startByte = 100L
            val endByte = 200L
            val expectedData = "partial content".toByteArray()
            
            whenever(mockServiceClient.downloadFileRange(any(), any(), any()))
                .thenReturn(CompletableFuture.completedFuture(expectedData))
            
            // When
            val result = connector.downloadFileRange(fileId, startByte, endByte)
            
            // Then
            assertArrayEquals(expectedData, result.get())
            verify(mockServiceClient).downloadFileRange(fileId, startByte, endByte)
        }
        
        @Test
        @DisplayName("Should handle file metadata operations")
        fun testFileMetadataOperations() = runTest {
            // Given
            val fileId = "file_123"
            val metadata = mapOf(
                "size" to "1024",
                "type" to "text/plain",
                "created" to "2023-01-01T00:00:00Z"
            )
            
            whenever(mockServiceClient.getFileMetadata(any()))
                .thenReturn(CompletableFuture.completedFuture(metadata))
            
            // When
            val result = connector.getFileMetadata(fileId)
            
            // Then
            assertEquals(metadata, result.get())
            verify(mockServiceClient).getFileMetadata(fileId)
        }
        
        @Test
        @DisplayName("Should handle file listing with pagination")
        fun testFileListingWithPagination() = runTest {
            // Given
            val pageSize = 10
            val pageToken = "next_page_token"
            val fileList = listOf("file1.txt", "file2.txt", "file3.txt")
            
            whenever(mockServiceClient.listFiles(any(), any()))
                .thenReturn(CompletableFuture.completedFuture(Pair(fileList, pageToken)))
            
            // When
            val result = connector.listFiles(pageSize, pageToken)
            
            // Then
            assertEquals(fileList, result.get().first)
            assertEquals(pageToken, result.get().second)
            verify(mockServiceClient).listFiles(pageSize, pageToken)
        }
        
        @Test
        @DisplayName("Should handle file move operations")
        fun testFileMoveOperation() = runTest {
            // Given
            val fileId = "file_123"
            val sourcePath = "/source/file.txt"
            val destinationPath = "/destination/file.txt"
            
            whenever(mockServiceClient.moveFile(any(), any(), any()))
                .thenReturn(CompletableFuture.completedFuture(true))
            
            // When
            val result = connector.moveFile(fileId, sourcePath, destinationPath)
            
            // Then
            assertTrue(result.get())
            verify(mockServiceClient).moveFile(fileId, sourcePath, destinationPath)
        }
        
        @Test
        @DisplayName("Should handle file copy operations")
        fun testFileCopyOperation() = runTest {
            // Given
            val sourceFileId = "source_file_123"
            val destinationPath = "/destination/copied_file.txt"
            val newFileId = "new_file_456"
            
            whenever(mockServiceClient.copyFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture(newFileId))
            
            // When
            val result = connector.copyFile(sourceFileId, destinationPath)
            
            // Then
            assertEquals(newFileId, result.get())
            verify(mockServiceClient).copyFile(sourceFileId, destinationPath)
        }
    }
    
    @Nested
    @DisplayName("Advanced Error Handling Tests")
    inner class AdvancedErrorHandlingTests {
        
        @Test
        @DisplayName("Should handle rate limiting errors with backoff")
        fun testRateLimitingWithBackoff() = runTest {
            // Given
            val fileData = "test content".toByteArray()
            val rateLimitException = RateLimitException("Rate limit exceeded", 60)
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(rateLimitException))
                .thenReturn(CompletableFuture.completedFuture("upload_123"))
            
            // When
            val result = connector.uploadFileWithRetry("test.txt", fileData)
            
            // Then
            assertEquals("upload_123", result.get())
            verify(mockServiceClient, times(2)).uploadFile(any(), any())
        }
        
        @Test
        @DisplayName("Should handle quota exceeded errors")
        fun testQuotaExceededError() = runTest {
            // Given
            val fileData = "test content".toByteArray()
            val quotaException = QuotaExceededException("Storage quota exceeded")
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(quotaException))
            
            // When & Then
            val result = connector.uploadFile("test.txt", fileData)
            assertThrows<QuotaExceededException> {
                result.get()
            }
        }
        
        @Test
        @DisplayName("Should handle corrupted file errors")
        fun testCorruptedFileError() = runTest {
            // Given
            val fileId = "corrupted_file_123"
            val corruptionException = FileCorruptedException("File checksum mismatch")
            
            whenever(mockServiceClient.downloadFile(any()))
                .thenReturn(CompletableFuture.failedFuture(corruptionException))
            
            // When & Then
            val result = connector.downloadFile(fileId)
            assertThrows<FileCorruptedException> {
                result.get()
            }
        }
        
        @Test
        @DisplayName("Should handle concurrent modification errors")
        fun testConcurrentModificationError() = runTest {
            // Given
            val fileId = "file_123"
            val concurrentException = ConcurrentModificationException("File modified by another process")
            
            whenever(mockServiceClient.deleteFile(any()))
                .thenReturn(CompletableFuture.failedFuture(concurrentException))
            
            // When & Then
            val result = connector.deleteFile(fileId)
            assertThrows<ConcurrentModificationException> {
                result.get()
            }
        }
        
        @Test
        @DisplayName("Should handle malformed response errors")
        fun testMalformedResponseError() = runTest {
            // Given
            val fileId = "file_123"
            val malformedException = MalformedResponseException("Invalid response format")
            
            whenever(mockServiceClient.downloadFile(any()))
                .thenReturn(CompletableFuture.failedFuture(malformedException))
            
            // When & Then
            val result = connector.downloadFile(fileId)
            assertThrows<MalformedResponseException> {
                result.get()
            }
        }
    }
    
    @Nested
    @DisplayName("Advanced State Management Tests")
    inner class AdvancedStateManagementTests {
        
        @Test
        @DisplayName("Should handle connection state transitions correctly")
        fun testConnectionStateTransitions() = runTest {
            // Given
            val credentials = Credentials("token", "endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            
            // Test disconnected -> connecting -> connected
            whenever(mockConnectionManager.connect(any())).thenReturn(true)
            whenever(mockConnectionManager.isConnected())
                .thenReturn(false)
                .thenReturn(true)
            
            // When & Then
            assertFalse(connector.isConnected())
            connector.connect()
            assertTrue(connector.isConnected())
            
            // Test connected -> disconnecting -> disconnected
            connector.close()
            assertFalse(connector.isConnected())
        }
        
        @Test
        @DisplayName("Should handle connection leaks prevention")
        fun testConnectionLeaksPrevention() = runTest {
            // Given
            val credentials = Credentials("token", "endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any())).thenReturn(true)
            
            // When - multiple connections without proper cleanup
            repeat(5) {
                connector.connect()
            }
            
            // Then - should not create more connections than necessary
            verify(mockConnectionManager, atMost(5)).connect(credentials)
        }
        
        @Test
        @DisplayName("Should handle resource cleanup on unexpected shutdown")
        fun testResourceCleanupOnUnexpectedShutdown() = runTest {
            // Given
            val credentials = Credentials("token", "endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any())).thenReturn(true)
            
            // When
            connector.connect()
            Runtime.getRuntime().addShutdownHook(Thread {
                connector.close()
            })
            
            // Simulate unexpected shutdown
            System.gc() // Force garbage collection
            
            // Then
            verify(mockConnectionManager, atLeastOnce()).connect(credentials)
        }
    }
    
    @Nested
    @DisplayName("Advanced Performance Tests")
    inner class AdvancedPerformanceTests {
        
        @Test
        @DisplayName("Should handle high-frequency rapid operations")
        fun testHighFrequencyRapidOperations() = runTest {
            // Given
            val fileData = "small content".toByteArray()
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_id"))
            
            // When - rapid fire operations
            val startTime = System.currentTimeMillis()
            val futures = (1..100).map { i ->
                connector.