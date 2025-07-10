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
    @DisplayName("Advanced Edge Cases and Boundary Tests")
    inner class AdvancedEdgeCasesTests {
        
        @Test
        @DisplayName("Should handle extremely long file names")
        fun testExtremelyLongFileName() = runTest {
            // Given
            val longFileName = "a".repeat(1000) + ".txt"
            val fileData = "content".toByteArray()
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_123"))
            
            // When
            val result = connector.uploadFile(longFileName, fileData)
            
            // Then
            assertEquals("upload_123", result.get())
            verify(mockServiceClient).uploadFile(longFileName, fileData)
        }
        
        @Test
        @DisplayName("Should handle unicode characters in file names")
        fun testUnicodeFileNames() = runTest {
            // Given
            val unicodeFileName = "测试文件_🚀_файл.txt"
            val fileData = "unicode content".toByteArray()
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_unicode"))
            
            // When
            val result = connector.uploadFile(unicodeFileName, fileData)
            
            // Then
            assertEquals("upload_unicode", result.get())
            verify(mockServiceClient).uploadFile(unicodeFileName, fileData)
        }
        
        @Test
        @DisplayName("Should handle binary file data correctly")
        fun testBinaryFileData() = runTest {
            // Given
            val binaryData = byteArrayOf(0x00, 0xFF.toByte(), 0x7F, 0x80.toByte(), 0x01)
            val fileName = "binary.dat"
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("binary_upload"))
            whenever(mockServiceClient.downloadFile(any()))
                .thenReturn(CompletableFuture.completedFuture(binaryData))
            
            // When
            val uploadResult = connector.uploadFile(fileName, binaryData)
            val downloadResult = connector.downloadFile("binary_file_id")
            
            // Then
            assertEquals("binary_upload", uploadResult.get())
            assertArrayEquals(binaryData, downloadResult.get())
        }
        
        @Test
        @DisplayName("Should handle file names with path separators")
        fun testFileNamesWithPathSeparators() = runTest {
            // Given
            val fileName = "folder/subfolder/file.txt"
            val fileData = "content".toByteArray()
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("path_upload"))
            
            // When
            val result = connector.uploadFile(fileName, fileData)
            
            // Then
            assertEquals("path_upload", result.get())
            verify(mockServiceClient).uploadFile(fileName, fileData)
        }
        
        @Test
        @DisplayName("Should handle maximum size file data")
        fun testMaximumSizeFileData() = runTest {
            // Given
            val maxSizeData = ByteArray(100 * 1024 * 1024) // 100MB
            val fileName = "max_size.dat"
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("max_upload"))
            
            // When
            val result = connector.uploadFile(fileName, maxSizeData)
            
            // Then
            assertEquals("max_upload", result.get())
            verify(mockServiceClient).uploadFile(fileName, maxSizeData)
        }
        
        @Test
        @DisplayName("Should handle whitespace-only file names")
        fun testWhitespaceOnlyFileNames() = runTest {
            // Given
            val whitespaceFileName = "   \t\n   "
            val fileData = "content".toByteArray()
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.uploadFile(whitespaceFileName, fileData)
            }
        }
        
        @Test
        @DisplayName("Should handle file names with only dots")
        fun testFileNamesWithOnlyDots() = runTest {
            // Given
            val dotFileName = "..."
            val fileData = "content".toByteArray()
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.uploadFile(dotFileName, fileData)
            }
        }
    }
    
    @Nested
    @DisplayName("Advanced Error Handling and Recovery Tests")
    inner class AdvancedErrorHandlingTests {
        
        @Test
        @DisplayName("Should handle partial upload failures with retry")
        fun testPartialUploadFailureWithRetry() = runTest {
            // Given
            val fileData = "content".toByteArray()
            val fileName = "partial.txt"
            val partialException = IOException("Partial upload failed")
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(partialException))
                .thenReturn(CompletableFuture.completedFuture("retry_success"))
            
            // When
            val result = connector.uploadFileWithRetry(fileName, fileData, maxRetries = 2)
            
            // Then
            assertEquals("retry_success", result.get())
            verify(mockServiceClient, times(2)).uploadFile(fileName, fileData)
        }
        
        @Test
        @DisplayName("Should handle connection drops during file operations")
        fun testConnectionDropDuringOperations() = runTest {
            // Given
            val fileName = "connection_drop.txt"
            val fileData = "content".toByteArray()
            
            whenever(mockConnectionManager.isConnected())
                .thenReturn(true)
                .thenReturn(false)
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(IOException("Connection lost")))
            
            // When & Then
            assertThrows<IOException> {
                connector.uploadFile(fileName, fileData).get()
            }
        }
        
        @Test
        @DisplayName("Should handle rate limiting gracefully")
        fun testRateLimitingHandling() = runTest {
            // Given
            val fileName = "rate_limited.txt"
            val fileData = "content".toByteArray()
            val rateLimitException = RateLimitException("Rate limit exceeded")
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(rateLimitException))
                .thenReturn(CompletableFuture.completedFuture("rate_limit_success"))
            
            // When
            val result = connector.uploadFileWithBackoff(fileName, fileData)
            
            // Then
            assertEquals("rate_limit_success", result.get())
            verify(mockServiceClient, times(2)).uploadFile(fileName, fileData)
        }
        
        @Test
        @DisplayName("Should handle memory exhaustion during large file operations")
        fun testMemoryExhaustionHandling() = runTest {
            // Given
            val fileName = "memory_test.txt"
            val fileData = "content".toByteArray()
            val memoryException = OutOfMemoryError("Java heap space")
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(memoryException))
            
            // When & Then
            assertThrows<OutOfMemoryError> {
                connector.uploadFile(fileName, fileData).get()
            }
        }
        
        @Test
        @DisplayName("Should handle disk space issues")
        fun testDiskSpaceIssues() = runTest {
            // Given
            val fileName = "disk_space.txt"
            val fileData = "content".toByteArray()
            val diskSpaceException = IOException("No space left on device")
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(diskSpaceException))
            
            // When & Then
            assertThrows<IOException> {
                connector.uploadFile(fileName, fileData).get()
            }
        }
        
        @Test
        @DisplayName("Should handle corrupted file data during download")
        fun testCorruptedFileDataDownload() = runTest {
            // Given
            val fileId = "corrupted_file"
            val corruptionException = DataCorruptionException("File data is corrupted")
            
            whenever(mockServiceClient.downloadFile(any()))
                .thenReturn(CompletableFuture.failedFuture(corruptionException))
            
            // When & Then
            assertThrows<DataCorruptionException> {
                connector.downloadFile(fileId).get()
            }
        }
        
        @Test
        @DisplayName("Should handle service maintenance mode")
        fun testServiceMaintenanceMode() = runTest {
            // Given
            val fileName = "maintenance.txt"
            val fileData = "content".toByteArray()
            val maintenanceException = ServiceMaintenanceException("Service under maintenance")
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(maintenanceException))
            
            // When & Then
            assertThrows<ServiceMaintenanceException> {
                connector.uploadFile(fileName, fileData).get()
            }
        }
    }
    
    @Nested
    @DisplayName("Advanced State Management and Resource Tests")
    inner class AdvancedStateManagementTests {
        
        @Test
        @DisplayName("Should handle connection pool exhaustion")
        fun testConnectionPoolExhaustion() = runTest {
            // Given
            val poolException = ConnectionPoolExhaustedException("Connection pool exhausted")
            whenever(mockConnectionManager.connect(any()))
                .thenThrow(poolException)
            
            // When & Then
            assertThrows<ConnectionPoolExhaustedException> {
                connector.connect()
            }
        }
        
        @Test
        @DisplayName("Should handle session expiration during operations")
        fun testSessionExpirationDuringOperations() = runTest {
            // Given
            val fileName = "session_expired.txt"
            val fileData = "content".toByteArray()
            val sessionException = SessionExpiredException("Session expired")
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(sessionException))
            
            // When & Then
            assertThrows<SessionExpiredException> {
                connector.uploadFile(fileName, fileData).get()
            }
        }
        
        @Test
        @DisplayName("Should handle cleanup on JVM shutdown")
        fun testCleanupOnJVMShutdown() = runTest {
            // Given
            val shutdownHook = mock<Thread>()
            whenever(mockConnectionManager.isConnected()).thenReturn(true)
            
            // When
            connector.registerShutdownHook(shutdownHook)
            connector.close()
            
            // Then
            verify(mockConnectionManager).close()
            verify(mockServiceClient).shutdown()
        }
        
        @Test
        @DisplayName("Should detect and handle resource leaks")
        fun testResourceLeakDetection() = runTest {
            // Given
            val fileName = "leak_test.txt"
            val fileData = "content".toByteArray()
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_123"))
            
            // When
            repeat(1000) {
                connector.uploadFile("$fileName$it", fileData)
            }
            
            // Then
            verify(mockServiceClient, times(1000)).uploadFile(any(), any())
            assertTrue(connector.getResourceUsageStats().connectionsCreated <= 1000)
        }
        
        @Test
        @DisplayName("Should handle connection timeout during active operations")
        fun testConnectionTimeoutDuringOperations() = runTest {
            // Given
            val fileName = "timeout_test.txt"
            val fileData = "content".toByteArray()
            val slowFuture = CompletableFuture<String>()
            
            whenever(mockServiceClient.uploadFile(any(), any())).thenReturn(slowFuture)
            
            // When & Then
            assertThrows<TimeoutException> {
                connector.uploadFileWithTimeout(fileName, fileData, 100)
            }
        }
        
        @Test
        @DisplayName("Should handle connection reuse efficiency")
        fun testConnectionReuseEfficiency() = runTest {
            // Given
            val credentials = Credentials("token", "endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any())).thenReturn(true)
            whenever(mockConnectionManager.isConnected()).thenReturn(true)
            
            // When
            connector.connect()
            repeat(10) {
                connector.uploadFile("reuse_test_$it.txt", "content".toByteArray())
            }
            
            // Then
            verify(mockConnectionManager, times(1)).connect(credentials)
            verify(mockServiceClient, times(10)).uploadFile(any(), any())
        }
    }
    
    @Nested
    @DisplayName("Advanced Performance and Stress Tests")
    inner class AdvancedPerformanceTests {
        
        @Test
        @DisplayName("Should handle stress test with many small files")
        fun testStressTestManySmallFiles() = runTest {
            // Given
            val smallFileData = "small".toByteArray()
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_success"))
            
            // When
            val futures = (1..1000).map { i ->
                connector.uploadFile("small_$i.txt", smallFileData)
            }
            
            // Then
            futures.forEach { future ->
                assertEquals("upload_success", future.get())
            }
            verify(mockServiceClient, times(1000)).uploadFile(any(), any())
        }
        
        @Test
        @DisplayName("Should validate upload/download speed metrics")
        fun testUploadDownloadSpeedMetrics() = runTest {
            // Given
            val largeFileData = ByteArray(10 * 1024 * 1024) // 10MB
            val fileName = "speed_test.dat"
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("speed_upload"))
            whenever(mockServiceClient.downloadFile(any()))
                .thenReturn(CompletableFuture.completedFuture(largeFileData))
            
            // When
            val uploadStart = System.currentTimeMillis()
            val uploadResult = connector.uploadFile(fileName, largeFileData)
            uploadResult.get()
            val uploadEnd = System.currentTimeMillis()
            
            val downloadStart = System.currentTimeMillis()
            val downloadResult = connector.downloadFile("speed_file_id")
            downloadResult.get()
            val downloadEnd = System.currentTimeMillis()
            
            // Then
            val uploadTime = uploadEnd - uploadStart
            val downloadTime = downloadEnd - downloadStart
            assertTrue(uploadTime < 5000) // Should complete within 5 seconds
            assertTrue(downloadTime < 5000) // Should complete within 5 seconds
        }
        
        @Test
        @DisplayName("Should monitor memory usage during operations")
        fun testMemoryUsageDuringOperations() = runTest {
            // Given
            val mediumFileData = ByteArray(1024 * 1024) // 1MB
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("memory_upload"))
            
            // When
            val initialMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
            repeat(100) {
                connector.uploadFile("memory_test_$it.txt", mediumFileData)
            }
            val finalMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
            
            // Then
            val memoryIncrease = finalMemory - initialMemory
            assertTrue(memoryIncrease < 100 * 1024 * 1024) // Should not increase by more than 100MB
        }
        
        @Test
        @DisplayName("Should handle concurrent upload and download operations")
        fun testConcurrentUploadDownloadOperations() = runTest {
            // Given
            val fileData = "concurrent content".toByteArray()
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("concurrent_upload"))
            whenever(mockServiceClient.downloadFile(any()))
                .thenReturn(CompletableFuture.completedFuture(fileData))
            
            // When
            val uploadFutures = (1..50).map { i ->
                connector.uploadFile("upload_$i.txt", fileData)
            }
            val downloadFutures = (1..50).map { i ->
                connector.downloadFile("download_$i")
            }
            
            // Then
            uploadFutures.forEach { future ->
                assertEquals("concurrent_upload", future.get())
            }
            downloadFutures.forEach { future ->
                assertArrayEquals(fileData, future.get())
            }
        }
        
        @Test
        @DisplayName("Should handle batch operations efficiently")
        fun testBatchOperationsEfficiency() = runTest {
            // Given
            val batchFileData = "batch content".toByteArray()
            val batchFiles = (1..100).map { "batch_$it.txt" to batchFileData }
            
            whenever(mockServiceClient.uploadBatch(any()))
                .thenReturn(CompletableFuture.completedFuture(
                    batchFiles.map { "upload_${it.first}" }
                ))
            
            // When
            val result = connector.uploadBatch(batchFiles)
            
            // Then
            val uploadIds = result.get()
            assertEquals(100, uploadIds.size)
            assertTrue(uploadIds.all { it.startsWith("upload_batch_") })
        }
    }
    
    @Nested
    @DisplayName("Advanced Integration and Workflow Tests")
    inner class AdvancedIntegrationTests {
        
        @Test
        @DisplayName("Should handle multiple concurrent connections to different endpoints")
        fun testMultipleConcurrentConnections() = runTest {
            // Given
            val endpoint1 = Credentials("token1", "endpoint1")
            val endpoint2 = Credentials("token2", "endpoint2")
            
            whenever(mockAuthProvider.getCredentials())
                .thenReturn(endpoint1)
                .thenReturn(endpoint2)
            whenever(mockConnectionManager.connect(any())).thenReturn(true)
            
            // When
            val connector1 = OracleDriveServiceConnector(mockServiceClient, mockConnectionManager, mockAuthProvider)
            val connector2 = OracleDriveServiceConnector(mockServiceClient, mockConnectionManager, mockAuthProvider)
            
            connector1.connect()
            connector2.connect()
            
            // Then
            verify(mockConnectionManager).connect(endpoint1)
            verify(mockConnectionManager).connect(endpoint2)
        }
        
        @Test
        @DisplayName("Should handle connection failover scenarios")
        fun testConnectionFailoverScenarios() = runTest {
            // Given
            val primaryEndpoint = Credentials("token", "primary_endpoint")
            val backupEndpoint = Credentials("token", "backup_endpoint")
            
            whenever(mockAuthProvider.getCredentials())
                .thenReturn(primaryEndpoint)
                .thenReturn(backupEndpoint)
            whenever(mockConnectionManager.connect(primaryEndpoint))
                .thenThrow(IOException("Primary endpoint unreachable"))
            whenever(mockConnectionManager.connect(backupEndpoint))
                .thenReturn(true)
            
            // When
            val result = connector.connectWithFailover()
            
            // Then
            assertTrue(result)
            verify(mockConnectionManager).connect(primaryEndpoint)
            verify(mockConnectionManager).connect(backupEndpoint)
        }
        
        @Test
        @DisplayName("Should handle service degradation gracefully")
        fun testServiceDegradationHandling() = runTest {
            // Given
            val fileName = "degraded.txt"
            val fileData = "content".toByteArray()
            val degradationException = ServiceDegradedException("Service running in degraded mode")
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(degradationException))
                .thenReturn(CompletableFuture.completedFuture("degraded_upload"))
            
            // When
            val result = connector.uploadFileWithDegradationHandling(fileName, fileData)
            
            // Then
            assertEquals("degraded_upload", result.get())
            verify(mockServiceClient, times(2)).uploadFile(fileName, fileData)
        }
        
        @Test
        @DisplayName("Should handle file checksum validation")
        fun testFileChecksumValidation() = runTest {
            // Given
            val fileName = "checksum_test.txt"
            val fileData = "checksum content".toByteArray()
            val expectedChecksum = "abc123def456"
            
            whenever(mockServiceClient.uploadFileWithChecksum(any(), any(), any()))
                .thenReturn(CompletableFuture.completedFuture(UploadResult("upload_123", expectedChecksum)))
            
            // When
            val result = connector.uploadFileWithChecksum(fileName, fileData, expectedChecksum)
            
            // Then
            assertEquals("upload_123", result.get().uploadId)
            assertEquals(expectedChecksum, result.get().checksum)
        }
        
        @Test
        @DisplayName("Should handle progress tracking for large uploads")
        fun testProgressTrackingForLargeUploads() = runTest {
            // Given
            val largeFileData = ByteArray(50 * 1024 * 1024) // 50MB
            val fileName = "progress_test.dat"
            val progressCallback = mock<(Long, Long) -> Unit>()
            
            whenever(mockServiceClient.uploadFileWithProgress(any(), any(), any()))
                .thenReturn(CompletableFuture.completedFuture("progress_upload"))
            
            // When
            val result = connector.uploadFileWithProgress(fileName, largeFileData, progressCallback)
            
            // Then
            assertEquals("progress_upload", result.get())
            verify(progressCallback, atLeastOnce()).invoke(any(), any())
        }
        
        @Test
        @DisplayName("Should handle upload cancellation")
        fun testUploadCancellation() = runTest {
            // Given
            val fileName = "cancellation_test.txt"
            val fileData = "content".toByteArray()
            val cancellableFuture = CompletableFuture<String>()
            
            whenever(mockServiceClient.uploadFile(any(), any())).thenReturn(cancellableFuture)
            
            // When
            val uploadFuture = connector.uploadFile(fileName, fileData)
            uploadFuture.cancel(true)
            
            // Then
            assertTrue(uploadFuture.isCancelled)
        }
    }

// Additional exception classes for comprehensive testing
class RateLimitException(message: String) : Exception(message)
class DataCorruptionException(message: String) : Exception(message)
class ServiceMaintenanceException(message: String) : Exception(message)
class ConnectionPoolExhaustedException(message: String) : Exception(message)
class SessionExpiredException(message: String) : Exception(message)
class ServiceDegradedException(message: String) : Exception(message)

// Additional data classes for testing
data class UploadResult(val uploadId: String, val checksum: String)
<<<<<<< HEAD
data class ResourceUsageStats(val connectionsCreated: Int, val memoryUsed: Long)
=======
data class ResourceUsageStats(val connectionsCreated: Int, val memoryUsed: Long)
    @Nested
    @DisplayName("Configuration and Validation Tests")
    inner class ConfigurationValidationTests {

        @Test
        @DisplayName("Should validate configuration parameters at initialization")
        fun testConfigurationValidation() = runTest {
            // When & Then
            assertThrows<IllegalArgumentException> {
                OracleDriveServiceConnector(
                    serviceClient = null,
                    connectionManager = mockConnectionManager,
                    authProvider = mockAuthProvider
                )
            }

            assertThrows<IllegalArgumentException> {
                OracleDriveServiceConnector(
                    serviceClient = mockServiceClient,
                    connectionManager = null,
                    authProvider = mockAuthProvider
                )
            }

            assertThrows<IllegalArgumentException> {
                OracleDriveServiceConnector(
                    serviceClient = mockServiceClient,
                    connectionManager = mockConnectionManager,
                    authProvider = null
                )
            }
        }

        @Test
        @DisplayName("Should validate connection timeout configuration")
        fun testConnectionTimeoutConfiguration() = runTest {
            // Given
            val invalidTimeout = -1L
            val validTimeout = 30000L

            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.setConnectionTimeout(invalidTimeout)
            }

            assertDoesNotThrow {
                connector.setConnectionTimeout(validTimeout)
            }
        }

        @Test
        @DisplayName("Should validate maximum retry attempts configuration")
        fun testMaxRetryAttemptsConfiguration() = runTest {
            // Given
            val invalidRetries = -1
            val validRetries = 3

            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.setMaxRetryAttempts(invalidRetries)
            }

            assertDoesNotThrow {
                connector.setMaxRetryAttempts(validRetries)
            }
        }

        @Test
        @DisplayName("Should validate buffer size configuration")
        fun testBufferSizeConfiguration() = runTest {
            // Given
            val invalidBufferSize = 0
            val validBufferSize = 8192

            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.setBufferSize(invalidBufferSize)
            }

            assertDoesNotThrow {
                connector.setBufferSize(validBufferSize)
            }
        }

        @Test
        @DisplayName("Should validate endpoint URL format")
        fun testEndpointUrlValidation() = runTest {
            // Given
            val invalidEndpoints = listOf("", "not-a-url", "ftp://invalid.com")
            val validEndpoints = listOf("https://valid.com", "http://localhost:8080")

            // When & Then
            invalidEndpoints.forEach { endpoint ->
                assertThrows<IllegalArgumentException> {
                    connector.validateEndpoint(endpoint)
                }
            }

            validEndpoints.forEach { endpoint ->
                assertDoesNotThrow {
                    connector.validateEndpoint(endpoint)
                }
            }
        }
    }

    @Nested
    @DisplayName("Security and Authentication Tests")
    inner class SecurityAuthenticationTests {

        @Test
        @DisplayName("Should handle token refresh before expiration")
        fun testTokenRefreshBeforeExpiration() = runTest {
            // Given
            val expiringToken = Credentials("expiring_token", "endpoint")
            val refreshedToken = Credentials("refreshed_token", "endpoint")

            whenever(mockAuthProvider.getCredentials()).thenReturn(expiringToken)
            whenever(mockAuthProvider.isTokenExpiring(any())).thenReturn(true)
            whenever(mockAuthProvider.refreshToken()).thenReturn(refreshedToken)
            whenever(mockConnectionManager.connect(any())).thenReturn(true)

            // When
            connector.connect()

            // Then
            verify(mockAuthProvider).refreshToken()
            verify(mockConnectionManager).connect(refreshedToken)
        }

        @Test
        @DisplayName("Should handle multi-factor authentication")
        fun testMultiFactorAuthentication() = runTest {
            // Given
            val mfaCredentials = Credentials("mfa_token", "endpoint")
            val mfaCode = "123456"

            whenever(mockAuthProvider.requiresMFA()).thenReturn(true)
            whenever(mockAuthProvider.getCredentials()).thenReturn(mfaCredentials)
            whenever(mockAuthProvider.verifyMFA(any())).thenReturn(true)
            whenever(mockConnectionManager.connect(any())).thenReturn(true)

            // When
            val result = connector.connectWithMFA(mfaCode)

            // Then
            assertTrue(result)
            verify(mockAuthProvider).verifyMFA(mfaCode)
            verify(mockConnectionManager).connect(mfaCredentials)
        }

        @Test
        @DisplayName("Should handle invalid MFA code")
        fun testInvalidMFACode() = runTest {
            // Given
            val invalidMfaCode = "invalid"

            whenever(mockAuthProvider.requiresMFA()).thenReturn(true)
            whenever(mockAuthProvider.verifyMFA(any())).thenReturn(false)

            // When & Then
            assertThrows<SecurityException> {
                connector.connectWithMFA(invalidMfaCode)
            }
        }

        @Test
        @DisplayName("Should handle certificate validation")
        fun testCertificateValidation() = runTest {
            // Given
            val credentials = Credentials("token", "https://secure.endpoint.com")

            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.validateCertificate(any())).thenReturn(true)
            whenever(mockConnectionManager.connect(any())).thenReturn(true)

            // When
            val result = connector.connectWithCertificateValidation()

            // Then
            assertTrue(result)
            verify(mockConnectionManager).validateCertificate(credentials.endpoint)
        }

        @Test
        @DisplayName("Should handle certificate validation failure")
        fun testCertificateValidationFailure() = runTest {
            // Given
            val credentials = Credentials("token", "https://untrusted.endpoint.com")

            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.validateCertificate(any())).thenReturn(false)

            // When & Then
            assertThrows<SecurityException> {
                connector.connectWithCertificateValidation()
            }
        }

        @Test
        @DisplayName("Should handle role-based access control")
        fun testRoleBasedAccessControl() = runTest {
            // Given
            val adminCredentials = Credentials("admin_token", "endpoint")
            val userCredentials = Credentials("user_token", "endpoint")
            val fileName = "admin_only_file.txt"
            val fileData = "sensitive content".toByteArray()

            whenever(mockAuthProvider.getCredentials()).thenReturn(adminCredentials)
            whenever(mockAuthProvider.hasPermission(any(), any())).thenReturn(true)
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("admin_upload"))

            // When
            val result = connector.uploadFileWithPermissionCheck(fileName, fileData, "ADMIN")

            // Then
            assertEquals("admin_upload", result.get())
            verify(mockAuthProvider).hasPermission(adminCredentials, "ADMIN")
        }

        @Test
        @DisplayName("Should handle insufficient permissions")
        fun testInsufficientPermissions() = runTest {
            // Given
            val userCredentials = Credentials("user_token", "endpoint")
            val fileName = "admin_only_file.txt"
            val fileData = "sensitive content".toByteArray()

            whenever(mockAuthProvider.getCredentials()).thenReturn(userCredentials)
            whenever(mockAuthProvider.hasPermission(any(), any())).thenReturn(false)

            // When & Then
            assertThrows<SecurityException> {
                connector.uploadFileWithPermissionCheck(fileName, fileData, "ADMIN")
            }
        }
    }

    @Nested
    @DisplayName("Monitoring and Observability Tests")
    inner class MonitoringObservabilityTests {

        @Test
        @DisplayName("Should track connection metrics")
        fun testConnectionMetricsTracking() = runTest {
            // Given
            val credentials = Credentials("token", "endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any())).thenReturn(true)

            // When
            connector.connect()
            val metrics = connector.getConnectionMetrics()

            // Then
            assertEquals(1, metrics.totalConnections)
            assertEquals(1, metrics.successfulConnections)
            assertEquals(0, metrics.failedConnections)
            assertTrue(metrics.averageConnectionTime > 0)
        }

        @Test
        @DisplayName("Should track operation metrics")
        fun testOperationMetricsTracking() = runTest {
            // Given
            val fileName = "metrics_test.txt"
            val fileData = "content".toByteArray()

            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_123"))

            // When
            connector.uploadFile(fileName, fileData)
            val metrics = connector.getOperationMetrics()

            // Then
            assertEquals(1, metrics.totalOperations)
            assertEquals(1, metrics.successfulOperations)
            assertEquals(0, metrics.failedOperations)
            assertTrue(metrics.averageOperationTime > 0)
        }

        @Test
        @DisplayName("Should emit health check events")
        fun testHealthCheckEvents() = runTest {
            // Given
            val healthCheckListener = mock<HealthCheckListener>()
            connector.addHealthCheckListener(healthCheckListener)

            whenever(mockConnectionManager.isHealthy()).thenReturn(true)

            // When
            connector.performHealthCheck()

            // Then
            verify(healthCheckListener).onHealthCheckCompleted(true)
        }

        @Test
        @DisplayName("Should handle health check failures")
        fun testHealthCheckFailures() = runTest {
            // Given
            val healthCheckListener = mock<HealthCheckListener>()
            connector.addHealthCheckListener(healthCheckListener)

            whenever(mockConnectionManager.isHealthy()).thenReturn(false)

            // When
            connector.performHealthCheck()

            // Then
            verify(healthCheckListener).onHealthCheckCompleted(false)
        }

        @Test
        @DisplayName("Should track error rates")
        fun testErrorRateTracking() = runTest {
            // Given
            val fileName = "error_test.txt"
            val fileData = "content".toByteArray()

            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(IOException("Upload failed")))

            // When
            repeat(5) {
                try {
                    connector.uploadFile(fileName, fileData).get()
                } catch (e: Exception) {
                    // Expected
                }
            }

            val errorRate = connector.getErrorRate()

            // Then
            assertEquals(1.0, errorRate, 0.01) // 100% error rate
        }

        @Test
        @DisplayName("Should generate alerting when error threshold exceeded")
        fun testAlertingOnErrorThreshold() = runTest {
            // Given
            val alertListener = mock<AlertListener>()
            connector.addAlertListener(alertListener)
            connector.setErrorThreshold(0.5) // 50% error threshold

            val fileName = "alert_test.txt"
            val fileData = "content".toByteArray()

            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(IOException("Upload failed")))

            // When
            repeat(10) {
                try {
                    connector.uploadFile(fileName, fileData).get()
                } catch (e: Exception) {
                    // Expected
                }
            }

            // Then
            verify(alertListener, atLeastOnce()).onErrorThresholdExceeded(any())
        }
    }

    @Nested
    @DisplayName("Thread Safety and Concurrent Access Tests")
    inner class ThreadSafetyConcurrentAccessTests {

        @Test
        @DisplayName("Should handle concurrent connection attempts safely")
        fun testConcurrentConnectionAttempts() = runTest {
            // Given
            val credentials = Credentials("token", "endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any())).thenReturn(true)

            // When
            val results = (1..100).map {
                async { connector.connect() }
            }.awaitAll()

            // Then
            assertTrue(results.all { it })
            verify(mockConnectionManager, atLeast(1)).connect(credentials)
        }

        @Test
        @DisplayName("Should handle concurrent file operations safely")
        fun testConcurrentFileOperations() = runTest {
            // Given
            val fileData = "concurrent content".toByteArray()
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_success"))
            whenever(mockServiceClient.downloadFile(any()))
                .thenReturn(CompletableFuture.completedFuture(fileData))
            whenever(mockServiceClient.deleteFile(any()))
                .thenReturn(CompletableFuture.completedFuture(true))

            // When
            val operations = (1..50).flatMap { i ->
                listOf(
                    async { connector.uploadFile("file_$i.txt", fileData) },
                    async { connector.downloadFile("file_$i") },
                    async { connector.deleteFile("file_$i") }
                )
            }

            val results = operations.awaitAll()

            // Then
            assertEquals(150, results.size)
            verify(mockServiceClient, times(50)).uploadFile(any(), any())
            verify(mockServiceClient, times(50)).downloadFile(any())
            verify(mockServiceClient, times(50)).deleteFile(any())
        }

        @Test
        @DisplayName("Should handle thread interruption gracefully")
        fun testThreadInterruptionHandling() = runTest {
            // Given
            val fileName = "interrupt_test.txt"
            val fileData = "content".toByteArray()
            val interruptibleFuture = CompletableFuture<String>()

            whenever(mockServiceClient.uploadFile(any(), any())).thenReturn(interruptibleFuture)

            // When
            val uploadFuture = async { connector.uploadFile(fileName, fileData) }
            delay(100)
            uploadFuture.cancel()

            // Then
            assertTrue(uploadFuture.isCancelled)
        }

        @Test
        @DisplayName("Should maintain thread-safe state during operations")
        fun testThreadSafeStateManagement() = runTest {
            // Given
            val credentials = Credentials("token", "endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any())).thenReturn(true)
            whenever(mockConnectionManager.isConnected()).thenReturn(true)

            // When
            val connectOperations = (1..50).map {
                async { connector.connect() }
            }
            val stateChecks = (1..50).map {
                async { connector.isConnected() }
            }

            val connectResults = connectOperations.awaitAll()
            val stateResults = stateChecks.awaitAll()

            // Then
            assertTrue(connectResults.all { it })
            assertTrue(stateResults.all { it })
        }
    }

    @Nested
    @DisplayName("Resource Management and Cleanup Tests")
    inner class ResourceManagementCleanupTests {

        @Test
        @DisplayName("Should properly clean up resources on normal shutdown")
        fun testResourceCleanupOnNormalShutdown() = runTest {
            // Given
            val credentials = Credentials("token", "endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any())).thenReturn(true)
            whenever(mockConnectionManager.isConnected()).thenReturn(true)

            // When
            connector.connect()
            connector.shutdown()

            // Then
            verify(mockConnectionManager).close()
            verify(mockServiceClient).shutdown()
            verify(mockAuthProvider).cleanup()
        }

        @Test
        @DisplayName("Should handle resource cleanup on abnormal shutdown")
        fun testResourceCleanupOnAbnormalShutdown() = runTest {
            // Given
            val credentials = Credentials("token", "endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any())).thenReturn(true)
            whenever(mockConnectionManager.close()).thenThrow(IOException("Connection close failed"))

            // When
            connector.connect()
            assertDoesNotThrow {
                connector.shutdown()
            }

            // Then
            verify(mockConnectionManager).close()
            verify(mockServiceClient).shutdown()
        }

        @Test
        @DisplayName("Should handle memory leak prevention")
        fun testMemoryLeakPrevention() = runTest {
            // Given
            val fileName = "leak_prevention.txt"
            val fileData = "content".toByteArray()

            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_success"))

            // When
            repeat(1000) {
                connector.uploadFile("$fileName$it", fileData)
            }

            // Force garbage collection
            System.gc()

            // Then
            val memoryUsage = connector.getMemoryUsage()
            assertTrue(memoryUsage.totalMemory < 100 * 1024 * 1024) // Less than 100MB
        }

        @Test
        @DisplayName("Should handle connection pool cleanup")
        fun testConnectionPoolCleanup() = runTest {
            // Given
            val credentials = Credentials("token", "endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any())).thenReturn(true)
            whenever(mockConnectionManager.getActiveConnections()).thenReturn(5)

            // When
            repeat(10) {
                connector.connect()
            }
            connector.cleanupConnectionPool()

            // Then
            verify(mockConnectionManager).cleanupIdleConnections()
            assertTrue(connector.getActiveConnectionCount() <= 5)
        }

        @Test
        @DisplayName("Should handle file handle cleanup")
        fun testFileHandleCleanup() = runTest {
            // Given
            val fileName = "handle_cleanup.txt"
            val fileData = "content".toByteArray()

            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_success"))

            // When
            repeat(100) {
                connector.uploadFile("$fileName$it", fileData)
            }

            connector.cleanupFileHandles()

            // Then
            assertTrue(connector.getOpenFileHandleCount() == 0)
        }
    }

    @Nested
    @DisplayName("Compliance and Auditing Tests")
    inner class ComplianceAuditingTests {

        @Test
        @DisplayName("Should log all file operations for audit trail")
        fun testAuditTrailLogging() = runTest {
            // Given
            val auditLogger = mock<AuditLogger>()
            connector.setAuditLogger(auditLogger)

            val fileName = "audit_test.txt"
            val fileData = "sensitive content".toByteArray()

            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("audit_upload"))

            // When
            connector.uploadFile(fileName, fileData)

            // Then
            verify(auditLogger).logFileOperation(
                operation = "UPLOAD",
                fileName = fileName,
                fileSize = fileData.size.toLong(),
                timestamp = any(),
                userId = any()
            )
        }

        @Test
        @DisplayName("Should enforce data retention policies")
        fun testDataRetentionPolicyEnforcement() = runTest {
            // Given
            val fileName = "retention_test.txt"
            val fileData = "content".toByteArray()
            val retentionPolicy = RetentionPolicy(days = 30)

            whenever(mockServiceClient.uploadFileWithRetention(any(), any(), any()))
                .thenReturn(CompletableFuture.completedFuture("retention_upload"))

            // When
            val result = connector.uploadFileWithRetentionPolicy(fileName, fileData, retentionPolicy)

            // Then
            assertEquals("retention_upload", result.get())
            verify(mockServiceClient).uploadFileWithRetention(fileName, fileData, retentionPolicy)
        }

        @Test
        @DisplayName("Should handle data encryption requirements")
        fun testDataEncryptionRequirements() = runTest {
            // Given
            val fileName = "encrypted_test.txt"
            val fileData = "sensitive content".toByteArray()
            val encryptionKey = "encryption_key_123"

            whenever(mockServiceClient.uploadEncryptedFile(any(), any(), any()))
                .thenReturn(CompletableFuture.completedFuture("encrypted_upload"))

            // When
            val result = connector.uploadEncryptedFile(fileName, fileData, encryptionKey)

            // Then
            assertEquals("encrypted_upload", result.get())
            verify(mockServiceClient).uploadEncryptedFile(fileName, fileData, encryptionKey)
        }

        @Test
        @DisplayName("Should validate compliance with file size limits")
        fun testFileSizeLimitCompliance() = runTest {
            // Given
            val maxFileSize = 10 * 1024 * 1024 // 10MB
            val oversizedFile = ByteArray(maxFileSize + 1)
            val fileName = "oversized.dat"

            connector.setMaxFileSize(maxFileSize)

            // When & Then
            assertThrows<FileSizeExceededException> {
                connector.uploadFile(fileName, oversizedFile)
            }
        }

        @Test
        @DisplayName("Should handle GDPR compliance for personal data")
        fun testGDPRComplianceForPersonalData() = runTest {
            // Given
            val fileName = "personal_data.txt"
            val personalData = "John Doe, john@example.com, 1234567890".toByteArray()
            val gdprMetadata = GDPRMetadata(containsPersonalData = true, dataSubject = "john@example.com")

            whenever(mockServiceClient.uploadFileWithGDPRMetadata(any(), any(), any()))
                .thenReturn(CompletableFuture.completedFuture("gdpr_upload"))

            // When
            val result = connector.uploadFileWithGDPRMetadata(fileName, personalData, gdprMetadata)

            // Then
            assertEquals("gdpr_upload", result.get())
            verify(mockServiceClient).uploadFileWithGDPRMetadata(fileName, personalData, gdprMetadata)
        }
    }

// Additional data classes and interfaces for comprehensive testing
interface HealthCheckListener {
    fun onHealthCheckCompleted(isHealthy: Boolean)
}

interface AlertListener {
    fun onErrorThresholdExceeded(errorRate: Double)
}

interface AuditLogger {
    fun logFileOperation(
        operation: String,
        fileName: String,
        fileSize: Long,
        timestamp: Long,
        userId: String
    )
}

data class ConnectionMetrics(
    val totalConnections: Int,
    val successfulConnections: Int,
    val failedConnections: Int,
    val averageConnectionTime: Long
)

data class OperationMetrics(
    val totalOperations: Int,
    val successfulOperations: Int,
    val failedOperations: Int,
    val averageOperationTime: Long
)

data class MemoryUsage(
    val totalMemory: Long,
    val usedMemory: Long,
    val freeMemory: Long
)

data class RetentionPolicy(
    val days: Int
)

data class GDPRMetadata(
    val containsPersonalData: Boolean,
    val dataSubject: String
)

class FileSizeExceededException(message: String) : Exception(message)
>>>>>>> pr458merge

    @Nested
    @DisplayName("Advanced Input Validation and Boundary Tests")
    inner class AdvancedInputValidationTests {
        
        @Test
        @DisplayName("Should handle various invalid file name patterns")
        fun testInvalidFileNamePatterns() = runTest {
            // Given
            val invalidFileNames = listOf(
                "file\u0000.txt", // null character
                "file\t.txt", // tab character
                "file\n.txt", // newline character
                "file\r.txt", // carriage return
                "file|.txt", // pipe character
                "file<.txt", // less than
                "file>.txt", // greater than
                "file:.txt", // colon (Windows invalid)
                "file?.txt", // question mark
                "file*.txt", // asterisk
                "file\\.txt", // backslash
                "/file.txt", // leading slash
                "file/.txt", // embedded slash
                "CON.txt", // Windows reserved name
                "PRN.txt", // Windows reserved name
                "AUX.txt", // Windows reserved name
                "NUL.txt", // Windows reserved name
                "COM1.txt", // Windows reserved name
                "LPT1.txt" // Windows reserved name
            )
            val fileData = "content".toByteArray()
            
            // When & Then
            invalidFileNames.forEach { fileName ->
                assertThrows<IllegalArgumentException>("Failed for filename: $fileName") {
                    connector.uploadFile(fileName, fileData)
                }
            }
        }
        
        @Test
        @DisplayName("Should handle extreme file content scenarios")
        fun testExtremeFileContentScenarios() = runTest {
            // Given
            val fileName = "extreme_content.txt"
            val extremeContents = listOf(
                ByteArray(0), // empty
                ByteArray(1) { 0x00 }, // single null byte
                ByteArray(1) { 0xFF.toByte() }, // single max byte
                ByteArray(1024) { 0x00 }, // 1KB of nulls
                ByteArray(1024) { 0xFF.toByte() }, // 1KB of max bytes
                ByteArray(1024) { it.toByte() }, // repeating pattern
                "😀🚀🎉💯🔥".toByteArray(Charsets.UTF_8), // emoji content
                "\u0001\u0002\u0003\u0004\u0005".toByteArray(), // control characters
                "A".repeat(1024 * 1024).toByteArray() // 1MB of same character
            )
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("extreme_upload"))
            
            // When & Then
            extremeContents.forEach { content ->
                if (content.isNotEmpty()) {
                    val result = connector.uploadFile(fileName, content)
                    assertEquals("extreme_upload", result.get())
                } else {
                    assertThrows<IllegalArgumentException> {
                        connector.uploadFile(fileName, content)
                    }
                }
            }
        }
        
        @Test
        @DisplayName("Should validate file ID format constraints")
        fun testFileIdFormatValidation() = runTest {
            // Given
            val invalidFileIds = listOf(
                "", // empty
                " ", // whitespace
                "\t", // tab
                "\n", // newline
                "id with spaces",
                "id/with/slashes",
                "id\\with\\backslashes",
                "id:with:colons",
                "id?with?questions",
                "id*with*asterisks",
                "id|with|pipes",
                "id<with<less",
                "id>with>greater",
                "very_".repeat(100) + "long_id", // extremely long ID
                "\u0000null_char", // null character
                "unicode_🚀_id" // unicode characters
            )
            
            // When & Then
            invalidFileIds.forEach { fileId ->
                assertThrows<IllegalArgumentException>("Failed for fileId: '$fileId'") {
                    connector.downloadFile(fileId)
                }
                
                assertThrows<IllegalArgumentException>("Failed for fileId: '$fileId'") {
                    connector.deleteFile(fileId)
                }
            }
        }
        
        @Test
        @DisplayName("Should handle numeric boundary values for configuration")
        fun testNumericBoundaryValues() = runTest {
            // Given
            val boundaryValues = listOf(
                Long.MIN_VALUE,
                -1L,
                0L,
                1L,
                Long.MAX_VALUE
            )
            
            // When & Then
            boundaryValues.forEach { value ->
                if (value <= 0) {
                    assertThrows<IllegalArgumentException> {
                        connector.setConnectionTimeout(value)
                    }
                    assertThrows<IllegalArgumentException> {
                        connector.setRetryDelay(value)
                    }
                } else {
                    assertDoesNotThrow {
                        connector.setConnectionTimeout(value)
                        connector.setRetryDelay(value)
                    }
                }
            }
        }
        
        @Test
        @DisplayName("Should handle integer overflow scenarios")
        fun testIntegerOverflowScenarios() = runTest {
            // Given
            val overflowValues = listOf(
                Int.MAX_VALUE,
                Int.MAX_VALUE + 1L,
                Long.MAX_VALUE
            )
            
            // When & Then
            overflowValues.forEach { value ->
                if (value > Int.MAX_VALUE) {
                    assertThrows<IllegalArgumentException> {
                        connector.setMaxRetryAttempts(value.toInt())
                    }
                } else {
                    assertDoesNotThrow {
                        connector.setMaxRetryAttempts(value.toInt())
                    }
                }
            }
        }
    }
    
    @Nested
    @DisplayName("Advanced Protocol and Network Tests")
    inner class AdvancedProtocolNetworkTests {
        
        @Test
        @DisplayName("Should handle HTTP status code edge cases")
        fun testHttpStatusCodeEdgeCases() = runTest {
            // Given
            val fileName = "status_test.txt"
            val fileData = "content".toByteArray()
            val statusCodes = mapOf(
                100 to "Continue",
                102 to "Processing",
                103 to "Early Hints",
                200 to "OK",
                201 to "Created",
                202 to "Accepted",
                204 to "No Content",
                206 to "Partial Content",
                300 to "Multiple Choices",
                301 to "Moved Permanently",
                302 to "Found",
                304 to "Not Modified",
                307 to "Temporary Redirect",
                308 to "Permanent Redirect",
                400 to "Bad Request",
                401 to "Unauthorized",
                403 to "Forbidden",
                404 to "Not Found",
                405 to "Method Not Allowed",
                408 to "Request Timeout",
                409 to "Conflict",
                410 to "Gone",
                413 to "Payload Too Large",
                414 to "URI Too Long",
                415 to "Unsupported Media Type",
                429 to "Too Many Requests",
                500 to "Internal Server Error",
                501 to "Not Implemented",
                502 to "Bad Gateway",
                503 to "Service Unavailable",
                504 to "Gateway Timeout",
                507 to "Insufficient Storage",
                511 to "Network Authentication Required"
            )
            
            statusCodes.forEach { (code, message) ->
                whenever(mockServiceClient.uploadFile(any(), any()))
                    .thenReturn(CompletableFuture.failedFuture(
                        HttpStatusException(code, message)))
                
                // When & Then
                assertThrows<HttpStatusException> {
                    connector.uploadFile(fileName, fileData).get()
                }
            }
        }
        
        @Test
        @DisplayName("Should handle network partition scenarios")
        fun testNetworkPartitionScenarios() = runTest {
            // Given
            val fileName = "partition_test.txt"
            val fileData = "content".toByteArray()
            
            whenever(mockConnectionManager.isConnected())
                .thenReturn(true)
                .thenReturn(false) // simulate network partition
                .thenReturn(true) // network restored
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(
                    NetworkPartitionException("Network partition detected")))
                .thenReturn(CompletableFuture.completedFuture("partition_recovery"))
            
            // When
            val result = connector.uploadFileWithNetworkRecovery(fileName, fileData)
            
            // Then
            assertEquals("partition_recovery", result.get())
            verify(mockConnectionManager, atLeast(2)).isConnected()
        }
        
        @Test
        @DisplayName("Should handle DNS resolution failures")
        fun testDnsResolutionFailures() = runTest {
            // Given
            val credentials = Credentials("token", "https://non-existent-domain.invalid")
            
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any()))
                .thenThrow(DnsResolutionException("Unable to resolve hostname"))
            
            // When & Then
            assertThrows<DnsResolutionException> {
                connector.connect()
            }
        }
        
        @Test
        @DisplayName("Should handle SSL/TLS handshake failures")
        fun testSslTlsHandshakeFailures() = runTest {
            // Given
            val credentials = Credentials("token", "https://invalid-cert.badssl.com")
            
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any()))
                .thenThrow(SslHandshakeException("SSL handshake failed"))
            
            // When & Then
            assertThrows<SslHandshakeException> {
                connector.connect()
            }
        }
        
        @Test
        @DisplayName("Should handle connection pool exhaustion with backpressure")
        fun testConnectionPoolExhaustionWithBackpressure() = runTest {
            // Given
            val fileName = "backpressure_test.txt"
            val fileData = "content".toByteArray()
            
            whenever(mockConnectionManager.isPoolExhausted()).thenReturn(true)
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(
                    ConnectionPoolExhaustedException("Pool exhausted")))
                .thenReturn(CompletableFuture.completedFuture("backpressure_success"))
            
            // When
            val result = connector.uploadFileWithBackpressure(fileName, fileData)
            
            // Then
            assertEquals("backpressure_success", result.get())
            verify(mockConnectionManager).waitForAvailableConnection()
        }
    }
    
    @Nested
    @DisplayName("Advanced Data Integrity and Consistency Tests")
    inner class AdvancedDataIntegrityTests {
        
        @Test
        @DisplayName("Should detect and handle data corruption during transfer")
        fun testDataCorruptionDetection() = runTest {
            // Given
            val fileName = "corruption_test.txt"
            val originalData = "original content".toByteArray()
            val corruptedData = "corrupted content".toByteArray()
            val originalChecksum = "original_checksum"
            val corruptedChecksum = "corrupted_checksum"
            
            whenever(mockServiceClient.uploadFileWithChecksum(any(), any(), any()))
                .thenReturn(CompletableFuture.completedFuture(
                    UploadResult("upload_123", corruptedChecksum)))
            
            // When & Then
            assertThrows<DataIntegrityException> {
                connector.uploadFileWithIntegrityCheck(fileName, originalData, originalChecksum).get()
            }
        }
        
        @Test
        @DisplayName("Should handle checksum algorithm variations")
        fun testChecksumAlgorithmVariations() = runTest {
            // Given
            val fileName = "checksum_algo.txt"
            val fileData = "content".toByteArray()
            val algorithms = listOf("MD5", "SHA1", "SHA256", "SHA512", "CRC32")
            
            algorithms.forEach { algorithm ->
                val checksum = "${algorithm.lowercase()}_checksum_123"
                
                whenever(mockServiceClient.uploadFileWithChecksumAlgorithm(any(), any(), any(), eq(algorithm)))
                    .thenReturn(CompletableFuture.completedFuture(
                        UploadResult("upload_$algorithm", checksum)))
                
                // When
                val result = connector.uploadFileWithChecksumAlgorithm(fileName, fileData, checksum, algorithm)
                
                // Then
                assertEquals("upload_$algorithm", result.get().uploadId)
                assertEquals(checksum, result.get().checksum)
            }
        }
        
        @Test
        @DisplayName("Should handle partial download recovery")
        fun testPartialDownloadRecovery() = runTest {
            // Given
            val fileId = "partial_download_test"
            val fullData = "This is the complete file content".toByteArray()
            val partialData = "This is the partial".toByteArray()
            
            whenever(mockServiceClient.downloadFile(any()))
                .thenReturn(CompletableFuture.failedFuture(
                    PartialTransferException("Partial transfer", partialData.size)))
            whenever(mockServiceClient.downloadFileRange(any(), eq(partialData.size), any()))
                .thenReturn(CompletableFuture.completedFuture(
                    fullData.sliceArray(partialData.size until fullData.size)))
            
            // When
            val result = connector.downloadFileWithRecovery(fileId)
            
            // Then
            assertArrayEquals(fullData, result.get())
            verify(mockServiceClient).downloadFile(fileId)
            verify(mockServiceClient).downloadFileRange(fileId, partialData.size, fullData.size)
        }
        
        @Test
        @DisplayName("Should handle concurrent modifications during upload")
        fun testConcurrentModificationsDuringUpload() = runTest {
            // Given
            val fileName = "concurrent_mod.txt"
            val originalData = "original".toByteArray()
            val modifiedData = "modified".toByteArray()
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(
                    ConcurrentModificationException("File modified during upload")))
            
            // When & Then
            assertThrows<ConcurrentModificationException> {
                connector.uploadFile(fileName, originalData).get()
            }
        }
        
        @Test
        @DisplayName("Should validate file versioning consistency")
        fun testFileVersioningConsistency() = runTest {
            // Given
            val fileName = "versioned_file.txt"
            val fileData = "version 1 content".toByteArray()
            val version1 = "v1.0"
            val version2 = "v1.1"
            
            whenever(mockServiceClient.uploadFileWithVersion(any(), any(), eq(version1)))
                .thenReturn(CompletableFuture.completedFuture(
                    VersionedUploadResult("upload_v1", version1)))
            whenever(mockServiceClient.uploadFileWithVersion(any(), any(), eq(version2)))
                .thenReturn(CompletableFuture.completedFuture(
                    VersionedUploadResult("upload_v2", version2)))
            
            // When
            val result1 = connector.uploadFileWithVersion(fileName, fileData, version1)
            val result2 = connector.uploadFileWithVersion(fileName, fileData, version2)
            
            // Then
            assertEquals(version1, result1.get().version)
            assertEquals(version2, result2.get().version)
            verify(mockServiceClient).uploadFileWithVersion(fileName, fileData, version1)
            verify(mockServiceClient).uploadFileWithVersion(fileName, fileData, version2)
        }
    }
    
    @Nested
    @DisplayName("Advanced Performance Benchmarking Tests")
    inner class AdvancedPerformanceBenchmarkingTests {
        
        @Test
        @DisplayName("Should maintain performance under sustained load")
        fun testSustainedLoadPerformance() = runTest {
            // Given
            val fileName = "sustained_load.txt"
            val fileData = "load test content".toByteArray()
            val operationsCount = 1000
            val maxTimePerOperation = 100L // milliseconds
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("load_upload"))
            
            // When
            val startTime = System.currentTimeMillis()
            val results = (1..operationsCount).map { i ->
                connector.uploadFile("${fileName}_$i", fileData)
            }
            results.forEach { it.get() }
            val endTime = System.currentTimeMillis()
            
            // Then
            val totalTime = endTime - startTime
            val averageTimePerOperation = totalTime / operationsCount
            assertTrue(averageTimePerOperation <= maxTimePerOperation,
                "Average time per operation ($averageTimePerOperation ms) exceeded threshold ($maxTimePerOperation ms)")
            
            verify(mockServiceClient, times(operationsCount)).uploadFile(any(), any())
        }
        
        @Test
        @DisplayName("Should handle memory pressure gracefully")
        fun testMemoryPressureHandling() = runTest {
            // Given
            val largeFileData = ByteArray(10 * 1024 * 1024) // 10MB
            val fileName = "memory_pressure.dat"
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("memory_upload"))
            
            // When
            val initialMemory = Runtime.getRuntime().freeMemory()
            repeat(50) {
                connector.uploadFile("${fileName}_$it", largeFileData)
                // Force garbage collection periodically
                if (it % 10 == 0) System.gc()
            }
            val finalMemory = Runtime.getRuntime().freeMemory()
            
            // Then
            val memoryDifference = initialMemory - finalMemory
            assertTrue(memoryDifference < 500 * 1024 * 1024, // Less than 500MB difference
                "Memory usage increased by ${memoryDifference / (1024 * 1024)}MB")
        }
        
        @Test
        @DisplayName("Should optimize connection reuse efficiency")
        fun testConnectionReuseOptimization() = runTest {
            // Given
            val credentials = Credentials("token", "endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any())).thenReturn(true)
            whenever(mockConnectionManager.isConnected()).thenReturn(true)
            whenever(mockConnectionManager.getConnectionPoolSize()).thenReturn(5)
            
            val fileName = "reuse_test.txt"
            val fileData = "content".toByteArray()
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("reuse_upload"))
            
            // When
            connector.connect()
            repeat(100) {
                connector.uploadFile("${fileName}_$it", fileData)
            }
            
            // Then
            verify(mockConnectionManager, times(1)).connect(credentials)
            verify(mockServiceClient, times(100)).uploadFile(any(), any())
            assertTrue(connector.getConnectionReuseRatio() > 0.95) // 95% reuse rate
        }
        
        @Test
        @DisplayName("Should handle burst traffic patterns")
        fun testBurstTrafficHandling() = runTest {
            // Given
            val fileName = "burst_test.txt"
            val fileData = "burst content".toByteArray()
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("burst_upload"))
            
            // When - Simulate burst: 100 operations in 1 second, then pause, then another burst
            val burst1 = (1..100).map { i ->
                async { connector.uploadFile("burst1_${fileName}_$i", fileData) }
            }
            burst1.awaitAll()
            
            delay(2000) // 2 second pause
            
            val burst2 = (1..100).map { i ->
                async { connector.uploadFile("burst2_${fileName}_$i", fileData) }
            }
            burst2.awaitAll()
            
            // Then
            verify(mockServiceClient, times(200)).uploadFile(any(), any())
            assertTrue(connector.getThroughputMetrics().peakOperationsPerSecond > 50)
        }
        
        @Test
        @DisplayName("Should validate latency percentiles under load")
        fun testLatencyPercentilesUnderLoad() = runTest {
            // Given
            val fileName = "latency_test.txt"
            val fileData = "latency content".toByteArray()
            val latencies = mutableListOf<Long>()
            
            whenever(mockServiceClient.uploadFile(any(), any())).thenAnswer {
                val delay = (10..100).random().toLong() // Random delay 10-100ms
                CompletableFuture.supplyAsync {
                    Thread.sleep(delay)
                    "latency_upload"
                }
            }
            
            // When
            repeat(100) { i ->
                val startTime = System.currentTimeMillis()
                connector.uploadFile("${fileName}_$i", fileData).get()
                val endTime = System.currentTimeMillis()
                latencies.add(endTime - startTime)
            }
            
            // Then
            latencies.sort()
            val p50 = latencies[49] // 50th percentile
            val p95 = latencies[94] // 95th percentile
            val p99 = latencies[98] // 99th percentile
            
            assertTrue(p50 <= 150, "P50 latency ($p50 ms) exceeded threshold")
            assertTrue(p95 <= 200, "P95 latency ($p95 ms) exceeded threshold")
            assertTrue(p99 <= 250, "P99 latency ($p99 ms) exceeded threshold")
        }
    }
    
    @Nested
    @DisplayName("Advanced Security and Cryptography Tests")
    inner class AdvancedSecurityCryptographyTests {
        
        @Test
        @DisplayName("Should handle various encryption algorithms")
        fun testVariousEncryptionAlgorithms() = runTest {
            // Given
            val fileName = "encrypted_multi_algo.txt"
            val fileData = "sensitive content".toByteArray()
            val algorithms = listOf("AES-256-GCM", "AES-192-CBC", "AES-128-CTR", "ChaCha20-Poly1305")
            
            algorithms.forEach { algorithm ->
                val encryptionKey = "${algorithm}_key_123"
                
                whenever(mockServiceClient.uploadEncryptedFileWithAlgorithm(any(), any(), any(), eq(algorithm)))
                    .thenReturn(CompletableFuture.completedFuture("encrypted_${algorithm}_upload"))
                
                // When
                val result = connector.uploadEncryptedFileWithAlgorithm(fileName, fileData, encryptionKey, algorithm)
                
                // Then
                assertEquals("encrypted_${algorithm}_upload", result.get())
                verify(mockServiceClient).uploadEncryptedFileWithAlgorithm(fileName, fileData, encryptionKey, algorithm)
            }
        }
        
        @Test
        @DisplayName("Should handle key rotation scenarios")
        fun testKeyRotationScenarios() = runTest {
            // Given
            val fileName = "key_rotation.txt"
            val fileData = "rotating key content".toByteArray()
            val oldKey = "old_key_123"
            val newKey = "new_key_456"
            
            whenever(mockServiceClient.uploadEncryptedFile(any(), any(), eq(oldKey)))
                .thenReturn(CompletableFuture.failedFuture(
                    KeyExpiredException("Encryption key expired")))
            whenever(mockServiceClient.uploadEncryptedFile(any(), any(), eq(newKey)))
                .thenReturn(CompletableFuture.completedFuture("rotated_upload"))
            
            whenever(mockAuthProvider.rotateEncryptionKey()).thenReturn(newKey)
            
            // When
            val result = connector.uploadEncryptedFileWithKeyRotation(fileName, fileData, oldKey)
            
            // Then
            assertEquals("rotated_upload", result.get())
            verify(mockAuthProvider).rotateEncryptionKey()
            verify(mockServiceClient).uploadEncryptedFile(fileName, fileData, oldKey)
            verify(mockServiceClient).uploadEncryptedFile(fileName, fileData, newKey)
        }
        
        @Test
        @DisplayName("Should validate digital signatures")
        fun testDigitalSignatureValidation() = runTest {
            // Given
            val fileName = "signed_file.txt"
            val fileData = "signed content".toByteArray()
            val signature = "digital_signature_abc123"
            val publicKey = "public_key_def456"
            
            whenever(mockServiceClient.uploadSignedFile(any(), any(), any(), any()))
                .thenReturn(CompletableFuture.completedFuture("signed_upload"))
            whenever(mockAuthProvider.validateSignature(any(), any(), any())).thenReturn(true)
            
            // When
            val result = connector.uploadSignedFile(fileName, fileData, signature, publicKey)
            
            // Then
            assertEquals("signed_upload", result.get())
            verify(mockAuthProvider).validateSignature(fileData, signature, publicKey)
            verify(mockServiceClient).uploadSignedFile(fileName, fileData, signature, publicKey)
        }
        
        @Test
        @DisplayName("Should handle invalid digital signatures")
        fun testInvalidDigitalSignatures() = runTest {
            // Given
            val fileName = "invalid_signed_file.txt"
            val fileData = "content".toByteArray()
            val invalidSignature = "invalid_signature"
            val publicKey = "public_key"
            
            whenever(mockAuthProvider.validateSignature(any(), any(), any())).thenReturn(false)
            
            // When & Then
            assertThrows<SignatureValidationException> {
                connector.uploadSignedFile(fileName, fileData, invalidSignature, publicKey)
            }
        }
        
        @Test
        @DisplayName("Should handle secure key derivation")
        fun testSecureKeyDerivation() = runTest {
            // Given
            val password = "user_password_123"
            val salt = "random_salt_456"
            val iterations = 100000
            val keyLength = 256
            
            whenever(mockAuthProvider.deriveKey(any(), any(), eq(iterations), eq(keyLength)))
                .thenReturn("derived_key_abc123def456")
            
            // When
            val derivedKey = connector.deriveEncryptionKey(password, salt, iterations, keyLength)
            
            // Then
            assertEquals("derived_key_abc123def456", derivedKey)
            verify(mockAuthProvider).deriveKey(password, salt, iterations, keyLength)
        }
        
        @Test
        @DisplayName("Should handle zero-knowledge proof verification")
        fun testZeroKnowledgeProofVerification() = runTest {
            // Given
            val fileName = "zkp_file.txt"
            val fileData = "zero knowledge content".toByteArray()
            val proof = "zero_knowledge_proof_xyz789"
            val verificationKey = "verification_key_uvw123"
            
            whenever(mockServiceClient.uploadFileWithZKProof(any(), any(), any(), any()))
                .thenReturn(CompletableFuture.completedFuture("zkp_upload"))
            whenever(mockAuthProvider.verifyZeroKnowledgeProof(any(), any())).thenReturn(true)
            
            // When
            val result = connector.uploadFileWithZeroKnowledgeProof(fileName, fileData, proof, verificationKey)
            
            // Then
            assertEquals("zkp_upload", result.get())
            verify(mockAuthProvider).verifyZeroKnowledgeProof(proof, verificationKey)
            verify(mockServiceClient).uploadFileWithZKProof(fileName, fileData, proof, verificationKey)
        }
    }

// Additional exception classes for comprehensive testing
class HttpStatusException(val statusCode: Int, message: String) : Exception(message)
class NetworkPartitionException(message: String) : Exception(message)
class DnsResolutionException(message: String) : Exception(message)
class SslHandshakeException(message: String) : Exception(message)
class DataIntegrityException(message: String) : Exception(message)
class PartialTransferException(message: String, val bytesTransferred: Int) : Exception(message)
class ConcurrentModificationException(message: String) : Exception(message)
class KeyExpiredException(message: String) : Exception(message)
class SignatureValidationException(message: String) : Exception(message)

// Additional data classes for testing
data class VersionedUploadResult(val uploadId: String, val version: String)
data class ThroughputMetrics(val operationsPerSecond: Double, val peakOperationsPerSecond: Double)

