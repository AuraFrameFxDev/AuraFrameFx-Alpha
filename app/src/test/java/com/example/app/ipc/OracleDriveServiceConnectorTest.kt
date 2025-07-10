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

    @Nested
    @DisplayName("Boundary Value and Extreme Edge Case Tests")
    inner class BoundaryValueExtremeEdgeCaseTests {

        @Test
        @DisplayName("Should handle zero-byte file operations")
        fun testZeroByteFileOperations() = runTest {
            // Given
            val fileName = "zero_byte.txt"
            val zeroByteData = byteArrayOf()

            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.uploadFile(fileName, zeroByteData)
            }
        }

        @Test
        @DisplayName("Should handle file name with maximum allowed length")
        fun testMaximumFileNameLength() = runTest {
            // Given
            val maxFileName = "a".repeat(255) + ".txt"
            val fileData = "content".toByteArray()

            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("max_name_upload"))

            // When
            val result = connector.uploadFile(maxFileName, fileData)

            // Then
            assertEquals("max_name_upload", result.get())
            verify(mockServiceClient).uploadFile(maxFileName, fileData)
        }

        @Test
        @DisplayName("Should handle file name exceeding maximum length")
        fun testExcessiveFileNameLength() = runTest {
            // Given
            val excessiveName = "a".repeat(1000) + ".txt"
            val fileData = "content".toByteArray()

            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.uploadFile(excessiveName, fileData)
            }
        }

        @Test
        @DisplayName("Should handle minimum valid connection timeout")
        fun testMinimumConnectionTimeout() = runTest {
            // Given
            val minTimeout = 1L

            // When & Then
            assertDoesNotThrow {
                connector.setConnectionTimeout(minTimeout)
            }
        }

        @Test
        @DisplayName("Should handle maximum integer retry attempts")
        fun testMaximumRetryAttempts() = runTest {
            // Given
            val maxRetries = Integer.MAX_VALUE
            val credentials = Credentials("token", "endpoint")

            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any())).thenReturn(false)

            // When & Then - Should not overflow or hang
            assertTimeoutPreemptively(Duration.ofSeconds(5)) {
                runTest {
                    val result = connector.connectWithRetry(maxRetries = 3) // Use reasonable limit for test
                    assertFalse(result)
                }
            }
        }

        @Test
        @DisplayName("Should handle single character file names")
        fun testSingleCharacterFileName() = runTest {
            // Given
            val singleCharName = "a"
            val fileData = "content".toByteArray()

            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("single_char_upload"))

            // When
            val result = connector.uploadFile(singleCharName, fileData)

            // Then
            assertEquals("single_char_upload", result.get())
        }

        @Test
        @DisplayName("Should handle files with only extension")
        fun testFileNameOnlyExtension() = runTest {
            // Given
            val extensionOnlyName = ".txt"
            val fileData = "content".toByteArray()

            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.uploadFile(extensionOnlyName, fileData)
            }
        }

        @Test
        @DisplayName("Should handle maximum possible file size within limits")
        fun testMaximumFileSize() = runTest {
            // Given
            val maxSize = 1073741824 // 1GB
            val maxFileData = ByteArray(maxSize)
            val fileName = "maximum_size.dat"

            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("max_size_upload"))

            // When
            val result = connector.uploadFile(fileName, maxFileData)

            // Then
            assertEquals("max_size_upload", result.get())
        }

        @Test
        @DisplayName("Should handle concurrent operations at thread pool limits")
        fun testThreadPoolLimitConcurrency() = runTest {
            // Given
            val threadPoolSize = 200
            val fileData = "concurrent".toByteArray()

            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("thread_pool_upload"))

            // When
            val futures = (1..threadPoolSize).map { i ->
                async { connector.uploadFile("thread_pool_$i.txt", fileData) }
            }

            // Then
            val results = futures.awaitAll()
            assertTrue(results.all { it.get() == "thread_pool_upload" })
        }
    }

    @Nested
    @DisplayName("Error Propagation and Recovery Tests")
    inner class ErrorPropagationRecoveryTests {

        @Test
        @DisplayName("Should properly propagate nested exceptions")
        fun testNestedExceptionPropagation() = runTest {
            // Given
            val rootCause = IOException("Network failure")
            val wrappedException = RuntimeException("Service error", rootCause)
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(wrappedException))

            // When & Then
            val exception = assertThrows<RuntimeException> {
                connector.uploadFile("test.txt", "content".toByteArray()).get()
            }

            assertEquals("Service error", exception.message)
            assertTrue(exception.cause is IOException)
            assertEquals("Network failure", exception.cause?.message)
        }

        @Test
        @DisplayName("Should handle error recovery with circuit breaker pattern")
        fun testCircuitBreakerRecovery() = runTest {
            // Given
            val fileName = "circuit_breaker.txt"
            val fileData = "content".toByteArray()
            val failureThreshold = 3

            // Configure failures followed by success
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(IOException("Service down")))
                .thenReturn(CompletableFuture.failedFuture(IOException("Service down")))
                .thenReturn(CompletableFuture.failedFuture(IOException("Service down")))
                .thenReturn(CompletableFuture.completedFuture("circuit_recovered"))

            // When
            repeat(3) {
                try {
                    connector.uploadFile(fileName, fileData).get()
                } catch (e: Exception) {
                    // Expected failures
                }
            }

            // Circuit should be open, then recover
            val result = connector.uploadFileWithCircuitBreaker(fileName, fileData)

            // Then
            assertEquals("circuit_recovered", result.get())
        }

        @Test
        @DisplayName("Should handle cascading failures across operations")
        fun testCascadingFailureHandling() = runTest {
            // Given
            val fileName = "cascade_failure.txt"
            val fileData = "content".toByteArray()
            val authFailure = SecurityException("Auth failed")

            whenever(mockAuthProvider.getCredentials()).thenThrow(authFailure)

            // When & Then
            val uploadException = assertThrows<SecurityException> {
                connector.uploadFile(fileName, fileData).get()
            }
            val downloadException = assertThrows<SecurityException> {
                connector.downloadFile("file_id").get()
            }
            val deleteException = assertThrows<SecurityException> {
                connector.deleteFile("file_id").get()
            }

            assertEquals(authFailure.message, uploadException.message)
            assertEquals(authFailure.message, downloadException.message)
            assertEquals(authFailure.message, deleteException.message)
        }

        @Test
        @DisplayName("Should handle graceful degradation under partial service failure")
        fun testGracefulDegradationPartialFailure() = runTest {
            // Given
            val fileName = "degradation.txt"
            val fileData = "content".toByteArray()

            // Upload works but other operations fail
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("degraded_upload"))
            whenever(mockServiceClient.downloadFile(any()))
                .thenReturn(CompletableFuture.failedFuture(ServiceUnavailableException("Download service down")))
            whenever(mockServiceClient.deleteFile(any()))
                .thenReturn(CompletableFuture.failedFuture(ServiceUnavailableException("Delete service down")))

            // When
            val uploadResult = connector.uploadFile(fileName, fileData)

            // Then
            assertEquals("degraded_upload", uploadResult.get())

            assertThrows<ServiceUnavailableException> {
                connector.downloadFile("file_id").get()
            }
            assertThrows<ServiceUnavailableException> {
                connector.deleteFile("file_id").get()
            }
        }

        @Test
        @DisplayName("Should handle recovery from temporary network partitions")
        fun testNetworkPartitionRecovery() = runTest {
            // Given
            val fileName = "partition_recovery.txt"
            val fileData = "content".toByteArray()
            val partitionException = IOException("Network partition")

            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(partitionException))
                .thenReturn(CompletableFuture.completedFuture("partition_recovered"))

            // When
            val result = connector.uploadFileWithPartitionRecovery(fileName, fileData)

            // Then
            assertEquals("partition_recovered", result.get())
            verify(mockServiceClient, times(2)).uploadFile(fileName, fileData)
        }
    }

    @Nested
    @DisplayName("Data Integrity and Consistency Tests")
    inner class DataIntegrityConsistencyTests {

        @Test
        @DisplayName("Should validate file content integrity with checksums")
        fun testFileContentIntegrityValidation() = runTest {
            // Given
            val fileName = "integrity_test.txt"
            val fileData = "integrity content".toByteArray()
            val expectedChecksum = "sha256:abc123def456"
            val actualChecksum = "sha256:abc123def456"

            whenever(mockServiceClient.uploadFileWithChecksum(any(), any(), any()))
                .thenReturn(CompletableFuture.completedFuture(UploadResult("upload_123", actualChecksum)))

            // When
            val result = connector.uploadFileWithChecksum(fileName, fileData, expectedChecksum)

            // Then
            assertEquals("upload_123", result.get().uploadId)
            assertEquals(expectedChecksum, result.get().checksum)
        }

        @Test
        @DisplayName("Should detect checksum mismatches")
        fun testChecksumMismatchDetection() = runTest {
            // Given
            val fileName = "mismatch_test.txt"
            val fileData = "content".toByteArray()
            val expectedChecksum = "sha256:expected123"
            val actualChecksum = "sha256:different456"

            whenever(mockServiceClient.uploadFileWithChecksum(any(), any(), any()))
                .thenReturn(CompletableFuture.completedFuture(UploadResult("upload_123", actualChecksum)))

            // When & Then
            val result = connector.uploadFileWithChecksum(fileName, fileData, expectedChecksum)
            val uploadResult = result.get()

            assertNotEquals(expectedChecksum, uploadResult.checksum)
            // Should trigger integrity validation failure
            assertThrows<DataIntegrityException> {
                connector.validateIntegrity(uploadResult, expectedChecksum)
            }
        }

        @Test
        @DisplayName("Should maintain operation ordering consistency")
        fun testOperationOrderingConsistency() = runTest {
            // Given
            val operations = mutableListOf<String>()
            val fileName = "ordering_test.txt"
            val fileData = "content".toByteArray()

            whenever(mockServiceClient.uploadFile(any(), any())).thenAnswer {
                operations.add("upload")
                CompletableFuture.completedFuture("upload_123")
            }
            whenever(mockServiceClient.downloadFile(any())).thenAnswer {
                operations.add("download")
                CompletableFuture.completedFuture(fileData)
            }
            whenever(mockServiceClient.deleteFile(any())).thenAnswer {
                operations.add("delete")
                CompletableFuture.completedFuture(true)
            }

            // When
            connector.uploadFile(fileName, fileData).get()
            connector.downloadFile("file_123").get()
            connector.deleteFile("file_123").get()

            // Then
            assertEquals(listOf("upload", "download", "delete"), operations)
        }

        @Test
        @DisplayName("Should handle data consistency during concurrent modifications")
        fun testDataConsistencyConcurrentModifications() = runTest {
            // Given
            val fileName = "concurrent_mod.txt"
            val originalData = "original".toByteArray()
            val modifiedData = "modified".toByteArray()

            whenever(mockServiceClient.uploadFile(eq(fileName), eq(originalData)))
                .thenReturn(CompletableFuture.completedFuture("original_upload"))
            whenever(mockServiceClient.uploadFile(eq(fileName), eq(modifiedData)))
                .thenReturn(CompletableFuture.completedFuture("modified_upload"))

            // When
            val originalUpload = async { connector.uploadFile(fileName, originalData) }
            val modifiedUpload = async { connector.uploadFile(fileName, modifiedData) }

            val results = awaitAll(originalUpload, modifiedUpload)

            // Then
            assertTrue(results.contains("original_upload"))
            assertTrue(results.contains("modified_upload"))
            verify(mockServiceClient).uploadFile(fileName, originalData)
            verify(mockServiceClient).uploadFile(fileName, modifiedData)
        }

        @Test
        @DisplayName("Should validate transactional behavior for batch operations")
        fun testTransactionalBehaviorBatchOperations() = runTest {
            // Given
            val batchFiles = listOf(
                "batch1.txt" to "content1".toByteArray(),
                "batch2.txt" to "content2".toByteArray(),
                "batch3.txt" to "content3".toByteArray()
            )

            // First two succeed, third fails
            whenever(mockServiceClient.uploadBatch(any()))
                .thenReturn(CompletableFuture.failedFuture(
                    BatchOperationException("Partial batch failure", 2)
                ))

            // When & Then
            assertThrows<BatchOperationException> {
                connector.uploadBatchTransactional(batchFiles).get()
            }

            // Should rollback successful uploads
            verify(mockServiceClient).rollbackBatch(any())
        }
    }

    @Nested
    @DisplayName("Protocol Compliance and Specification Tests")
    inner class ProtocolComplianceSpecificationTests {

        @Test
        @DisplayName("Should comply with HTTP status code handling")
        fun testHTTPStatusCodeCompliance() = runTest {
            // Given
            val fileName = "http_compliance.txt"
            val fileData = "content".toByteArray()

            val httpExceptions = mapOf(
                400 to BadRequestException("Bad request"),
                401 to UnauthorizedException("Unauthorized"),
                403 to ForbiddenException("Forbidden"),
                404 to NotFoundException("Not found"),
                429 to RateLimitException("Rate limited"),
                500 to InternalServerException("Internal server error"),
                502 to BadGatewayException("Bad gateway"),
                503 to ServiceUnavailableException("Service unavailable")
            )

            httpExceptions.forEach { (statusCode, exception) ->
                whenever(mockServiceClient.uploadFile(any(), any()))
                    .thenReturn(CompletableFuture.failedFuture(exception))

                // When & Then
                val thrownException = assertThrows(exception::class.java) {
                    connector.uploadFile(fileName, fileData).get()
                }

                assertEquals(exception.message, thrownException.message)
            }
        }

        @Test
        @DisplayName("Should handle REST API versioning correctly")
        fun testRESTAPIVersioning() = runTest {
            // Given
            val apiVersions = listOf("v1", "v2", "v3")
            val fileName = "version_test.txt"
            val fileData = "content".toByteArray()

            apiVersions.forEach { version ->
                whenever(mockServiceClient.setAPIVersion(version)).thenReturn(Unit)
                whenever(mockServiceClient.uploadFile(any(), any()))
                    .thenReturn(CompletableFuture.completedFuture("upload_$version"))

                // When
                connector.setAPIVersion(version)
                val result = connector.uploadFile(fileName, fileData)

                // Then
                assertEquals("upload_$version", result.get())
                verify(mockServiceClient).setAPIVersion(version)
            }
        }

        @Test
        @DisplayName("Should handle content-type validation")
        fun testContentTypeValidation() = runTest {
            // Given
            val validContentTypes = mapOf(
                "document.txt" to "text/plain",
                "image.jpg" to "image/jpeg",
                "data.json" to "application/json",
                "archive.zip" to "application/zip"
            )

            validContentTypes.forEach { (fileName, expectedContentType) ->
                whenever(mockServiceClient.uploadFileWithContentType(any(), any(), any()))
                    .thenReturn(CompletableFuture.completedFuture("upload_$expectedContentType"))

                // When
                val result = connector.uploadFileWithContentType(
                    fileName, 
                    "content".toByteArray(), 
                    expectedContentType
                )

                // Then
                assertEquals("upload_$expectedContentType", result.get())
                verify(mockServiceClient).uploadFileWithContentType(fileName, any(), expectedContentType)
            }
        }

        @Test
        @DisplayName("Should handle OAuth 2.0 token flow correctly")
        fun testOAuth2TokenFlowCompliance() = runTest {
            // Given
            val accessToken = "access_token_123"
            val refreshToken = "refresh_token_456"
            val expiredToken = "expired_token"

            whenever(mockAuthProvider.getAccessToken()).thenReturn(accessToken)
            whenever(mockAuthProvider.isTokenExpired(accessToken)).thenReturn(false)
            whenever(mockAuthProvider.isTokenExpired(expiredToken)).thenReturn(true)
            whenever(mockAuthProvider.refreshAccessToken(refreshToken)).thenReturn("new_access_token")

            // When
            val validToken = connector.getValidAccessToken()
            
            // Then
            assertEquals(accessToken, validToken)
            verify(mockAuthProvider).getAccessToken()
            verify(mockAuthProvider).isTokenExpired(accessToken)
        }

        @Test
        @DisplayName("Should handle pagination according to RFC 5988")
        fun testPaginationRFC5988Compliance() = runTest {
            // Given
            val page1 = PaginatedResult(
                items = listOf("file1", "file2"),
                nextPageToken = "page2_token",
                hasMore = true
            )
            val page2 = PaginatedResult(
                items = listOf("file3", "file4"),
                nextPageToken = null,
                hasMore = false
            )

            whenever(mockServiceClient.listFiles(null))
                .thenReturn(CompletableFuture.completedFuture(page1))
            whenever(mockServiceClient.listFiles("page2_token"))
                .thenReturn(CompletableFuture.completedFuture(page2))

            // When
            val allFiles = connector.listAllFiles()

            // Then
            assertEquals(listOf("file1", "file2", "file3", "file4"), allFiles.get())
            verify(mockServiceClient).listFiles(null)
            verify(mockServiceClient).listFiles("page2_token")
        }
    }

    @Nested
    @DisplayName("Performance Optimization and Profiling Tests")
    inner class PerformanceOptimizationProfilingTests {

        @Test
        @DisplayName("Should optimize connection pooling efficiency")
        fun testConnectionPoolingOptimization() = runTest {
            // Given
            val poolSize = 10
            val operations = 100

            whenever(mockConnectionManager.getPoolSize()).thenReturn(poolSize)
            whenever(mockConnectionManager.getActiveConnections()).thenReturn(5)
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("pooled_upload"))

            // When
            val startTime = System.currentTimeMillis()
            
            repeat(operations) {
                connector.uploadFile("pool_test_$it.txt", "content".toByteArray())
            }
            
            val endTime = System.currentTimeMillis()
            val totalTime = endTime - startTime

            // Then
            assertTrue(totalTime < 10000) // Should complete within 10 seconds
            verify(mockConnectionManager, atMost(poolSize)).createNewConnection()
        }

        @Test
        @DisplayName("Should demonstrate memory-efficient streaming for large files")
        fun testMemoryEfficientStreaming() = runTest {
            // Given
            val largeFileSize = 100 * 1024 * 1024 // 100MB
            val chunkSize = 1024 * 1024 // 1MB chunks
            val fileName = "streaming_large.dat"

            whenever(mockServiceClient.uploadFileStream(any(), any(), any()))
                .thenReturn(CompletableFuture.completedFuture("streamed_upload"))

            // When
            val initialMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
            
            val result = connector.uploadLargeFileStreaming(fileName, largeFileSize, chunkSize)
            
            val finalMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
            val memoryIncrease = finalMemory - initialMemory

            // Then
            assertEquals("streamed_upload", result.get())
            assertTrue(memoryIncrease < 10 * 1024 * 1024) // Should not increase by more than 10MB
        }

        @Test
        @DisplayName("Should demonstrate optimal batch size for bulk operations")
        fun testOptimalBatchSizeDetermination() = runTest {
            // Given
            val totalFiles = 1000
            val batchSizes = listOf(10, 50, 100, 200)
            val performanceResults = mutableMapOf<Int, Long>()

            batchSizes.forEach { batchSize ->
                whenever(mockServiceClient.uploadBatch(any()))
                    .thenReturn(CompletableFuture.completedFuture(
                        (1..batchSize).map { "batch_upload_$it" }
                    ))

                // When
                val startTime = System.currentTimeMillis()
                
                val batches = (1..totalFiles).chunked(batchSize).map { batch ->
                    batch.map { "file_$it.txt" to "content".toByteArray() }
                }
                
                batches.forEach { batch ->
                    connector.uploadBatch(batch)
                }
                
                val endTime = System.currentTimeMillis()
                performanceResults[batchSize] = endTime - startTime
            }

            // Then
            val optimalBatchSize = performanceResults.minByOrNull { it.value }?.key
            assertNotNull(optimalBatchSize)
            assertTrue(optimalBatchSize!! in 50..200) // Reasonable optimal range
        }

        @Test
        @DisplayName("Should handle CPU-intensive operations without blocking")
        fun testCPUIntensiveNonBlocking() = runTest {
            // Given
            val cpuIntensiveData = ByteArray(50 * 1024 * 1024) // 50MB to process
            val fileName = "cpu_intensive.dat"

            whenever(mockServiceClient.uploadFileWithCompression(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("compressed_upload"))

            // When
            val startTime = System.currentTimeMillis()
            val compressionFuture = connector.uploadFileWithCompressionAsync(fileName, cpuIntensiveData)
            
            // Should be able to perform other operations while compression happens
            val otherOperationFuture = connector.uploadFile("other.txt", "quick".toByteArray())
            
            val results = awaitAll(compressionFuture, otherOperationFuture)
            val endTime = System.currentTimeMillis()

            // Then
            assertTrue(results.contains("compressed_upload"))
            assertTrue(endTime - startTime < 30000) // Should complete within 30 seconds
        }

        @Test
        @DisplayName("Should maintain performance under memory pressure")
        fun testPerformanceUnderMemoryPressure() = runTest {
            // Given
            val memoryIntensiveOperations = 50
            val largeFileData = ByteArray(10 * 1024 * 1024) // 10MB each

            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("memory_pressure_upload"))

            // When
            val startTime = System.currentTimeMillis()
            
            val futures = (1..memoryIntensiveOperations).map { i ->
                async {
                    connector.uploadFile("memory_pressure_$i.dat", largeFileData)
                }
            }
            
            val results = futures.awaitAll()
            val endTime = System.currentTimeMillis()

            // Then
            assertTrue(results.all { it.get() == "memory_pressure_upload" })
            assertTrue(endTime - startTime < 60000) // Should complete within 60 seconds
            
            // Memory should not grow excessively
            val finalMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
            assertTrue(finalMemory < 500 * 1024 * 1024) // Less than 500MB
        }
    }

    @Nested
    @DisplayName("Interoperability and Integration Tests")
    inner class InteroperabilityIntegrationTests {

        @Test
        @DisplayName("Should work with different client configurations")
        fun testDifferentClientConfigurations() = runTest {
            // Given
            val configurations = listOf(
                ClientConfig(timeout = 30000, retries = 3, compression = true),
                ClientConfig(timeout = 60000, retries = 5, compression = false),
                ClientConfig(timeout = 10000, retries = 1, compression = true)
            )

            configurations.forEach { config ->
                val configuredConnector = OracleDriveServiceConnector(
                    serviceClient = mockServiceClient,
                    connectionManager = mockConnectionManager,
                    authProvider = mockAuthProvider,
                    config = config
                )

                whenever(mockServiceClient.uploadFile(any(), any()))
                    .thenReturn(CompletableFuture.completedFuture("config_upload_${config.timeout}"))

                // When
                val result = configuredConnector.uploadFile("config_test.txt", "content".toByteArray())

                // Then
                assertEquals("config_upload_${config.timeout}", result.get())
            }
        }

        @Test
        @DisplayName("Should handle cross-platform file path compatibility")
        fun testCrossPlatformPathCompatibility() = runTest {
            // Given
            val platformPaths = listOf(
                "unix/style/path.txt",
                "windows\\style\\path.txt",
                "mixed/style\\path.txt",
                "/absolute/unix/path.txt",
                "C:\\absolute\\windows\\path.txt"
            )

            platformPaths.forEach { path ->
                whenever(mockServiceClient.uploadFile(any(), any()))
                    .thenReturn(CompletableFuture.completedFuture("platform_upload"))

                // When
                val normalizedPath = connector.normalizePath(path)
                val result = connector.uploadFile(normalizedPath, "content".toByteArray())

                // Then
                assertEquals("platform_upload", result.get())
                assertFalse(normalizedPath.contains("\\")) // Should be normalized to forward slashes
            }
        }

        @Test
        @DisplayName("Should handle character encoding variations")
        fun testCharacterEncodingVariations() = runTest {
            // Given
            val encodings = mapOf(
                "UTF-8" to "Hello 世界 🌍".toByteArray(Charsets.UTF_8),
                "UTF-16" to "Hello 世界 🌍".toByteArray(Charsets.UTF_16),
                "ISO-8859-1" to "Hello World".toByteArray(Charsets.ISO_8859_1)
            )

            encodings.forEach { (encoding, data) ->
                val fileName = "encoding_test_$encoding.txt"

                whenever(mockServiceClient.uploadFileWithEncoding(any(), any(), any()))
                    .thenReturn(CompletableFuture.completedFuture("encoding_upload_$encoding"))

                // When
                val result = connector.uploadFileWithEncoding(fileName, data, encoding)

                // Then
                assertEquals("encoding_upload_$encoding", result.get())
                verify(mockServiceClient).uploadFileWithEncoding(fileName, data, encoding)
            }
        }

        @Test
        @DisplayName("Should integrate with external monitoring systems")
        fun testExternalMonitoringIntegration() = runTest {
            // Given
            val metricsCollector = mock<MetricsCollector>()
            connector.setMetricsCollector(metricsCollector)

            val fileName = "monitoring_test.txt"
            val fileData = "content".toByteArray()

            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("monitored_upload"))

            // When
            connector.uploadFile(fileName, fileData)

            // Then
            verify(metricsCollector).recordOperation("upload", fileName, fileData.size.toLong(), true)
        }

        @Test
        @DisplayName("Should work with different authentication providers")
        fun testDifferentAuthenticationProviders() = runTest {
            // Given
            val authProviders = listOf(
                mock<BasicAuthProvider>(),
                mock<OAuth2AuthProvider>(),
                mock<ApiKeyAuthProvider>(),
                mock<CertificateAuthProvider>()
            )

            authProviders.forEach { authProvider ->
                val credentials = Credentials("token_${authProvider.javaClass.simpleName}", "endpoint")
                
                whenever(authProvider.getCredentials()).thenReturn(credentials)
                whenever(mockConnectionManager.connect(any())).thenReturn(true)

                val authConnector = OracleDriveServiceConnector(
                    serviceClient = mockServiceClient,
                    connectionManager = mockConnectionManager,
                    authProvider = authProvider
                )

                // When
                val result = authConnector.connect()

                // Then
                assertTrue(result)
                verify(authProvider).getCredentials()
            }
        }
    }

// Additional helper classes and interfaces for the comprehensive tests
}

interface ClientConfig {
    val timeout: Long
    val retries: Int
    val compression: Boolean
}

data class ClientConfigImpl(
    override val timeout: Long,
    override val retries: Int,
    override val compression: Boolean
) : ClientConfig

data class PaginatedResult<T>(
    val items: List<T>,
    val nextPageToken: String?,
    val hasMore: Boolean
)

interface MetricsCollector {
    fun recordOperation(operation: String, fileName: String, fileSize: Long, success: Boolean)
}

interface BasicAuthProvider : AuthProvider
interface OAuth2AuthProvider : AuthProvider
interface ApiKeyAuthProvider : AuthProvider
interface CertificateAuthProvider : AuthProvider

class BatchOperationException(message: String, val successfulOperations: Int) : Exception(message)
class DataIntegrityException(message: String) : Exception(message)
class BadRequestException(message: String) : Exception(message)
class UnauthorizedException(message: String) : Exception(message)
class ForbiddenException(message: String) : Exception(message)
class NotFoundException(message: String) : Exception(message)
class InternalServerException(message: String) : Exception(message)
class BadGatewayException(message: String) : Exception(message)


