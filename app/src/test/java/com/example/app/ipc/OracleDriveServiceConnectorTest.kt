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
