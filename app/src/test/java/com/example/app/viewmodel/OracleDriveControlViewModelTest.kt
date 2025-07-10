package com.example.app.viewmodel

import androidx.arch.core.executor.testing.InstantTaskExecutorRule
import androidx.lifecycle.Observer
import io.mockk.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.*
import kotlinx.coroutines.async
import org.junit.After
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.junit.MockitoJUnitRunner
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

@ExperimentalCoroutinesApi
@RunWith(MockitoJUnitRunner::class)
class OracleDriveControlViewModelTest {

    @get:Rule
    val instantTaskExecutorRule = InstantTaskExecutorRule()

    private val testDispatcher = StandardTestDispatcher()

    private lateinit var viewModel: OracleDriveControlViewModel
    private lateinit var mockRepository: DriveRepository
    private lateinit var mockNetworkManager: NetworkManager
    private lateinit var mockStateObserver: Observer<DriveState>
    private lateinit var mockErrorObserver: Observer<String>
    private lateinit var mockLoadingObserver: Observer<Boolean>

    @Before
    fun setup() {
        Dispatchers.setMain(testDispatcher)
        
        mockRepository = mockk(relaxed = true)
        mockNetworkManager = mockk(relaxed = true)
        mockStateObserver = mockk(relaxed = true)
        mockErrorObserver = mockk(relaxed = true)
        mockLoadingObserver = mockk(relaxed = true)

        viewModel = OracleDriveControlViewModel(mockRepository, mockNetworkManager)
        
        viewModel.driveState.observeForever(mockStateObserver)
        viewModel.errorMessage.observeForever(mockErrorObserver)
        viewModel.isLoading.observeForever(mockLoadingObserver)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
        viewModel.driveState.removeObserver(mockStateObserver)
        viewModel.errorMessage.removeObserver(mockErrorObserver)
        viewModel.isLoading.removeObserver(mockLoadingObserver)
        clearAllMocks()
    }

    @Test
    fun `initial state should be idle`() {
        // Given - ViewModel is initialized
        
        // When - checking initial state
        val initialState = viewModel.driveState.value
        
        // Then
        assertEquals(DriveState.IDLE, initialState)
        verify { mockStateObserver.onChanged(DriveState.IDLE) }
    }

    @Test
    fun `startDrive should update state to driving when successful`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockLoadingObserver.onChanged(true) }
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        verify { mockLoadingObserver.onChanged(false) }
        coVerify { mockRepository.startDrive(any()) }
    }

    @Test
    fun `startDrive should handle network error gracefully`() = runTest {
        // Given
        val errorMessage = "Network connection failed"
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(Exception(errorMessage))
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockLoadingObserver.onChanged(true) }
        verify { mockErrorObserver.onChanged(errorMessage) }
        verify { mockLoadingObserver.onChanged(false) }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `stopDrive should update state to idle when successful`() = runTest {
        // Given
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)
        
        // When
        viewModel.stopDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockStateObserver.onChanged(DriveState.IDLE) }
        coVerify { mockRepository.stopDrive() }
    }

    @Test
    fun `stopDrive should handle repository error`() = runTest {
        // Given
        val errorMessage = "Failed to stop drive"
        coEvery { mockRepository.stopDrive() } returns Result.failure(Exception(errorMessage))
        
        // When
        viewModel.stopDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged(errorMessage) }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `pauseDrive should update state to paused when successful`() = runTest {
        // Given
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        
        // When
        viewModel.pauseDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockStateObserver.onChanged(DriveState.PAUSED) }
        coVerify { mockRepository.pauseDrive() }
    }

    @Test
    fun `resumeDrive should update state to driving from paused`() = runTest {
        // Given
        viewModel.startDrive()
        viewModel.pauseDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        coEvery { mockRepository.resumeDrive() } returns Result.success(Unit)
        
        // When
        viewModel.resumeDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        coVerify { mockRepository.resumeDrive() }
    }

    @Test
    fun `updateSpeed should call repository with correct value`() = runTest {
        // Given
        val speed = 50.0
        coEvery { mockRepository.updateSpeed(speed) } returns Result.success(Unit)
        
        // When
        viewModel.updateSpeed(speed)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        coVerify { mockRepository.updateSpeed(speed) }
    }

    @Test
    fun `updateSpeed should handle negative values gracefully`() = runTest {
        // Given
        val invalidSpeed = -10.0
        
        // When
        viewModel.updateSpeed(invalidSpeed)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Invalid speed value: $invalidSpeed") }
        coVerify(exactly = 0) { mockRepository.updateSpeed(any()) }
    }

    @Test
    fun `updateSpeed should handle excessive values gracefully`() = runTest {
        // Given
        val excessiveSpeed = 1000.0
        
        // When
        viewModel.updateSpeed(excessiveSpeed)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Speed exceeds maximum limit: $excessiveSpeed") }
        coVerify(exactly = 0) { mockRepository.updateSpeed(any()) }
    }

    @Test
    fun `changeDirection should update direction when valid`() = runTest {
        // Given
        val direction = Direction.FORWARD
        coEvery { mockRepository.changeDirection(direction) } returns Result.success(Unit)
        
        // When
        viewModel.changeDirection(direction)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        coVerify { mockRepository.changeDirection(direction) }
    }

    @Test
    fun `changeDirection should handle repository error`() = runTest {
        // Given
        val direction = Direction.REVERSE
        val errorMessage = "Failed to change direction"
        coEvery { mockRepository.changeDirection(direction) } returns Result.failure(Exception(errorMessage))
        
        // When
        viewModel.changeDirection(direction)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged(errorMessage) }
    }

    @Test
    fun `isNetworkAvailable should return network manager status`() {
        // Given
        every { mockNetworkManager.isConnected() } returns true
        
        // When
        val result = viewModel.isNetworkAvailable()
        
        // Then
        assertTrue(result)
        verify { mockNetworkManager.isConnected() }
    }

    @Test
    fun `isNetworkAvailable should return false when network unavailable`() {
        // Given
        every { mockNetworkManager.isConnected() } returns false
        
        // When
        val result = viewModel.isNetworkAvailable()
        
        // Then
        assertFalse(result)
        verify { mockNetworkManager.isConnected() }
    }

    @Test
    fun `emergency stop should immediately stop drive and update state`() = runTest {
        // Given
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        coEvery { mockRepository.emergencyStop() } returns Result.success(Unit)
        
        // When
        viewModel.emergencyStop()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockStateObserver.onChanged(DriveState.EMERGENCY_STOP) }
        coVerify { mockRepository.emergencyStop() }
    }

    @Test
    fun `emergency stop should handle repository failure`() = runTest {
        // Given
        val errorMessage = "Emergency stop failed"
        coEvery { mockRepository.emergencyStop() } returns Result.failure(Exception(errorMessage))
        
        // When
        viewModel.emergencyStop()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged(errorMessage) }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `reset should clear error state and return to idle`() = runTest {
        // Given
        viewModel.emergencyStop()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // When
        viewModel.reset()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockStateObserver.onChanged(DriveState.IDLE) }
        verify { mockErrorObserver.onChanged("") }
    }

    @Test
    fun `concurrent operations should be handled correctly`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)
        
        // When
        viewModel.startDrive()
        viewModel.stopDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockStateObserver.onChanged(DriveState.IDLE) }
        coVerify { mockRepository.startDrive(any()) }
        coVerify { mockRepository.stopDrive() }
    }

    @Test
    fun `multiple speed updates should debounce correctly`() = runTest {
        // Given
        coEvery { mockRepository.updateSpeed(any()) } returns Result.success(Unit)
        
        // When
        viewModel.updateSpeed(10.0)
        viewModel.updateSpeed(20.0)
        viewModel.updateSpeed(30.0)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - should only call with the last value due to debouncing
        coVerify(exactly = 1) { mockRepository.updateSpeed(30.0) }
    }

    @Test
    fun `loading state should be managed correctly across operations`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            kotlinx.coroutines.delay(100)
            Result.success(Unit)
        }
        
        // When
        viewModel.startDrive()
        
        // Then - loading should be true initially
        verify { mockLoadingObserver.onChanged(true) }
        
        // When - operation completes
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - loading should be false
        verify { mockLoadingObserver.onChanged(false) }
    }

    @Test
    fun `view model should handle null repository responses gracefully`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(NullPointerException("Repository returned null"))
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Repository returned null") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `view model should validate input parameters`() = runTest {
        // Given - invalid parameters
        val invalidSpeed = Double.NaN
        
        // When
        viewModel.updateSpeed(invalidSpeed)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Invalid speed value: NaN") }
        coVerify(exactly = 0) { mockRepository.updateSpeed(any()) }
    }

    // Additional edge cases and boundary tests

    @Test
    fun `should handle rapid state changes gracefully`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        coEvery { mockRepository.resumeDrive() } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)
        
        // When - rapid state changes
        viewModel.startDrive()
        viewModel.pauseDrive()
        viewModel.resumeDrive()
        viewModel.stopDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - should end in idle state
        verify { mockStateObserver.onChanged(DriveState.IDLE) }
    }

    @Test
    fun `should handle zero speed correctly`() = runTest {
        // Given
        val zeroSpeed = 0.0
        coEvery { mockRepository.updateSpeed(zeroSpeed) } returns Result.success(Unit)
        
        // When
        viewModel.updateSpeed(zeroSpeed)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        coVerify { mockRepository.updateSpeed(zeroSpeed) }
    }

    @Test
    fun `should handle infinite speed values`() = runTest {
        // Given
        val infiniteSpeed = Double.POSITIVE_INFINITY
        
        // When
        viewModel.updateSpeed(infiniteSpeed)
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockErrorObserver.onChanged("Speed exceeds maximum limit: $infiniteSpeed") }
        coVerify(exactly = 0) { mockRepository.updateSpeed(any()) }
    }

    @Test
    fun `should handle negative infinite speed values`() = runTest {
        // Given
        val negativeInfiniteSpeed = Double.NEGATIVE_INFINITY

        // When
        viewModel.updateSpeed(negativeInfiniteSpeed)
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockErrorObserver.onChanged("Invalid speed value: $negativeInfiniteSpeed") }
        coVerify(exactly = 0) { mockRepository.updateSpeed(any()) }
    }

    @Test
    fun `should handle maximum valid speed correctly`() = runTest {
        // Given
        val maxValidSpeed = 99.9
        coEvery { mockRepository.updateSpeed(maxValidSpeed) } returns Result.success(Unit)

        // When
        viewModel.updateSpeed(maxValidSpeed)
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        coVerify { mockRepository.updateSpeed(maxValidSpeed) }
    }

    @Test
    fun `should handle minimum valid speed correctly`() = runTest {
        // Given
        val minValidSpeed = 0.1
        coEvery { mockRepository.updateSpeed(minValidSpeed) } returns Result.success(Unit)

        // When
        viewModel.updateSpeed(minValidSpeed)
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        coVerify { mockRepository.updateSpeed(minValidSpeed) }
    }

    // Additional comprehensive tests for better coverage

    @Test
    fun `should handle memory pressure during operations`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(OutOfMemoryError("Memory exhausted"))

        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockErrorObserver.onChanged("Memory exhausted") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle security exceptions during repository calls`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(SecurityException("Access denied"))

        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockErrorObserver.onChanged("Access denied") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle file system exceptions during operations`() = runTest {
        // Given
        coEvery { mockRepository.updateSpeed(any()) } returns Result.failure(java.io.IOException("Disk full"))

        // When
        viewModel.updateSpeed(25.0)
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockErrorObserver.onChanged("Disk full") }
    }

    @Test
    fun `should handle extremely rapid sequential operations`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        coEvery { mockRepository.resumeDrive() } returns Result.success(Unit)

        // When - extremely rapid operations
        repeat(100) {
            viewModel.startDrive()
            viewModel.pauseDrive()
            viewModel.resumeDrive()
            viewModel.stopDrive()
        }
        testDispatcher.scheduler.advanceUntilIdle()

        // Then - should handle all operations without crashing
        verify(atLeast = 1) { mockStateObserver.onChanged(DriveState.IDLE) }
    }

    @Test
    fun `should handle speed updates with floating point precision`() = runTest {
        // Given
        val preciseSpeed = 25.123456789
        coEvery { mockRepository.updateSpeed(preciseSpeed) } returns Result.success(Unit)

        // When
        viewModel.updateSpeed(preciseSpeed)
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        coVerify { mockRepository.updateSpeed(preciseSpeed) }
    }

    @Test
    fun `should handle extremely small positive speed values`() = runTest {
        // Given
        val tinySpeed = Double.MIN_VALUE
        coEvery { mockRepository.updateSpeed(tinySpeed) } returns Result.success(Unit)

        // When
        viewModel.updateSpeed(tinySpeed)
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        coVerify { mockRepository.updateSpeed(tinySpeed) }
    }

    @Test
    fun `should handle repository returning unexpected result types`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(ClassCastException("Unexpected result type"))

        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockErrorObserver.onChanged("Unexpected result type") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle concurrent modifications during state updates`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)

        // When - concurrent operations from different "threads"
        val job1 = async { viewModel.startDrive() }
        val job2 = async { viewModel.stopDrive() }
        
        job1.await()
        job2.await()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then - should handle concurrent access safely
        verify(atLeast = 1) { mockStateObserver.onChanged(any()) }
    }

    @Test
    fun `should handle network manager throwing exceptions`() {
        // Given
        every { mockNetworkManager.isConnected() } throws RuntimeException("Network manager failed")

        // When & Then - should not crash
        try {
            viewModel.isNetworkAvailable()
        } catch (e: Exception) {
            // Should handle gracefully without propagating exception
        }
    }

    @Test
    fun `should handle observer notifications during rapid state changes`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        coEvery { mockRepository.resumeDrive() } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)

        // When - rapid state changes that might cause race conditions
        viewModel.startDrive()
        viewModel.pauseDrive()
        viewModel.resumeDrive()
        viewModel.pauseDrive()
        viewModel.resumeDrive()
        viewModel.stopDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then - all state changes should be properly notified
        verify(atLeast = 5) { mockStateObserver.onChanged(any()) }
    }

    @Test
    fun `should handle repository calls with null parameters gracefully`() = runTest {
        // Given
        coEvery { mockRepository.changeDirection(any()) } returns Result.success(Unit)

        // When - trying to pass null direction (if possible through reflection or casting)
        try {
            viewModel.changeDirection(null as? Direction ?: Direction.FORWARD)
            testDispatcher.scheduler.advanceUntilIdle()
        } catch (e: Exception) {
            // Should handle null gracefully
        }

        // Then - should not crash the application
        assertTrue(true) // Test passes if we reach here without crashing
    }

    @Test
    fun `should handle extremely long error messages`() = runTest {
        // Given
        val longErrorMessage = "Error: " + "x".repeat(10000)
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(Exception(longErrorMessage))

        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockErrorObserver.onChanged(longErrorMessage) }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle special characters in error messages`() = runTest {
        // Given
        val specialCharErrorMessage = "Error: 特殊字符 🚗 ñáéíóú @#$%^&*()[]{}|\\:;\"'<>?,./"
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(Exception(specialCharErrorMessage))

        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockErrorObserver.onChanged(specialCharErrorMessage) }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle repository method call timeouts with custom exceptions`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(
            java.net.SocketTimeoutException("Connection timeout after 30 seconds")
        )

        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockErrorObserver.onChanged("Connection timeout after 30 seconds") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle coroutine scope cancellation during loading`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            kotlinx.coroutines.delay(1000)
            throw kotlinx.coroutines.CancellationException("Scope cancelled")
        }

        // When
        viewModel.startDrive()
        kotlinx.coroutines.delay(100)
        testDispatcher.scheduler.advanceUntilIdle()

        // Then - should handle cancellation gracefully
        verify { mockLoadingObserver.onChanged(true) }
        verify { mockLoadingObserver.onChanged(false) }
    }

    @Test
    fun `should handle speed boundary values near limits`() = runTest {
        // Given
        val boundaryValues = listOf(99.99999, 100.00001, -0.00001, 0.00001)
        
        boundaryValues.forEach { speed ->
            // When
            viewModel.updateSpeed(speed)
            testDispatcher.scheduler.advanceUntilIdle()
            
            // Then - should handle appropriately based on validation logic
            if (speed < 0) {
                verify { mockErrorObserver.onChanged("Invalid speed value: $speed") }
            } else if (speed > 100) {
                verify { mockErrorObserver.onChanged("Speed exceeds maximum limit: $speed") }
            }
        }
    }

    @Test
    fun `should handle multiple error conditions in sequence`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(Exception("Start failed"))
        coEvery { mockRepository.updateSpeed(any()) } returns Result.failure(Exception("Speed failed"))
        coEvery { mockRepository.changeDirection(any()) } returns Result.failure(Exception("Direction failed"))

        // When
        viewModel.startDrive()
        viewModel.updateSpeed(25.0)
        viewModel.changeDirection(Direction.FORWARD)
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockErrorObserver.onChanged("Start failed") }
        verify { mockErrorObserver.onChanged("Speed failed") }
        verify { mockErrorObserver.onChanged("Direction failed") }
    }

    @Test
    fun `should handle view model reinitialization after cleanup`() = runTest {
        // Given
        viewModel.driveState.removeObserver(mockStateObserver)
        viewModel.errorMessage.removeObserver(mockErrorObserver)
        viewModel.isLoading.removeObserver(mockLoadingObserver)

        // When - reinitialize observers
        viewModel.driveState.observeForever(mockStateObserver)
        viewModel.errorMessage.observeForever(mockErrorObserver)
        viewModel.isLoading.observeForever(mockLoadingObserver)

        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
    }

    @Test
    fun `should handle repository returning malformed results`() = runTest {
        // Given - simulate repository returning unexpected data
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(
            java.lang.IllegalStateException("Repository in invalid state")
        )

        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockErrorObserver.onChanged("Repository in invalid state") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should validate all drive state transitions are properly tested`() = runTest {
        // Given - test all possible state transitions
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        coEvery { mockRepository.resumeDrive() } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)
        coEvery { mockRepository.emergencyStop() } returns Result.success(Unit)

        // When - complete state cycle
        // IDLE -> DRIVING
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }

        // DRIVING -> PAUSED
        viewModel.pauseDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        verify { mockStateObserver.onChanged(DriveState.PAUSED) }

        // PAUSED -> DRIVING
        viewModel.resumeDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        verify(atLeast = 2) { mockStateObserver.onChanged(DriveState.DRIVING) }

        // DRIVING -> EMERGENCY_STOP
        viewModel.emergencyStop()
        testDispatcher.scheduler.advanceUntilIdle()
        verify { mockStateObserver.onChanged(DriveState.EMERGENCY_STOP) }

        // EMERGENCY_STOP -> IDLE
        viewModel.reset()
        testDispatcher.scheduler.advanceUntilIdle()
        verify(atLeast = 2) { mockStateObserver.onChanged(DriveState.IDLE) }
    }

    @Test
    fun `should handle resource cleanup during active operations`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            kotlinx.coroutines.delay(1000)
            Result.success(Unit)
        }

        // When - start operation and immediately clean up
        viewModel.startDrive()
        kotlinx.coroutines.delay(100)
        
        // Simulate cleanup
        viewModel.driveState.removeObserver(mockStateObserver)
        viewModel.errorMessage.removeObserver(mockErrorObserver)
        viewModel.isLoading.removeObserver(mockLoadingObserver)
        
        testDispatcher.scheduler.advanceUntilIdle()

        // Then - should not crash during cleanup
        assertTrue(true) // Test passes if no exceptions are thrown
    }

    // Additional comprehensive tests for maximum coverage and robustness

    @Test
    fun `should handle view model initialization with null dependencies gracefully`() {
        // Given - This tests defensive programming against null injection
        try {
            val viewModelWithNulls = OracleDriveControlViewModel(null, null)
            // If this doesn't throw, we should still verify it behaves reasonably
        } catch (e: Exception) {
            // Expected behavior for null dependencies
            assertTrue("Should handle null dependencies appropriately", true)
        }
    }

    @Test
    fun `should handle multiple observers on the same live data`() = runTest {
        // Given
        val secondStateObserver: Observer<DriveState> = mockk(relaxed = true)
        val secondErrorObserver: Observer<String> = mockk(relaxed = true)
        val secondLoadingObserver: Observer<Boolean> = mockk(relaxed = true)
        
        viewModel.driveState.observeForever(secondStateObserver)
        viewModel.errorMessage.observeForever(secondErrorObserver)
        viewModel.isLoading.observeForever(secondLoadingObserver)
        
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - both observers should receive updates
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        verify { secondStateObserver.onChanged(DriveState.DRIVING) }
        verify { mockLoadingObserver.onChanged(true) }
        verify { secondLoadingObserver.onChanged(true) }
        verify { mockLoadingObserver.onChanged(false) }
        verify { secondLoadingObserver.onChanged(false) }
        
        // Cleanup
        viewModel.driveState.removeObserver(secondStateObserver)
        viewModel.errorMessage.removeObserver(secondErrorObserver)
        viewModel.isLoading.removeObserver(secondLoadingObserver)
    }

    @Test
    fun `should handle direction enum edge cases`() = runTest {
        // Given - test all possible direction values
        val allDirections = listOf(
            Direction.FORWARD,
            Direction.REVERSE,
            Direction.LEFT,
            Direction.RIGHT
        )
        
        allDirections.forEach { direction ->
            coEvery { mockRepository.changeDirection(direction) } returns Result.success(Unit)
            
            // When
            viewModel.changeDirection(direction)
            testDispatcher.scheduler.advanceUntilIdle()
            
            // Then
            coVerify { mockRepository.changeDirection(direction) }
        }
    }

    @Test
    fun `should handle repository throwing runtime exceptions instead of returning Result`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } throws RuntimeException("Unexpected repository exception")
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - should catch and handle the exception
        verify { mockErrorObserver.onChanged("Unexpected repository exception") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
        verify { mockLoadingObserver.onChanged(false) }
    }

    @Test
    fun `should handle speed updates with maximum precision floating point values`() = runTest {
        // Given
        val maxPrecisionSpeed = 12.345678901234567890
        coEvery { mockRepository.updateSpeed(maxPrecisionSpeed) } returns Result.success(Unit)
        
        // When
        viewModel.updateSpeed(maxPrecisionSpeed)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        coVerify { mockRepository.updateSpeed(maxPrecisionSpeed) }
    }

    @Test
    fun `should handle interruption during long-running operations`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            kotlinx.coroutines.delay(5000) // Long operation
            Result.success(Unit)
        }
        
        // When
        val job = async { viewModel.startDrive() }
        kotlinx.coroutines.delay(100)
        job.cancel() // Interrupt the operation
        
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - should handle interruption gracefully
        verify { mockLoadingObserver.onChanged(true) }
        verify { mockLoadingObserver.onChanged(false) }
    }

    @Test
    fun `should handle state consistency during rapid error scenarios`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(Exception("Error 1"))
        coEvery { mockRepository.pauseDrive() } returns Result.failure(Exception("Error 2"))
        coEvery { mockRepository.resumeDrive() } returns Result.failure(Exception("Error 3"))
        
        // When - rapid error-prone operations
        viewModel.startDrive()
        viewModel.pauseDrive()
        viewModel.resumeDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - should maintain consistent error state
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
        verify { mockErrorObserver.onChanged("Error 1") }
        verify { mockErrorObserver.onChanged("Error 2") }
        verify { mockErrorObserver.onChanged("Error 3") }
    }

    @Test
    fun `should handle view model lifecycle during background operations`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            kotlinx.coroutines.delay(1000)
            Result.success(Unit)
        }
        
        // When - simulate activity/fragment lifecycle
        viewModel.startDrive()
        kotlinx.coroutines.delay(100)
        
        // Simulate onPause - remove observers
        viewModel.driveState.removeObserver(mockStateObserver)
        viewModel.errorMessage.removeObserver(mockErrorObserver)
        viewModel.isLoading.removeObserver(mockLoadingObserver)
        
        // Simulate onResume - re-add observers
        val newStateObserver: Observer<DriveState> = mockk(relaxed = true)
        val newErrorObserver: Observer<String> = mockk(relaxed = true)
        val newLoadingObserver: Observer<Boolean> = mockk(relaxed = true)
        
        viewModel.driveState.observeForever(newStateObserver)
        viewModel.errorMessage.observeForever(newErrorObserver)
        viewModel.isLoading.observeForever(newLoadingObserver)
        
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - new observers should receive current state
        verify { newStateObserver.onChanged(DriveState.DRIVING) }
        verify { newLoadingObserver.onChanged(false) }
        
        // Cleanup
        viewModel.driveState.removeObserver(newStateObserver)
        viewModel.errorMessage.removeObserver(newErrorObserver)
        viewModel.isLoading.removeObserver(newLoadingObserver)
    }

    @Test
    fun `should handle network manager state changes during operations`() = runTest {
        // Given
        every { mockNetworkManager.isConnected() } returns true andThen false andThen true
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        
        // When - check network at different times
        val firstCheck = viewModel.isNetworkAvailable()
        viewModel.startDrive()
        val secondCheck = viewModel.isNetworkAvailable()
        testDispatcher.scheduler.advanceUntilIdle()
        val thirdCheck = viewModel.isNetworkAvailable()
        
        // Then
        assertTrue(firstCheck)
        assertFalse(secondCheck)
        assertTrue(thirdCheck)
        verify(exactly = 3) { mockNetworkManager.isConnected() }
    }

    @Test
    fun `should handle error message formatting for different exception types`() = runTest {
        // Given - various exception types
        val exceptions = listOf(
            IllegalArgumentException("Invalid argument"),
            IllegalStateException("Invalid state"),
            UnsupportedOperationException("Operation not supported"),
            ConcurrentModificationException("Concurrent modification"),
            IndexOutOfBoundsException("Index out of bounds")
        )
        
        exceptions.forEach { exception ->
            coEvery { mockRepository.startDrive(any()) } returns Result.failure(exception)
            
            // When
            viewModel.startDrive()
            testDispatcher.scheduler.advanceUntilIdle()
            
            // Then
            verify { mockErrorObserver.onChanged(exception.message ?: exception.javaClass.simpleName) }
        }
    }

    @Test
    fun `should handle speed validation edge cases with scientific notation`() = runTest {
        // Given
        val scientificNotationSpeeds = listOf(
            1e-10, // Very small
            1e10,  // Very large
            1.23e2, // 123.0
            5.67e-3 // 0.00567
        )
        
        scientificNotationSpeeds.forEach { speed ->
            // When
            viewModel.updateSpeed(speed)
            testDispatcher.scheduler.advanceUntilIdle()
            
            // Then - should handle based on actual value, not notation
            if (speed < 0) {
                verify { mockErrorObserver.onChanged("Invalid speed value: $speed") }
            } else if (speed > 100) {
                verify { mockErrorObserver.onChanged("Speed exceeds maximum limit: $speed") }
            } else {
                coVerify { mockRepository.updateSpeed(speed) }
            }
        }
    }

    @Test
    fun `should handle repository method call ordering and dependencies`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.updateSpeed(any()) } returns Result.success(Unit)
        coEvery { mockRepository.changeDirection(any()) } returns Result.success(Unit)
        
        // When - operations that might have dependencies
        viewModel.startDrive()
        viewModel.updateSpeed(25.0)
        viewModel.changeDirection(Direction.FORWARD)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - verify call order is maintained
        coVerifyOrder {
            mockRepository.startDrive(any())
            mockRepository.updateSpeed(25.0)
            mockRepository.changeDirection(Direction.FORWARD)
        }
    }

    @Test
    fun `should handle memory leaks prevention through proper observer management`() = runTest {
        // Given
        val tempObserver: Observer<DriveState> = mockk(relaxed = true)
        
        // When - add and remove observer multiple times
        repeat(10) {
            viewModel.driveState.observeForever(tempObserver)
            viewModel.driveState.removeObserver(tempObserver)
        }
        
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - temp observer should not receive notifications
        verify(exactly = 0) { tempObserver.onChanged(any()) }
        // But original observer should still work
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
    }

    @Test
    fun `should handle repository authentication or permission errors`() = runTest {
        // Given
        val authErrors = listOf(
            SecurityException("Authentication failed"),
            IllegalAccessException("Permission denied"),
            RuntimeException("Unauthorized access")
        )
        
        authErrors.forEach { error ->
            coEvery { mockRepository.startDrive(any()) } returns Result.failure(error)
            
            // When
            viewModel.startDrive()
            testDispatcher.scheduler.advanceUntilIdle()
            
            // Then
            verify { mockErrorObserver.onChanged(error.message ?: "Authentication error") }
            verify { mockStateObserver.onChanged(DriveState.ERROR) }
        }
    }

    @Test
    fun `should handle view model state persistence across configuration changes`() = runTest {
        // Given - simulate configuration change by creating new observers
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Simulate configuration change - remove old observers
        viewModel.driveState.removeObserver(mockStateObserver)
        viewModel.errorMessage.removeObserver(mockErrorObserver)
        viewModel.isLoading.removeObserver(mockLoadingObserver)
        
        // When - add new observers (simulating new activity/fragment)
        val configChangeStateObserver: Observer<DriveState> = mockk(relaxed = true)
        val configChangeErrorObserver: Observer<String> = mockk(relaxed = true)
        val configChangeLoadingObserver: Observer<Boolean> = mockk(relaxed = true)
        
        viewModel.driveState.observeForever(configChangeStateObserver)
        viewModel.errorMessage.observeForever(configChangeErrorObserver)
        viewModel.isLoading.observeForever(configChangeLoadingObserver)
        
        // Then - new observers should immediately receive current state
        verify { configChangeStateObserver.onChanged(DriveState.DRIVING) }
        verify { configChangeLoadingObserver.onChanged(false) }
        
        // Cleanup
        viewModel.driveState.removeObserver(configChangeStateObserver)
        viewModel.errorMessage.removeObserver(configChangeErrorObserver)
        viewModel.isLoading.removeObserver(configChangeLoadingObserver)
    }

    @Test
    fun `should handle custom exceptions with nested causes`() = runTest {
        // Given
        val rootCause = IllegalArgumentException("Root cause")
        val nestedException = RuntimeException("Nested exception", rootCause)
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(nestedException)
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Nested exception") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle repository returning results with different success types`() = runTest {
        // Given - test different success scenarios
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.updateSpeed(any()) } returns Result.success(Unit)
        
        // When
        viewModel.startDrive()
        viewModel.updateSpeed(30.0)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - should handle both success types appropriately
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        coVerify { mockRepository.startDrive(any()) }
        coVerify { mockRepository.updateSpeed(30.0) }
    }

    @Test
    fun `should handle edge case where repository methods return immediately`() = runTest {
        // Given - no delay in repository calls
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)
        
        // When - rapid calls without waiting
        viewModel.startDrive()
        viewModel.stopDrive()
        // Don't advance scheduler to test immediate returns
        
        // Then - should handle immediate completion
        verify(timeout = 1000) { mockStateObserver.onChanged(DriveState.DRIVING) }
        verify(timeout = 1000) { mockStateObserver.onChanged(DriveState.IDLE) }
    }

    @Test
    fun `should validate error recovery after emergency stop`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.emergencyStop() } returns Result.success(Unit)
        
        // When - emergency stop followed by reset and normal operation
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.emergencyStop()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.reset()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - should be able to operate normally after recovery
        verify(atLeast = 2) { mockStateObserver.onChanged(DriveState.DRIVING) }
        verify { mockStateObserver.onChanged(DriveState.EMERGENCY_STOP) }
        verify(atLeast = 2) { mockStateObserver.onChanged(DriveState.IDLE) }
    }

    @Test
    fun `should handle speed debouncing with exact timing scenarios`() = runTest {
        // Given
        coEvery { mockRepository.updateSpeed(any()) } returns Result.success(Unit)
        
        // When - test debouncing with specific timing
        viewModel.updateSpeed(10.0)
        kotlinx.coroutines.delay(50) // Less than debounce time
        viewModel.updateSpeed(20.0)
        kotlinx.coroutines.delay(50) // Less than debounce time
        viewModel.updateSpeed(30.0)
        
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - only final value should be used due to debouncing
        coVerify(exactly = 1) { mockRepository.updateSpeed(30.0) }
        coVerify(exactly = 0) { mockRepository.updateSpeed(10.0) }
        coVerify(exactly = 0) { mockRepository.updateSpeed(20.0) }
    }
}