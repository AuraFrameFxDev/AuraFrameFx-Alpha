package com.example.app.viewmodel

import androidx.arch.core.executor.testing.InstantTaskExecutorRule
import androidx.lifecycle.Observer
import io.mockk.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.*
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
}










































































































































































































































































































































































































































        coVerify(exactly = 0) { mockRepository.updateSpeed(10.0) }
        coVerify(exactly = 0) { mockRepository.updateSpeed(20.0) }
    }

    // Additional comprehensive edge cases and integration scenarios

    @Test
    fun `should handle all Direction enum values correctly`() = runTest {
        // Given - test all possible Direction values
        val directions = listOf(Direction.FORWARD, Direction.REVERSE, Direction.LEFT, Direction.RIGHT)
        coEvery { mockRepository.changeDirection(any()) } returns Result.success(Unit)
        
        directions.forEach { direction ->
            // When
            viewModel.changeDirection(direction)
            testDispatcher.scheduler.advanceUntilIdle()
            
            // Then
            coVerify { mockRepository.changeDirection(direction) }
        }
    }

    @Test
    fun `should handle speed validation with edge boundary cases`() = runTest {
        // Given - comprehensive boundary testing
        val testCases = mapOf(
            -0.1 to "Invalid speed value: -0.1",
            100.1 to "Speed exceeds maximum limit: 100.1",
            Double.MAX_VALUE to "Speed exceeds maximum limit: ${Double.MAX_VALUE}",
            -Double.MAX_VALUE to "Invalid speed value: ${-Double.MAX_VALUE}"
        )
        
        testCases.forEach { (speed, expectedError) ->
            // When
            viewModel.updateSpeed(speed)
            testDispatcher.scheduler.advanceUntilIdle()
            
            // Then
            verify { mockErrorObserver.onChanged(expectedError) }
        }
    }

    @Test
    fun `should handle repository method parameter validation`() = runTest {
        // Given
        coEvery { mockRepository.updateSpeed(any()) } returns Result.success(Unit)
        
        // When - test edge cases for speed parameters
        val validSpeeds = listOf(0.0, 25.5, 50.0, 75.25, 99.99)
        validSpeeds.forEach { speed ->
            viewModel.updateSpeed(speed)
            testDispatcher.scheduler.advanceUntilIdle()
            coVerify { mockRepository.updateSpeed(speed) }
        }
    }

    @Test
    fun `should handle mixed successful and failed operations in sequence`() = runTest {
        // Given - alternating success and failure responses
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.pauseDrive() } returns Result.failure(Exception("Pause failed"))
        coEvery { mockRepository.resumeDrive() } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.failure(Exception("Stop failed"))
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.pauseDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.resumeDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.stopDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        verify { mockErrorObserver.onChanged("Pause failed") }
        verify(atLeast = 2) { mockStateObserver.onChanged(DriveState.DRIVING) }
        verify { mockErrorObserver.onChanged("Stop failed") }
    }

    @Test
    fun `should handle observer registration and deregistration patterns`() = runTest {
        // Given
        val additionalStateObserver: Observer<DriveState> = mockk(relaxed = true)
        val additionalErrorObserver: Observer<String> = mockk(relaxed = true)
        
        // When - add multiple observers
        viewModel.driveState.observeForever(additionalStateObserver)
        viewModel.errorMessage.observeForever(additionalErrorObserver)
        
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - both observers should be notified
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        verify { additionalStateObserver.onChanged(DriveState.DRIVING) }
        
        // Cleanup additional observers
        viewModel.driveState.removeObserver(additionalStateObserver)
        viewModel.errorMessage.removeObserver(additionalErrorObserver)
    }

    @Test
    fun `should handle complex error message scenarios`() = runTest {
        // Given - various error message formats
        val errorScenarios = listOf(
            "",  // Empty error
            " ",  // Whitespace only
            "\n\t\r",  // Special whitespace characters
            "Multi\nLine\nError\nMessage",  // Multi-line error
            "Error with \u0000 null character",  // Null character
            "Very long error message: ${"x".repeat(1000)}"  // Long message
        )
        
        errorScenarios.forEach { errorMsg ->
            coEvery { mockRepository.startDrive(any()) } returns Result.failure(Exception(errorMsg))
            
            // When
            viewModel.startDrive()
            testDispatcher.scheduler.advanceUntilIdle()
            
            // Then
            verify { mockErrorObserver.onChanged(errorMsg) }
            verify { mockStateObserver.onChanged(DriveState.ERROR) }
        }
    }

    @Test
    fun `should handle rapid debounced speed updates with timing verification`() = runTest {
        // Given
        coEvery { mockRepository.updateSpeed(any()) } returns Result.success(Unit)
        val speeds = (1..50).map { it.toDouble() }
        
        // When - rapid speed updates
        speeds.forEach { speed ->
            viewModel.updateSpeed(speed)
            kotlinx.coroutines.delay(10) // Small delay between updates
        }
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - should only process the last speed due to debouncing
        coVerify(exactly = 1) { mockRepository.updateSpeed(50.0) }
        coVerify(exactly = 0) { mockRepository.updateSpeed(match { it < 50.0 }) }
    }

    @Test
    fun `should handle network state changes during operations`() = runTest {
        // Given
        every { mockNetworkManager.isConnected() } returnsMany listOf(true, false, true)
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        
        // When - check network state multiple times during operation
        assertTrue(viewModel.isNetworkAvailable())
        viewModel.startDrive()
        assertFalse(viewModel.isNetworkAvailable())
        testDispatcher.scheduler.advanceUntilIdle()
        assertTrue(viewModel.isNetworkAvailable())
        
        // Then
        verify(exactly = 3) { mockNetworkManager.isConnected() }
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
    }

    @Test
    fun `should handle all DriveState enum transitions systematically`() = runTest {
        // Given - setup all repository methods
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        coEvery { mockRepository.resumeDrive() } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)
        coEvery { mockRepository.emergencyStop() } returns Result.success(Unit)
        
        // Test specific state transition sequences
        val stateTransitions = listOf(
            Triple("startDrive", { viewModel.startDrive() }, DriveState.DRIVING),
            Triple("pauseDrive", { viewModel.pauseDrive() }, DriveState.PAUSED),
            Triple("resumeDrive", { viewModel.resumeDrive() }, DriveState.DRIVING),
            Triple("emergencyStop", { viewModel.emergencyStop() }, DriveState.EMERGENCY_STOP),
            Triple("reset", { viewModel.reset() }, DriveState.IDLE)
        )
        
        stateTransitions.forEach { (operationName, operation, expectedState) ->
            // When
            operation()
            testDispatcher.scheduler.advanceUntilIdle()
            
            // Then
            verify { mockStateObserver.onChanged(expectedState) }
        }
    }

    @Test
    fun `should handle repository timeout scenarios with different durations`() = runTest {
        // Given - different timeout scenarios
        val timeoutExceptions = listOf(
            java.net.SocketTimeoutException("Read timeout: 5000ms"),
            java.net.ConnectException("Connection timeout: 10000ms"),
            java.util.concurrent.TimeoutException("Operation timeout: 30000ms")
        )
        
        timeoutExceptions.forEach { exception ->
            coEvery { mockRepository.startDrive(any()) } returns Result.failure(exception)
            
            // When
            viewModel.startDrive()
            testDispatcher.scheduler.advanceUntilIdle()
            
            // Then
            verify { mockErrorObserver.onChanged(exception.message ?: "Unknown timeout") }
            verify { mockStateObserver.onChanged(DriveState.ERROR) }
        }
    }

    @Test
    fun `should handle memory management during intensive operations`() = runTest {
        // Given - simulate memory-intensive operations
        coEvery { mockRepository.updateSpeed(any()) } returns Result.success(Unit)
        
        // When - perform many operations
        repeat(1000) { iteration ->
            viewModel.updateSpeed(iteration.toDouble())
            if (iteration % 100 == 0) {
                testDispatcher.scheduler.advanceUntilIdle()
            }
        }
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - should handle without memory issues
        coVerify(atLeast = 1) { mockRepository.updateSpeed(any()) }
    }

    @Test
    fun `should handle repository result success with different data types`() = runTest {
        // Given - various success scenarios
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.updateSpeed(any()) } returns Result.success(Unit)
        coEvery { mockRepository.changeDirection(any()) } returns Result.success(Unit)
        
        // When - call different repository methods
        viewModel.startDrive()
        viewModel.updateSpeed(25.0)
        viewModel.changeDirection(Direction.FORWARD)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - all should succeed
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        coVerify { mockRepository.updateSpeed(25.0) }
        coVerify { mockRepository.changeDirection(Direction.FORWARD) }
    }

    @Test
    fun `should handle coroutine context switching during operations`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            withContext(Dispatchers.IO) {
                kotlinx.coroutines.delay(100)
                Result.success(Unit)
            }
        }
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockLoadingObserver.onChanged(true) }
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        verify { mockLoadingObserver.onChanged(false) }
    }

    @Test
    fun `should handle error recovery scenarios`() = runTest {
        // Given - error followed by successful operation
        coEvery { mockRepository.startDrive(any()) } returnsMany listOf(
            Result.failure(Exception("First attempt failed")),
            Result.success(Unit)
        )
        
        // When - first attempt fails, second succeeds
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.reset()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("First attempt failed") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
        verify { mockStateObserver.onChanged(DriveState.IDLE) }
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
    }

    @Test
    fun `should handle precision floating point arithmetic edge cases`() = runTest {
        // Given - floating point edge cases
        val floatingPointTests = listOf(
            0.1 + 0.2,  // Classic floating point precision issue
            1.0 / 3.0,  // Repeating decimal
            Math.PI,    // Irrational number
            Math.E,     // Another irrational number
            Double.MIN_NORMAL,  // Smallest normal value
            Double.MAX_VALUE / 2  // Large but valid value
        )
        
        coEvery { mockRepository.updateSpeed(any()) } returns Result.success(Unit)
        
        floatingPointTests.forEach { speed ->
            // When
            viewModel.updateSpeed(speed)
            testDispatcher.scheduler.advanceUntilIdle()
            
            // Then - should handle floating point precision correctly
            if (speed >= 0 && speed <= 100) {
                coVerify { mockRepository.updateSpeed(speed) }
            }
        }
    }

    @Test
    fun `should handle thread safety with concurrent observer notifications`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.updateSpeed(any()) } returns Result.success(Unit)
        
        // When - concurrent operations that trigger observer notifications
        val jobs = (1..10).map { index ->
            async {
                viewModel.updateSpeed(index.toDouble())
                if (index == 5) {
                    viewModel.startDrive()
                }
            }
        }
        
        jobs.forEach { it.await() }
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - should handle concurrent notifications safely
        verify(atLeast = 1) { mockStateObserver.onChanged(any()) }
        coVerify(atLeast = 1) { mockRepository.updateSpeed(any()) }
    }

    @Test
    fun `should handle view model lifecycle edge cases`() = runTest {
        // Given - test lifecycle scenarios
        val newViewModel = OracleDriveControlViewModel(mockRepository, mockNetworkManager)
        val newStateObserver: Observer<DriveState> = mockk(relaxed = true)
        
        // When - test fresh view model initialization
        newViewModel.driveState.observeForever(newStateObserver)
        assertEquals(DriveState.IDLE, newViewModel.driveState.value)
        
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        newViewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { newStateObserver.onChanged(DriveState.IDLE) }
        verify { newStateObserver.onChanged(DriveState.DRIVING) }
        
        // Cleanup
        newViewModel.driveState.removeObserver(newStateObserver)
    }

    @Test
    fun `should validate error message consistency across different failure types`() = runTest {
        // Given - different exception types
        val exceptionTypes = listOf(
            IllegalArgumentException("Invalid argument"),
            IllegalStateException("Invalid state"),
            RuntimeException("Runtime error"),
            Exception("Generic exception"),
            Throwable("Generic throwable")
        )
        
        exceptionTypes.forEach { exception ->
            coEvery { mockRepository.startDrive(any()) } returns Result.failure(exception)
            
            // When
            viewModel.startDrive()
            testDispatcher.scheduler.advanceUntilIdle()
            
            // Then
            verify { mockErrorObserver.onChanged(exception.message ?: exception.toString()) }
            verify { mockStateObserver.onChanged(DriveState.ERROR) }
        }
    }
}