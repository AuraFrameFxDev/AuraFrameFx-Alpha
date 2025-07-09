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
    fun `should handle timeout scenarios gracefully`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            kotlinx.coroutines.delay(30000) // Simulate timeout
            Result.failure(Exception("Operation timed out"))
        }
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceTimeBy(30000)
        
        // Then
        verify { mockErrorObserver.onChanged("Operation timed out") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle memory pressure scenarios`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(OutOfMemoryError("Out of memory"))
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Out of memory") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should validate speed boundary conditions`() = runTest {
        // Given - test zero speed
        coEvery { mockRepository.updateSpeed(0.0) } returns Result.success(Unit)
        
        // When
        viewModel.updateSpeed(0.0)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        coVerify { mockRepository.updateSpeed(0.0) }
    }

    @Test
    fun `should handle floating point precision in speed updates`() = runTest {
        // Given
        val preciseSpeed = 12.345678901234567
        coEvery { mockRepository.updateSpeed(any()) } returns Result.success(Unit)
        
        // When
        viewModel.updateSpeed(preciseSpeed)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        coVerify { mockRepository.updateSpeed(preciseSpeed) }
    }

    @Test
    fun `should handle infinite speed values`() = runTest {
        // Given
        val infiniteSpeed = Double.POSITIVE_INFINITY
        
        // When
        viewModel.updateSpeed(infiniteSpeed)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Invalid speed value: Infinity") }
        coVerify(exactly = 0) { mockRepository.updateSpeed(any()) }
    }

    @Test
    fun `should handle negative infinity speed values`() = runTest {
        // Given
        val negativeInfiniteSpeed = Double.NEGATIVE_INFINITY
        
        // When
        viewModel.updateSpeed(negativeInfiniteSpeed)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Invalid speed value: -Infinity") }
        coVerify(exactly = 0) { mockRepository.updateSpeed(any()) }
    }

    @Test
    fun `should handle state transitions in specific order`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        coEvery { mockRepository.resumeDrive() } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)
        
        // When - specific state transition sequence
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.pauseDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.resumeDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.stopDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - verify state transitions occurred in correct order
        verifyOrder {
            mockStateObserver.onChanged(DriveState.IDLE)
            mockStateObserver.onChanged(DriveState.DRIVING)
            mockStateObserver.onChanged(DriveState.PAUSED)
            mockStateObserver.onChanged(DriveState.DRIVING)
            mockStateObserver.onChanged(DriveState.IDLE)
        }
    }

    @Test
    fun `should handle repository connection failures`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(java.net.ConnectException("Connection refused"))
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Connection refused") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle security exceptions gracefully`() = runTest {
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
    fun `should handle multiple observer registrations`() = runTest {
        // Given
        val additionalStateObserver: Observer<DriveState> = mockk(relaxed = true)
        viewModel.driveState.observeForever(additionalStateObserver)
        
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        verify { additionalStateObserver.onChanged(DriveState.DRIVING) }
        
        // Cleanup
        viewModel.driveState.removeObserver(additionalStateObserver)
    }

    @Test
    fun `should handle observer removal during operation`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            kotlinx.coroutines.delay(100)
            Result.success(Unit)
        }
        
        // When
        viewModel.startDrive()
        viewModel.driveState.removeObserver(mockStateObserver)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - should not crash and should complete operation
        verify { mockLoadingObserver.onChanged(true) }
        verify { mockLoadingObserver.onChanged(false) }
    }

    @Test
    fun `should handle network manager throwing exceptions`() {
        // Given
        every { mockNetworkManager.isConnected() } throws RuntimeException("Network check failed")
        
        // When & Then
        try {
            viewModel.isNetworkAvailable()
        } catch (e: RuntimeException) {
            assertEquals("Network check failed", e.message)
        }
    }

    @Test
    fun `should handle repository method calls with custom parameters`() = runTest {
        // Given
        val customParameter = "custom_drive_mode"
        coEvery { mockRepository.startDrive(customParameter) } returns Result.success(Unit)
        
        // When
        viewModel.startDrive(customParameter)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        coVerify { mockRepository.startDrive(customParameter) }
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
    }

    @Test
    fun `should handle all direction enum values`() = runTest {
        // Given
        val directions = listOf(Direction.FORWARD, Direction.REVERSE, Direction.LEFT, Direction.RIGHT)
        directions.forEach { direction ->
            coEvery { mockRepository.changeDirection(direction) } returns Result.success(Unit)
        }
        
        // When & Then
        directions.forEach { direction ->
            viewModel.changeDirection(direction)
            testDispatcher.scheduler.advanceUntilIdle()
            coVerify { mockRepository.changeDirection(direction) }
        }
    }

    @Test
    fun `should handle emergency stop from different states`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        coEvery { mockRepository.emergencyStop() } returns Result.success(Unit)
        
        val states = listOf(
            { viewModel.startDrive() },
            { 
                viewModel.startDrive()
                testDispatcher.scheduler.advanceUntilIdle()
                viewModel.pauseDrive()
            }
        )
        
        // When & Then
        states.forEach { stateSetup ->
            stateSetup()
            testDispatcher.scheduler.advanceUntilIdle()
            viewModel.emergencyStop()
            testDispatcher.scheduler.advanceUntilIdle()
            
            verify { mockStateObserver.onChanged(DriveState.EMERGENCY_STOP) }
            
            // Reset for next test
            viewModel.reset()
            testDispatcher.scheduler.advanceUntilIdle()
            clearMocks(mockStateObserver)
        }
    }

    @Test
    fun `should handle coroutine cancellation gracefully`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            kotlinx.coroutines.delay(1000)
            Result.success(Unit)
        }
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceTimeBy(500)
        
        // Cancel by starting another operation
        viewModel.emergencyStop()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockStateObserver.onChanged(DriveState.EMERGENCY_STOP) }
    }

    @Test
    fun `should handle viewmodel clearing properly`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // When - simulate viewmodel being cleared
        viewModel.onCleared()
        
        // Then - should handle cleanup gracefully
        // This test ensures no crashes occur during cleanup
        assertTrue(true) // If we reach here without exception, test passes
    }

    @Test
    fun `should handle background thread operations`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        
        // When - simulate background operation
        val backgroundJob = launch {
            viewModel.startDrive()
        }
        
        testDispatcher.scheduler.advanceUntilIdle()
        backgroundJob.join()
        
        // Then
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
    }

    @Test
    fun `should handle main thread operations`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        
        // When - ensure operation happens on main thread
        launch(Dispatchers.Main) {
            viewModel.startDrive()
        }
        
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
    }

    @Test
    fun `should handle repository returning empty results`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - should handle empty success result
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        verify { mockLoadingObserver.onChanged(false) }
    }

    @Test
    fun `should handle speed validation edge cases`() = runTest {
        // Given - various edge case speeds
        val edgeCases = listOf(
            Double.MIN_VALUE,
            Double.MAX_VALUE,
            0.0000001,
            999.99999
        )
        
        edgeCases.forEach { speed ->
            coEvery { mockRepository.updateSpeed(speed) } returns Result.success(Unit)
        }
        
        // When & Then
        edgeCases.forEach { speed ->
            viewModel.updateSpeed(speed)
            testDispatcher.scheduler.advanceUntilIdle()
            
            if (speed >= 0 && speed <= 1000) {
                coVerify { mockRepository.updateSpeed(speed) }
            }
        }
    }

    @Test
    fun `should handle multiple error conditions simultaneously`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(Exception("Primary error"))
        every { mockNetworkManager.isConnected() } returns false
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Primary error") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
        assertFalse(viewModel.isNetworkAvailable())
    }

    @Test
    fun `should handle resource cleanup on destruction`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // When - simulate destruction
        viewModel.driveState.removeObserver(mockStateObserver)
        viewModel.errorMessage.removeObserver(mockErrorObserver)
        viewModel.isLoading.removeObserver(mockLoadingObserver)
        
        // Then - should not crash
        viewModel.stopDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Verify operations still work even without observers
        coVerify { mockRepository.stopDrive() }
    }

    @Test
    fun `should handle ViewModel state persistence across configuration changes`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // When - simulate configuration change by creating new observers
        val newStateObserver: Observer<DriveState> = mockk(relaxed = true)
        viewModel.driveState.observeForever(newStateObserver)
        
        // Then - new observer should receive current state
        verify { newStateObserver.onChanged(DriveState.DRIVING) }
        
        // Cleanup
        viewModel.driveState.removeObserver(newStateObserver)
    }
}