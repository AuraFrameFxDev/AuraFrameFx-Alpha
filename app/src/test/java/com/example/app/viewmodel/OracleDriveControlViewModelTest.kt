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
    fun `should handle repository timeout exceptions`() = runTest {
        // Given
        val timeoutException = java.util.concurrent.TimeoutException("Operation timed out")
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(timeoutException)

        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockErrorObserver.onChanged("Operation timed out") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle repository interrupted exceptions`() = runTest {
        // Given
        val interruptedException = InterruptedException("Operation interrupted")
        coEvery { mockRepository.pauseDrive() } returns Result.failure(interruptedException)

        // When
        viewModel.pauseDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockErrorObserver.onChanged("Operation interrupted") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle repository illegal state exceptions`() = runTest {
        // Given
        val illegalStateException = IllegalStateException("Invalid state transition")
        coEvery { mockRepository.resumeDrive() } returns Result.failure(illegalStateException)

        // When
        viewModel.resumeDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockErrorObserver.onChanged("Invalid state transition") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle repository illegal argument exceptions`() = runTest {
        // Given
        val illegalArgumentException = IllegalArgumentException("Invalid argument provided")
        coEvery { mockRepository.updateSpeed(any()) } returns Result.failure(illegalArgumentException)

        // When
        viewModel.updateSpeed(50.0)
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockErrorObserver.onChanged("Invalid argument provided") }
    }

    @Test
    fun `should handle network connectivity changes during operations`() = runTest {
        // Given
        every { mockNetworkManager.isConnected() } returns false
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(Exception("Network unavailable"))

        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockErrorObserver.onChanged("Network unavailable") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle multiple observers correctly`() = runTest {
        // Given
        val mockSecondStateObserver: Observer<DriveState> = mockk(relaxed = true)
        val mockSecondErrorObserver: Observer<String> = mockk(relaxed = true)

        viewModel.driveState.observeForever(mockSecondStateObserver)
        viewModel.errorMessage.observeForever(mockSecondErrorObserver)

        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)

        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        verify { mockSecondStateObserver.onChanged(DriveState.DRIVING) }

        // Cleanup
        viewModel.driveState.removeObserver(mockSecondStateObserver)
        viewModel.errorMessage.removeObserver(mockSecondErrorObserver)
    }

    @Test
    fun `should handle repository method call ordering correctly`() = runTest {
        // Given
        val callOrder = mutableListOf<String>()
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            callOrder.add("startDrive")
            Result.success(Unit)
        }
        coEvery { mockRepository.updateSpeed(any()) } coAnswers {
            callOrder.add("updateSpeed")
            Result.success(Unit)
        }
        coEvery { mockRepository.changeDirection(any()) } coAnswers {
            callOrder.add("changeDirection")
            Result.success(Unit)
        }

        // When
        viewModel.startDrive()
        viewModel.updateSpeed(25.0)
        viewModel.changeDirection(Direction.FORWARD)
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        coVerify(ordering = Ordering.ORDERED) {
            mockRepository.startDrive(any())
            mockRepository.updateSpeed(25.0)
            mockRepository.changeDirection(Direction.FORWARD)
        }
    }

    // Additional comprehensive test cases for edge scenarios and boundary conditions

    @Test
    fun `should handle very small positive speed values near zero`() = runTest {
        // Given
        val microSpeed = 0.0001
        coEvery { mockRepository.updateSpeed(microSpeed) } returns Result.success(Unit)

        // When
        viewModel.updateSpeed(microSpeed)
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        coVerify { mockRepository.updateSpeed(microSpeed) }
    }

    @Test
    fun `should handle decimal precision edge cases`() = runTest {
        // Given
        val precisionSpeed = 33.333333333333336 // Common floating point precision issue
        coEvery { mockRepository.updateSpeed(any()) } returns Result.success(Unit)

        // When
        viewModel.updateSpeed(precisionSpeed)
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        coVerify { mockRepository.updateSpeed(precisionSpeed) }
    }

    @Test
    fun `should handle state persistence across multiple operations`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        coEvery { mockRepository.resumeDrive() } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)

        // When - complete lifecycle
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.pauseDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.resumeDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.stopDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then - verify complete state progression
        verifyOrder {
            mockStateObserver.onChanged(DriveState.IDLE) // Initial state
            mockStateObserver.onChanged(DriveState.DRIVING) // After start
            mockStateObserver.onChanged(DriveState.PAUSED) // After pause
            mockStateObserver.onChanged(DriveState.DRIVING) // After resume
            mockStateObserver.onChanged(DriveState.IDLE) // After stop
        }
    }

    @Test
    fun `should handle repository returning custom exceptions`() = runTest {
        // Given
        class CustomDriveException(message: String) : Exception(message)
        val customException = CustomDriveException("Custom drive error occurred")
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(customException)

        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockErrorObserver.onChanged("Custom drive error occurred") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle rapid successive emergency stops`() = runTest {
        // Given
        coEvery { mockRepository.emergencyStop() } returns Result.success(Unit)

        // When
        repeat(3) {
            viewModel.emergencyStop()
        }
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify(exactly = 3) { mockStateObserver.onChanged(DriveState.EMERGENCY_STOP) }
        coVerify(exactly = 3) { mockRepository.emergencyStop() }
    }

    @Test
    fun `should handle mixed successful and failed operations in sequence`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.updateSpeed(25.0) } returns Result.failure(Exception("Speed error"))
        coEvery { mockRepository.updateSpeed(50.0) } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)

        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        viewModel.updateSpeed(25.0)
        testDispatcher.scheduler.advanceUntilIdle()
        
        viewModel.updateSpeed(50.0)
        testDispatcher.scheduler.advanceUntilIdle()
        
        viewModel.stopDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        verify { mockErrorObserver.onChanged("Speed error") }
        verify { mockStateObserver.onChanged(DriveState.IDLE) }
    }

    @Test
    fun `should handle observer registration during ongoing operations`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            kotlinx.coroutines.delay(100)
            Result.success(Unit)
        }
        
        val lateObserver: Observer<DriveState> = mockk(relaxed = true)

        // When
        viewModel.startDrive()
        kotlinx.coroutines.delay(50) // Register observer mid-operation
        viewModel.driveState.observeForever(lateObserver)
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        verify { lateObserver.onChanged(DriveState.DRIVING) }

        // Cleanup
        viewModel.driveState.removeObserver(lateObserver)
    }

    @Test
    fun `should handle boundary speed values at validation thresholds`() = runTest {
        // Given
        val boundaryValues = listOf(
            -0.001, // Just below zero
            0.0,    // Exactly zero
            0.001,  // Just above zero
            99.999, // Just below max
            100.0,  // At max threshold
            100.001 // Just above max
        )

        // When & Then
        boundaryValues.forEach { speed ->
            clearMocks(mockRepository, mockErrorObserver)
            
            if (speed < 0) {
                viewModel.updateSpeed(speed)
                testDispatcher.scheduler.advanceUntilIdle()
                verify { mockErrorObserver.onChanged("Invalid speed value: $speed") }
            } else if (speed > 100) {
                viewModel.updateSpeed(speed)
                testDispatcher.scheduler.advanceUntilIdle()
                verify { mockErrorObserver.onChanged("Speed exceeds maximum limit: $speed") }
            } else {
                coEvery { mockRepository.updateSpeed(speed) } returns Result.success(Unit)
                viewModel.updateSpeed(speed)
                testDispatcher.scheduler.advanceUntilIdle()
                coVerify { mockRepository.updateSpeed(speed) }
            }
        }
    }

    @Test
    fun `should handle coroutine scope cancellation during repository calls`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            kotlinx.coroutines.delay(1000)
            throw kotlinx.coroutines.CancellationException("Operation cancelled")
        }

        // When
        viewModel.startDrive()
        kotlinx.coroutines.delay(100)
        testDispatcher.cancel()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then - should handle cancellation gracefully without crashes
        verify { mockLoadingObserver.onChanged(true) }
    }

    @Test
    fun `should handle memory pressure scenarios with multiple observers`() = runTest {
        // Given - simulate memory pressure with many observers
        val observers = mutableListOf<Observer<DriveState>>()
        repeat(50) { 
            val observer = mockk<Observer<DriveState>>(relaxed = true)
            observers.add(observer)
            viewModel.driveState.observeForever(observer)
        }

        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)

        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then - all observers should receive updates
        observers.forEach { observer ->
            verify { observer.onChanged(DriveState.DRIVING) }
        }

        // Cleanup
        observers.forEach { observer ->
            viewModel.driveState.removeObserver(observer)
        }
    }

    @Test
    fun `should handle state recovery after system failures`() = runTest {
        // Given - simulate system failure scenario
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(Exception("System failure"))
        
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // System recovery
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)

        // When - retry after recovery
        viewModel.reset()
        testDispatcher.scheduler.advanceUntilIdle()
        
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
        verify { mockStateObserver.onChanged(DriveState.IDLE) }
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
    }

    @Test
    fun `should handle thread safety with concurrent state updates`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.updateSpeed(any()) } returns Result.success(Unit)
        coEvery { mockRepository.changeDirection(any()) } returns Result.success(Unit)

        // When - simulate concurrent operations
        val job1 = viewModel.startDrive()
        val job2 = viewModel.updateSpeed(30.0)
        val job3 = viewModel.changeDirection(Direction.FORWARD)
        
        listOf(job1, job2, job3).forEach { it.join() }
        testDispatcher.scheduler.advanceUntilIdle()

        // Then - all operations should complete successfully
        coVerify { mockRepository.startDrive(any()) }
        coVerify { mockRepository.updateSpeed(30.0) }
        coVerify { mockRepository.changeDirection(Direction.FORWARD) }
    }

    @Test
    fun `should handle complex state transitions under stress`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        coEvery { mockRepository.resumeDrive() } returns Result.success(Unit)
        coEvery { mockRepository.emergencyStop() } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)

        // When - stress test with rapid state changes
        repeat(10) {
            viewModel.startDrive()
            testDispatcher.scheduler.advanceUntilIdle()
            
            viewModel.pauseDrive()
            testDispatcher.scheduler.advanceUntilIdle()
            
            viewModel.resumeDrive()
            testDispatcher.scheduler.advanceUntilIdle()
            
            if (it % 3 == 0) {
                viewModel.emergencyStop()
                testDispatcher.scheduler.advanceUntilIdle()
                viewModel.reset()
                testDispatcher.scheduler.advanceUntilIdle()
            } else {
                viewModel.stopDrive()
                testDispatcher.scheduler.advanceUntilIdle()
            }
        }

        // Then - should handle all transitions without corruption
        coVerify(atLeast = 10) { mockRepository.startDrive(any()) }
        coVerify(atLeast = 10) { mockRepository.pauseDrive() }
        coVerify(atLeast = 10) { mockRepository.resumeDrive() }
    }

    @Test
    fun `should handle resource cleanup when observers are removed during operations`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            kotlinx.coroutines.delay(200)
            Result.success(Unit)
        }

        // When
        viewModel.startDrive()
        kotlinx.coroutines.delay(100)
        
        // Remove observers mid-operation
        viewModel.driveState.removeObserver(mockStateObserver)
        viewModel.errorMessage.removeObserver(mockErrorObserver)
        viewModel.isLoading.removeObserver(mockLoadingObserver)
        
        testDispatcher.scheduler.advanceUntilIdle()

        // Then - should complete operation without crashes
        coVerify { mockRepository.startDrive(any()) }
        
        // Re-add observers for teardown
        viewModel.driveState.observeForever(mockStateObserver)
        viewModel.errorMessage.observeForever(mockErrorObserver)
        viewModel.isLoading.observeForever(mockLoadingObserver)
    }
}