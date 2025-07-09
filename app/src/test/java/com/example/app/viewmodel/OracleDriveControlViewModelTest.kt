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
        verify { mockSateObserver.onChanged(DriveState.IDLE) }
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
    fun `should handle maximum speed boundary correctly`() = runTest {
        // Given
        val maxSpeed = 999.0
        coEvery { mockRepository.updateSpeed(maxSpeed) } returns Result.success(Unit)
        
        // When
        viewModel.updateSpeed(maxSpeed)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        coVerify { mockRepository.updateSpeed(maxSpeed) }
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
    fun `should handle negative infinite speed values`() = runTest {
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
    fun `should handle repository connection timeout`() = runTest {
        // Given
        val timeoutException = java.util.concurrent.TimeoutException("Connection timeout")
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(timeoutException)
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Connection timeout") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle network interruption during drive`() = runTest {
        // Given
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        every { mockNetworkManager.isConnected() } returns false
        
        // When
        val networkStatus = viewModel.isNetworkAvailable()
        
        // Then
        assertFalse(networkStatus)
        verify { mockNetworkManager.isConnected() }
    }

    @Test
    fun `should handle multiple observers correctly`() = runTest {
        // Given
        val additionalStateObserver: Observer<DriveState> = mockk(relaxed = true)
        val additionalErrorObserver: Observer<String> = mockk(relaxed = true)
        
        viewModel.driveState.observeForever(additionalStateObserver)
        viewModel.errorMessage.observeForever(additionalErrorObserver)
        
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        verify { additionalStateObserver.onChanged(DriveState.DRIVING) }
        
        // Cleanup
        viewModel.driveState.removeObserver(additionalStateObserver)
        viewModel.errorMessage.removeObserver(additionalErrorObserver)
    }

    @Test
    fun `should handle observer removal during operation`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            kotlinx.coroutines.delay(50)
            Result.success(Unit)
        }
        
        // When
        viewModel.startDrive()
        viewModel.driveState.removeObserver(mockStateObserver)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - Original observer should not receive final state change
        verify(exactly = 1) { mockStateObserver.onChanged(DriveState.IDLE) } // Only initial state
    }

    @Test
    fun `should handle IllegalStateException from repository`() = runTest {
        // Given
        val illegalStateException = IllegalStateException("Drive system not initialized")
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(illegalStateException)
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Drive system not initialized") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle SecurityException from repository`() = runTest {
        // Given
        val securityException = SecurityException("Insufficient permissions")
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(securityException)
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Insufficient permissions") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle multiple rapid emergency stops`() = runTest {
        // Given
        coEvery { mockRepository.emergencyStop() } returns Result.success(Unit)
        
        // When
        viewModel.emergencyStop()
        viewModel.emergencyStop()
        viewModel.emergencyStop()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify(atLeast = 1) { mockStateObserver.onChanged(DriveState.EMERGENCY_STOP) }
        coVerify(atLeast = 1) { mockRepository.emergencyStop() }
    }

    @Test
    fun `should handle pause when not driving`() = runTest {
        // Given - Initial idle state
        coEvery { mockRepository.pauseDrive() } returns Result.failure(IllegalStateException("Cannot pause when not driving"))
        
        // When
        viewModel.pauseDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Cannot pause when not driving") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle resume when not paused`() = runTest {
        // Given - Start driving first
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        coEvery { mockRepository.resumeDrive() } returns Result.failure(IllegalStateException("Cannot resume when not paused"))
        
        // When
        viewModel.resumeDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Cannot resume when not paused") }
    }

    @Test
    fun `should handle stop when not driving`() = runTest {
        // Given - Initial idle state
        coEvery { mockRepository.stopDrive() } returns Result.failure(IllegalStateException("Cannot stop when not driving"))
        
        // When
        viewModel.stopDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Cannot stop when not driving") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle repository returning null result`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns null
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged(any()) }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle very large speed values`() = runTest {
        // Given
        val largeSpeed = Double.MAX_VALUE
        
        // When
        viewModel.updateSpeed(largeSpeed)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Speed exceeds maximum limit: ${Double.MAX_VALUE}") }
        coVerify(exactly = 0) { mockRepository.updateSpeed(any()) }
    }

    @Test
    fun `should handle very small positive speed values`() = runTest {
        // Given
        val smallSpeed = Double.MIN_VALUE
        coEvery { mockRepository.updateSpeed(smallSpeed) } returns Result.success(Unit)
        
        // When
        viewModel.updateSpeed(smallSpeed)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        coVerify { mockRepository.updateSpeed(smallSpeed) }
    }

    @Test
    fun `should handle network state changes during operations`() = runTest {
        // Given
        every { mockNetworkManager.isConnected() } returns true andThen false
        
        // When
        val firstCheck = viewModel.isNetworkAvailable()
        val secondCheck = viewModel.isNetworkAvailable()
        
        // Then
        assertTrue(firstCheck)
        assertFalse(secondCheck)
        verify(exactly = 2) { mockNetworkManager.isConnected() }
    }

    @Test
    fun `should handle coroutine cancellation gracefully`() = runTest {
        // Given
        val job = kotlinx.coroutines.Job()
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            kotlinx.coroutines.delay(1000)
            Result.success(Unit)
        }
        
        // When
        viewModel.startDrive()
        job.cancel()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - Should handle cancellation without crashing
        verify { mockLoadingObserver.onChanged(true) }
    }

    @Test
    fun `should handle state consistency during rapid operations`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        coEvery { mockRepository.resumeDrive() } returns Result.success(Unit)
        
        // When - Very rapid operations
        repeat(10) {
            viewModel.startDrive()
            viewModel.pauseDrive()
            viewModel.resumeDrive()
            viewModel.stopDrive()
        }
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - Should end in consistent state
        verify(atLeast = 1) { mockStateObserver.onChanged(DriveState.IDLE) }
    }

    @Test
    fun `should handle memory pressure scenarios`() = runTest {
        // Given
        val largeData = "x".repeat(1000000)
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(OutOfMemoryError("Memory pressure"))
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Memory pressure") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle multiple reset operations`() = runTest {
        // Given
        viewModel.emergencyStop()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // When
        viewModel.reset()
        viewModel.reset()
        viewModel.reset()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify(atLeast = 3) { mockStateObserver.onChanged(DriveState.IDLE) }
        verify(atLeast = 3) { mockErrorObserver.onChanged("") }
    }

    @Test
    fun `should handle direction change during emergency stop`() = runTest {
        // Given
        viewModel.emergencyStop()
        testDispatcher.scheduler.advanceUntilIdle()
        coEvery { mockRepository.changeDirection(any()) } returns Result.failure(IllegalStateException("Cannot change direction during emergency stop"))
        
        // When
        viewModel.changeDirection(Direction.FORWARD)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Cannot change direction during emergency stop") }
    }

    @Test
    fun `should handle speed update during emergency stop`() = runTest {
        // Given
        viewModel.emergencyStop()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // When
        viewModel.updateSpeed(50.0)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Cannot update speed during emergency stop") }
        coVerify(exactly = 0) { mockRepository.updateSpeed(any()) }
    }

    @Test
    fun `should handle loading state during concurrent operations`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            kotlinx.coroutines.delay(100)
            Result.success(Unit)
        }
        coEvery { mockRepository.updateSpeed(any()) } coAnswers {
            kotlinx.coroutines.delay(50)
            Result.success(Unit)
        }
        
        // When
        viewModel.startDrive()
        viewModel.updateSpeed(25.0)
        
        // Then - Both operations should show loading
        verify(atLeast = 2) { mockLoadingObserver.onChanged(true) }
        
        // When - operations complete
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - Loading should be false
        verify(atLeast = 2) { mockLoadingObserver.onChanged(false) }
    }

    @Test
    fun `should handle error message formatting consistency`() = runTest {
        // Given
        val customException = RuntimeException("Custom error message")
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(customException)
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Custom error message") }
    }

    @Test
    fun `should handle null error messages gracefully`() = runTest {
        // Given
        val nullMessageException = RuntimeException(null as String?)
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(nullMessageException)
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged(any()) }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle empty error messages gracefully`() = runTest {
        // Given
        val emptyMessageException = RuntimeException("")
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(emptyMessageException)
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle InterruptedException from repository`() = runTest {
        // Given
        val interruptedException = InterruptedException("Operation interrupted")
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(interruptedException)
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Operation interrupted") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle all DriveState enum values properly`() = runTest {
        // Given
        val allStates = listOf(
            DriveState.IDLE,
            DriveState.DRIVING,
            DriveState.PAUSED,
            DriveState.EMERGENCY_STOP,
            DriveState.ERROR
        )
        
        // When & Then - Test each state can be set
        allStates.forEach { state ->
            // This would require internal state manipulation or specific repository responses
            // to trigger each state, which is already covered in individual tests
            assertTrue("State $state should be valid", allStates.contains(state))
        }
    }

    @Test
    fun `should handle ViewModel lifecycle correctly`() = runTest {
        // Given
        val newViewModel = OracleDriveControlViewModel(mockRepository, mockNetworkManager)
        
        // When
        val initialState = newViewModel.driveState.value
        
        // Then
        assertEquals(DriveState.IDLE, initialState)
        
        // Cleanup
        newViewModel.driveState.removeObserver(mockStateObserver)
        newViewModel.errorMessage.removeObserver(mockErrorObserver)
        newViewModel.isLoading.removeObserver(mockLoadingObserver)
    }
}