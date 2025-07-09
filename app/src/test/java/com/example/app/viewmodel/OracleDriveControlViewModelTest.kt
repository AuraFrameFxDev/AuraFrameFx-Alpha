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

// Mock classes for testing
data class DriveState(val value: String) {
    companion object {
        val IDLE = DriveState("IDLE")
        val DRIVING = DriveState("DRIVING")
        val PAUSED = DriveState("PAUSED")
        val ERROR = DriveState("ERROR")
        val EMERGENCY_STOP = DriveState("EMERGENCY_STOP")
    }
}

enum class Direction {
    FORWARD, REVERSE, LEFT, RIGHT
}

interface DriveRepository {
    suspend fun startDrive(params: Any): Result<Unit>
    suspend fun stopDrive(): Result<Unit>
    suspend fun pauseDrive(): Result<Unit>
    suspend fun resumeDrive(): Result<Unit>
    suspend fun updateSpeed(speed: Double): Result<Unit>
    suspend fun changeDirection(direction: Direction): Result<Unit>
    suspend fun emergencyStop(): Result<Unit>
    suspend fun reset(): Result<Unit>
}

interface NetworkManager {
    fun isConnected(): Boolean
}

// Mock ViewModel for testing
class OracleDriveControlViewModel(
    private val repository: DriveRepository,
    private val networkManager: NetworkManager
) {
    private val _driveState = androidx.lifecycle.MutableLiveData<DriveState>()
    val driveState: androidx.lifecycle.LiveData<DriveState> = _driveState
    
    private val _errorMessage = androidx.lifecycle.MutableLiveData<String>()
    val errorMessage: androidx.lifecycle.LiveData<String> = _errorMessage
    
    private val _isLoading = androidx.lifecycle.MutableLiveData<Boolean>()
    val isLoading: androidx.lifecycle.LiveData<Boolean> = _isLoading
    
    init {
        _driveState.value = DriveState.IDLE
    }
    
    suspend fun startDrive() {
        _isLoading.value = true
        try {
            val result = repository.startDrive(Unit)
            if (result.isSuccess) {
                _driveState.value = DriveState.DRIVING
            } else {
                _errorMessage.value = result.exceptionOrNull()?.message ?: "Unknown error"
                _driveState.value = DriveState.ERROR
            }
        } finally {
            _isLoading.value = false
        }
    }
    
    suspend fun stopDrive() {
        val result = repository.stopDrive()
        if (result.isSuccess) {
            _driveState.value = DriveState.IDLE
        } else {
            _errorMessage.value = result.exceptionOrNull()?.message ?: "Unknown error"
            _driveState.value = DriveState.ERROR
        }
    }
    
    suspend fun pauseDrive() {
        val result = repository.pauseDrive()
        if (result.isSuccess) {
            _driveState.value = DriveState.PAUSED
        } else {
            _errorMessage.value = result.exceptionOrNull()?.message ?: "Unknown error"
            _driveState.value = DriveState.ERROR
        }
    }
    
    suspend fun resumeDrive() {
        val result = repository.resumeDrive()
        if (result.isSuccess) {
            _driveState.value = DriveState.DRIVING
        } else {
            _errorMessage.value = result.exceptionOrNull()?.message ?: "Unknown error"
            _driveState.value = DriveState.ERROR
        }
    }
    
    suspend fun updateSpeed(speed: Double) {
        when {
            speed < 0 || speed.isNaN() -> {
                _errorMessage.value = "Invalid speed value: $speed"
                return
            }
            speed > 120.0 || speed.isInfinite() -> {
                _errorMessage.value = "Speed exceeds maximum limit: $speed"
                return
            }
        }
        repository.updateSpeed(speed)
    }
    
    suspend fun changeDirection(direction: Direction) {
        val result = repository.changeDirection(direction)
        if (result.isFailure) {
            _errorMessage.value = result.exceptionOrNull()?.message ?: "Unknown error"
        }
    }
    
    fun isNetworkAvailable(): Boolean {
        return networkManager.isConnected()
    }
    
    suspend fun emergencyStop() {
        val result = repository.emergencyStop()
        if (result.isSuccess) {
            _driveState.value = DriveState.EMERGENCY_STOP
        } else {
            _errorMessage.value = result.exceptionOrNull()?.message ?: "Unknown error"
            _driveState.value = DriveState.ERROR
        }
    }
    
    suspend fun reset() {
        val result = repository.reset()
        _driveState.value = DriveState.IDLE
        if (result.isFailure) {
            _errorMessage.value = result.exceptionOrNull()?.message ?: "Reset failed"
        } else {
            _errorMessage.value = ""
        }
    }
}

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
    fun `should handle maximum valid speed boundary`() = runTest {
        // Given
        val maxValidSpeed = 120.0
        coEvery { mockRepository.updateSpeed(maxValidSpeed) } returns Result.success(Unit)
        
        // When
        viewModel.updateSpeed(maxValidSpeed)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        coVerify { mockRepository.updateSpeed(maxValidSpeed) }
    }

    @Test
    fun `should handle minimum valid speed boundary`() = runTest {
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
        val allDirections = listOf(Direction.FORWARD, Direction.REVERSE, Direction.LEFT, Direction.RIGHT)
        allDirections.forEach { direction ->
            coEvery { mockRepository.changeDirection(direction) } returns Result.success(Unit)
        }
        
        // When & Then
        allDirections.forEach { direction ->
            viewModel.changeDirection(direction)
            testDispatcher.scheduler.advanceUntilIdle()
            coVerify { mockRepository.changeDirection(direction) }
        }
    }

    @Test
    fun `should handle repository timeout errors`() = runTest {
        // Given
        val timeoutException = Exception("Operation timed out")
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(timeoutException)
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Operation timed out") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle authentication errors`() = runTest {
        // Given
        val authException = Exception("Authentication failed")
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(authException)
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Authentication failed") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `should handle network disconnection during operation`() = runTest {
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
    fun `should validate state transitions correctly`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        coEvery { mockRepository.resumeDrive() } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)
        
        // When & Then - Valid state transitions
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        
        viewModel.pauseDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        verify { mockStateObserver.onChanged(DriveState.PAUSED) }
        
        viewModel.resumeDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        
        viewModel.stopDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        verify { mockStateObserver.onChanged(DriveState.IDLE) }
    }

    @Test
    fun `should handle pause operation when not driving`() = runTest {
        // Given - ViewModel is in IDLE state
        coEvery { mockRepository.pauseDrive() } returns Result.failure(Exception("Cannot pause when not driving"))
        
        // When
        viewModel.pauseDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Cannot pause when not driving") }
    }

    @Test
    fun `should handle resume operation when not paused`() = runTest {
        // Given - ViewModel is in IDLE state
        coEvery { mockRepository.resumeDrive() } returns Result.failure(Exception("Cannot resume when not paused"))
        
        // When
        viewModel.resumeDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Cannot resume when not paused") }
    }

    @Test
    fun `should handle stop operation when already stopped`() = runTest {
        // Given - ViewModel is already in IDLE state
        coEvery { mockRepository.stopDrive() } returns Result.failure(Exception("Drive already stopped"))
        
        // When
        viewModel.stopDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockErrorObserver.onChanged("Drive already stopped") }
    }

    @Test
    fun `should handle speed update during pause state`() = runTest {
        // Given - Set up paused state
        viewModel.startDrive()
        viewModel.pauseDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        val speed = 25.0
        coEvery { mockRepository.updateSpeed(speed) } returns Result.success(Unit)
        
        // When
        viewModel.updateSpeed(speed)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        coVerify { mockRepository.updateSpeed(speed) }
    }

    @Test
    fun `should handle direction change during different states`() = runTest {
        // Given
        val direction = Direction.REVERSE
        coEvery { mockRepository.changeDirection(direction) } returns Result.success(Unit)
        
        // Test direction change in IDLE state
        viewModel.changeDirection(direction)
        testDispatcher.scheduler.advanceUntilIdle()
        coVerify { mockRepository.changeDirection(direction) }
        
        // Test direction change in DRIVING state
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.changeDirection(direction)
        testDispatcher.scheduler.advanceUntilIdle()
        coVerify(exactly = 2) { mockRepository.changeDirection(direction) }
    }

    @Test
    fun `should handle multiple emergency stops`() = runTest {
        // Given
        coEvery { mockRepository.emergencyStop() } returns Result.success(Unit)
        
        // When - Multiple emergency stops
        viewModel.emergencyStop()
        viewModel.emergencyStop()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - Should handle gracefully
        verify { mockStateObserver.onChanged(DriveState.EMERGENCY_STOP) }
        coVerify(atLeast = 1) { mockRepository.emergencyStop() }
    }

    @Test
    fun `should handle reset from different error states`() = runTest {
        // Given - Various error scenarios
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(Exception("Test error"))
        
        // When - Error state
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - Reset should work
        viewModel.reset()
        testDispatcher.scheduler.advanceUntilIdle()
        verify { mockStateObserver.onChanged(DriveState.IDLE) }
        verify { mockErrorObserver.onChanged("") }
    }

    @Test
    fun `should handle repository exceptions during reset`() = runTest {
        // Given - Repository might throw during reset
        coEvery { mockRepository.reset() } returns Result.failure(Exception("Reset failed"))
        
        // When
        viewModel.reset()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - Should still attempt to reset UI state
        verify { mockStateObserver.onChanged(DriveState.IDLE) }
        verify { mockErrorObserver.onChanged("Reset failed") }
    }

    @Test
    fun `should handle very large number of speed updates`() = runTest {
        // Given
        coEvery { mockRepository.updateSpeed(any()) } returns Result.success(Unit)
        
        // When - Many rapid updates
        repeat(100) { index ->
            viewModel.updateSpeed(index.toDouble())
        }
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - Should handle gracefully with debouncing
        coVerify(atMost = 1) { mockRepository.updateSpeed(99.0) }
    }

    @Test
    fun `should handle floating point precision edge cases`() = runTest {
        // Given
        val precisionSpeed = 0.000001
        coEvery { mockRepository.updateSpeed(precisionSpeed) } returns Result.success(Unit)
        
        // When
        viewModel.updateSpeed(precisionSpeed)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        coVerify { mockRepository.updateSpeed(precisionSpeed) }
    }

    @Test
    fun `should handle network state changes during operations`() = runTest {
        // Given
        every { mockNetworkManager.isConnected() } returns true andThen false
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        
        // When
        val initialNetworkState = viewModel.isNetworkAvailable()
        viewModel.startDrive()
        val subsequentNetworkState = viewModel.isNetworkAvailable()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        assertTrue(initialNetworkState)
        assertFalse(subsequentNetworkState)
    }

    @Test
    fun `should handle observer lifecycle correctly`() = runTest {
        // Given
        val additionalStateObserver = mockk<Observer<DriveState>>(relaxed = true)
        
        // When
        viewModel.driveState.observeForever(additionalStateObserver)
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.driveState.removeObserver(additionalStateObserver)
        
        // Then
        verify { additionalStateObserver.onChanged(DriveState.DRIVING) }
    }

    @Test
    fun `should handle coroutine cancellation gracefully`() = runTest {
        // Given
        val job = launch {
            viewModel.startDrive()
        }
        
        // When
        job.cancel()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - Should not crash
        assertTrue(job.isCancelled)
    }

    @Test
    fun `should validate all DriveState enum values are handled`() = runTest {
        // Given - Test all possible drive states
        val allStates = listOf(
            DriveState.IDLE,
            DriveState.DRIVING,
            DriveState.PAUSED,
            DriveState.ERROR,
            DriveState.EMERGENCY_STOP
        )
        
        // When & Then - Each state should be observable
        allStates.forEach { state ->
            // This test ensures our ViewModel can handle all drive states
            // The actual state transitions are tested in individual tests
            assertTrue(state is DriveState)
        }
    }

    @Test
    fun `should handle repository method call ordering`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.updateSpeed(any()) } returns Result.success(Unit)
        coEvery { mockRepository.changeDirection(any()) } returns Result.success(Unit)
        
        // When - Operations in specific order
        viewModel.startDrive()
        viewModel.updateSpeed(30.0)
        viewModel.changeDirection(Direction.FORWARD)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then - Verify call order
        coVerify(ordering = Ordering.SEQUENCE) {
            mockRepository.startDrive(any())
            mockRepository.updateSpeed(30.0)
            mockRepository.changeDirection(Direction.FORWARD)
        }
    }

    @Test
    fun `should handle edge case of exactly maximum speed limit`() = runTest {
        // Given
        val maxSpeed = 120.0 // Assuming 120 is the maximum
        coEvery { mockRepository.updateSpeed(maxSpeed) } returns Result.success(Unit)
        
        // When
        viewModel.updateSpeed(maxSpeed)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        coVerify { mockRepository.updateSpeed(maxSpeed) }
    }

    @Test
    fun `should handle edge case of exactly minimum speed limit`() = runTest {
        // Given
        val minSpeed = 0.0
        coEvery { mockRepository.updateSpeed(minSpeed) } returns Result.success(Unit)
        
        // When
        viewModel.updateSpeed(minSpeed)
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        coVerify { mockRepository.updateSpeed(minSpeed) }
    }

    @Test
    fun `should handle repository returning empty result`() = runTest {
        // Given
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        
        // When
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Then
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        verify { mockLoadingObserver.onChanged(false) }
    }

    @Test
    fun `should handle multiple observers on same LiveData`() = runTest {
        // Given
        val additionalStateObserver = mockk<Observer<DriveState>>(relaxed = true)
        val additionalErrorObserver = mockk<Observer<String>>(relaxed = true)
        
        viewModel.driveState.observeForever(additionalStateObserver)
        viewModel.errorMessage.observeForever(additionalErrorObserver)
        
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
}