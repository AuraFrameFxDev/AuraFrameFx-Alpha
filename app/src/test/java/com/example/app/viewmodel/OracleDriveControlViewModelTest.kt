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
        testDispa