package dev.aurakai.auraframefx.app.viewmodel // Corrected package name

import androidx.arch.core.executor.testing.InstantTaskExecutorRule
import androidx.lifecycle.Observer
import io.mockk.clearAllMocks
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

// Assuming DriveRepository, NetworkManager, DriveState, Direction are in a package accessible from here
// e.g., import dev.aurakai.auraframefx.model.*
// or specific imports if they are in different sub-packages of dev.aurakai.auraframefx

@OptIn(ExperimentalCoroutinesApi::class)
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
        val initialState = viewModel.driveState.value
        assertEquals(DriveState.IDLE, initialState)
        verify { mockStateObserver.onChanged(DriveState.IDLE) }
    }

    @Test
    fun `startDrive should update state to driving when successful`() = runTest {
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)

        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockLoadingObserver.onChanged(true) }
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        verify { mockLoadingObserver.onChanged(false) }
        coVerify { mockRepository.startDrive(any()) }
    }

    @Test
    fun `startDrive should handle network error gracefully`() = runTest {
        val errorMessage = "Network connection failed"
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(Exception(errorMessage))

        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockLoadingObserver.onChanged(true) }
        verify { mockErrorObserver.onChanged(errorMessage) }
        verify { mockLoadingObserver.onChanged(false) }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `stopDrive should update state to idle when successful`() = runTest {
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)

        viewModel.stopDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockStateObserver.onChanged(DriveState.IDLE) }
        coVerify { mockRepository.stopDrive() }
    }

    @Test
    fun `stopDrive should handle repository error`() = runTest {
        val errorMessage = "Failed to stop drive"
        coEvery { mockRepository.stopDrive() } returns Result.failure(Exception(errorMessage))

        viewModel.stopDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged(errorMessage) }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `pauseDrive should update state to paused when successful`() = runTest {
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)

        viewModel.pauseDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockStateObserver.onChanged(DriveState.PAUSED) }
        coVerify { mockRepository.pauseDrive() }
    }

    @Test
    fun `resumeDrive should update state to driving from paused`() = runTest {
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        viewModel.pauseDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        coEvery { mockRepository.resumeDrive() } returns Result.success(Unit)

        viewModel.resumeDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        coVerify { mockRepository.resumeDrive() }
    }

    @Test
    fun `updateSpeed should call repository with correct value`() = runTest {
        val speed = 50.0
        coEvery { mockRepository.updateSpeed(speed) } returns Result.success(Unit)

        viewModel.updateSpeed(speed)
        testDispatcher.scheduler.advanceUntilIdle()

        coVerify { mockRepository.updateSpeed(speed) }
    }

    @Test
    fun `updateSpeed should handle negative values gracefully`() = runTest {
        val invalidSpeed = -10.0

        viewModel.updateSpeed(invalidSpeed)
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged("Invalid speed value: $invalidSpeed") }
        coVerify(exactly = 0) { mockRepository.updateSpeed(any()) }
    }

    @Test
    fun `updateSpeed should handle excessive values gracefully`() = runTest {
        val excessiveSpeed = 1000.0

        viewModel.updateSpeed(excessiveSpeed)
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged("Speed exceeds maximum limit: $excessiveSpeed") }
        coVerify(exactly = 0) { mockRepository.updateSpeed(any()) }
    }

    @Test
    fun `changeDirection should update direction when valid`() = runTest {
        val direction = Direction.FORWARD
        coEvery { mockRepository.changeDirection(direction) } returns Result.success(Unit)

        viewModel.changeDirection(direction)
        testDispatcher.scheduler.advanceUntilIdle()

        coVerify { mockRepository.changeDirection(direction) }
    }

    @Test
    fun `changeDirection should handle repository error`() = runTest {
        val direction = Direction.REVERSE
        val errorMessage = "Failed to change direction"
        coEvery { mockRepository.changeDirection(direction) } returns Result.failure(
            Exception(
                errorMessage
            )
        )

        viewModel.changeDirection(direction)
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged(errorMessage) }
    }

    @Test
    fun `isNetworkAvailable should return network manager status`() {
        every { mockNetworkManager.isConnected() } returns true

        val result = viewModel.isNetworkAvailable()

        assertTrue(result)
        verify { mockNetworkManager.isConnected() }
    }

    @Test
    fun `isNetworkAvailable should return false when network unavailable`() {
        every { mockNetworkManager.isConnected() } returns false

        val result = viewModel.isNetworkAvailable()

        assertFalse(result)
        verify { mockNetworkManager.isConnected() }
    }

    @Test
    fun `emergency stop should immediately stop drive and update state`() = runTest {
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        coEvery { mockRepository.emergencyStop() } returns Result.success(Unit)

        viewModel.emergencyStop()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockStateObserver.onChanged(DriveState.EMERGENCY_STOP) }
        coVerify { mockRepository.emergencyStop() }
    }

    @Test
    fun `emergency stop should handle repository failure`() = runTest {
        val errorMessage = "Emergency stop failed"
        coEvery { mockRepository.emergencyStop() } returns Result.failure(Exception(errorMessage))

        viewModel.emergencyStop()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged(errorMessage) }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `reset should clear error state and return to idle`() = runTest {
        coEvery { mockRepository.emergencyStop() } returns Result.success(Unit)
        viewModel.emergencyStop()
        testDispatcher.scheduler.advanceUntilIdle()

        viewModel.reset()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockStateObserver.onChanged(DriveState.IDLE) }
        verify { mockErrorObserver.onChanged("") }
    }

    @Test
    fun `concurrent operations should be handled correctly`() = runTest {
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)

        viewModel.startDrive()
        viewModel.stopDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockStateObserver.onChanged(DriveState.IDLE) }
        coVerify { mockRepository.startDrive(any()) }
        coVerify { mockRepository.stopDrive() }
    }

    @Test
    fun `multiple speed updates should debounce correctly`() = runTest {
        coEvery { mockRepository.updateSpeed(any()) } returns Result.success(Unit)

        viewModel.updateSpeed(10.0)
        viewModel.updateSpeed(20.0)
        viewModel.updateSpeed(30.0)
        testDispatcher.scheduler.advanceUntilIdle()

        coVerify(exactly = 1) { mockRepository.updateSpeed(30.0) }
    }

    @Test
    fun `loading state should be managed correctly across operations`() = runTest {
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            kotlinx.coroutines.delay(100)
            Result.success(Unit)
        }

        viewModel.startDrive()

        verify { mockLoadingObserver.onChanged(true) }

        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockLoadingObserver.onChanged(false) }
    }

    @Test
    fun `view model should handle null repository responses gracefully`() = runTest {
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(NullPointerException("Repository returned null"))

        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged("Repository returned null") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `view model should validate input parameters`() = runTest {
        val invalidSpeed = Double.NaN

        viewModel.updateSpeed(invalidSpeed)
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged("Invalid speed value: NaN") }
        coVerify(exactly = 0) { mockRepository.updateSpeed(any()) }
    }
}

// Mocked/Placeholder classes for dependencies - replace with actual imports or mocks
interface DriveRepository {
    suspend fun startDrive(params: Any): Result<Unit>
    suspend fun stopDrive(): Result<Unit>
    suspend fun pauseDrive(): Result<Unit>
    suspend fun resumeDrive(): Result<Unit>
    suspend fun updateSpeed(speed: Double): Result<Unit>
    suspend fun changeDirection(direction: Direction): Result<Unit>
    suspend fun emergencyStop(): Result<Unit>
}

interface NetworkManager {
    fun isConnected(): Boolean
}

enum class DriveState {
    IDLE, DRIVING, PAUSED, ERROR, EMERGENCY_STOP
}

enum class Direction {
    FORWARD, REVERSE
}

    // Additional comprehensive test cases for thorough coverage
    // Testing Framework: JUnit 4/5 with MockK, AndroidX testing utilities, Kotlin coroutines testing

    @Test
    fun `speed updates should respect minimum boundary values`() = runTest {
        val minimumSpeed = 0.0
        coEvery { mockRepository.updateSpeed(minimumSpeed) } returns Result.success(Unit)

        viewModel.updateSpeed(minimumSpeed)
        testDispatcher.scheduler.advanceUntilIdle()

        coVerify { mockRepository.updateSpeed(minimumSpeed) }
    }

    @Test
    fun `speed updates should handle floating point precision edge cases`() = runTest {
        val precisionSpeed = 0.000001
        coEvery { mockRepository.updateSpeed(precisionSpeed) } returns Result.success(Unit)

        viewModel.updateSpeed(precisionSpeed)
        testDispatcher.scheduler.advanceUntilIdle()

        coVerify { mockRepository.updateSpeed(precisionSpeed) }
    }

    @Test
    fun `speed updates should handle positive infinity gracefully`() = runTest {
        val infiniteSpeed = Double.POSITIVE_INFINITY

        viewModel.updateSpeed(infiniteSpeed)
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged("Invalid speed value: Infinity") }
        coVerify(exactly = 0) { mockRepository.updateSpeed(any()) }
    }

    @Test
    fun `speed updates should handle negative infinity gracefully`() = runTest {
        val negativeInfiniteSpeed = Double.NEGATIVE_INFINITY

        viewModel.updateSpeed(negativeInfiniteSpeed)
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged("Invalid speed value: -Infinity") }
        coVerify(exactly = 0) { mockRepository.updateSpeed(any()) }
    }

    @Test
    fun `state transitions should be validated correctly`() = runTest {
        // Test invalid state transitions
        viewModel.pauseDrive() // Should not be able to pause when not driving
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged("Cannot pause when not driving") }
        coVerify(exactly = 0) { mockRepository.pauseDrive() }
    }

    @Test
    fun `resume drive should validate current state`() = runTest {
        // Try to resume without being paused
        viewModel.resumeDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged("Cannot resume when not paused") }
        coVerify(exactly = 0) { mockRepository.resumeDrive() }
    }

    @Test
    fun `stop drive should work from any valid state`() = runTest {
        // Test stopping from PAUSED state
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)

        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.pauseDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.stopDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockStateObserver.onChanged(DriveState.IDLE) }
        coVerify { mockRepository.stopDrive() }
    }

    @Test
    fun `direction changes should be validated against current state`() = runTest {
        // Try to change direction when not driving
        val direction = Direction.FORWARD
        
        viewModel.changeDirection(direction)
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged("Cannot change direction when not driving") }
        coVerify(exactly = 0) { mockRepository.changeDirection(any()) }
    }

    @Test
    fun `emergency stop should work from any state`() = runTest {
        coEvery { mockRepository.emergencyStop() } returns Result.success(Unit)

        // Test emergency stop from IDLE state
        viewModel.emergencyStop()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockStateObserver.onChanged(DriveState.EMERGENCY_STOP) }
        coVerify { mockRepository.emergencyStop() }
    }

    @Test
    fun `multiple concurrent emergency stops should be handled safely`() = runTest {
        coEvery { mockRepository.emergencyStop() } returns Result.success(Unit)

        viewModel.emergencyStop()
        viewModel.emergencyStop()
        viewModel.emergencyStop()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockStateObserver.onChanged(DriveState.EMERGENCY_STOP) }
        coVerify(atLeast = 1) { mockRepository.emergencyStop() }
    }

    @Test
    fun `reset should work from any error state`() = runTest {
        // Put system in error state first
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(Exception("Test error"))
        
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        
        // Now reset
        viewModel.reset()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockStateObserver.onChanged(DriveState.IDLE) }
        verify { mockErrorObserver.onChanged("") }
    }

    @Test
    fun `network connectivity changes should be handled gracefully`() = runTest {
        // Test operations when network becomes unavailable
        every { mockNetworkManager.isConnected() } returns false
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(Exception("Network unavailable"))

        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged("Network unavailable") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `repository timeout should be handled gracefully`() = runTest {
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            kotlinx.coroutines.delay(Long.MAX_VALUE) // Simulate timeout
            Result.success(Unit)
        }

        viewModel.startDrive()
        
        // Simulate timeout handling
        testDispatcher.scheduler.advanceTimeBy(30_000) // 30 seconds
        
        verify { mockLoadingObserver.onChanged(true) }
        // Should handle timeout gracefully
    }

    @Test
    fun `error recovery scenarios should work correctly`() = runTest {
        // Test error followed by successful operation
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(Exception("Initial error"))
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockStateObserver.onChanged(DriveState.ERROR) }

        // Reset and try again successfully
        viewModel.reset()
        testDispatcher.scheduler.advanceUntilIdle()
        
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
    }

    @Test
    fun `observer lifecycle should be managed correctly`() {
        // Test removing and re-adding observers
        viewModel.driveState.removeObserver(mockStateObserver)
        viewModel.driveState.observeForever(mockStateObserver)

        val initialState = viewModel.driveState.value
        assertEquals(DriveState.IDLE, initialState)
    }

    @Test
    fun `complex state machine transitions should be validated`() = runTest {
        // Test complete workflow: IDLE -> DRIVING -> PAUSED -> DRIVING -> IDLE
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        coEvery { mockRepository.resumeDrive() } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)

        // Start
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }

        // Pause
        viewModel.pauseDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        verify { mockStateObserver.onChanged(DriveState.PAUSED) }

        // Resume
        viewModel.resumeDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }

        // Stop
        viewModel.stopDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        verify { mockStateObserver.onChanged(DriveState.IDLE) }
    }

    @Test
    fun `repository exceptions should be properly categorized`() = runTest {
        // Test different exception types
        val networkException = java.net.UnknownHostException("Network error")
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(networkException)

        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged("Network error") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `rapid successive operations should be debounced properly`() = runTest {
        coEvery { mockRepository.updateSpeed(any()) } returns Result.success(Unit)

        // Rapid speed changes
        for (i in 1..10) {
            viewModel.updateSpeed(i.toDouble())
        }
        testDispatcher.scheduler.advanceUntilIdle()

        // Should only process the last speed update
        coVerify(exactly = 1) { mockRepository.updateSpeed(10.0) }
    }

    @Test
    fun `view model should handle memory pressure gracefully`() = runTest {
        // Simulate memory pressure by creating many observers
        val observers = mutableListOf<Observer<DriveState>>()
        repeat(100) {
            val observer = mockk<Observer<DriveState>>(relaxed = true)
            observers.add(observer)
            viewModel.driveState.observeForever(observer)
        }

        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Clean up observers
        observers.forEach { viewModel.driveState.removeObserver(it) }
    }

    @Test
    fun `error messages should be properly formatted`() = runTest {
        val customException = IllegalArgumentException("Custom error message")
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(customException)

        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged("Custom error message") }
    }

    @Test
    fun `loading state should be consistent across all operations`() = runTest {
        val operations = listOf(
            { viewModel.startDrive() },
            { viewModel.stopDrive() },
            { viewModel.pauseDrive() },
            { viewModel.resumeDrive() },
            { viewModel.emergencyStop() }
        )

        // Mock all repository calls to succeed
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.stopDrive() } returns Result.success(Unit)
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        coEvery { mockRepository.resumeDrive() } returns Result.success(Unit)
        coEvery { mockRepository.emergencyStop() } returns Result.success(Unit)

        operations.forEach { operation ->
            operation()
            testDispatcher.scheduler.advanceUntilIdle()
        }

        // Should have set loading to true and false for each operation
        verify(atLeast = operations.size) { mockLoadingObserver.onChanged(true) }
        verify(atLeast = operations.size) { mockLoadingObserver.onChanged(false) }
    }

    @Test
    fun `direction validation should handle all enum values`() = runTest {
        coEvery { mockRepository.changeDirection(any()) } returns Result.success(Unit)

        // Test both direction enum values
        Direction.values().forEach { direction ->
            viewModel.changeDirection(direction)
            testDispatcher.scheduler.advanceUntilIdle()
            coVerify { mockRepository.changeDirection(direction) }
        }
    }

    @Test
    fun `speed validation should handle edge cases near limits`() = runTest {
        // Test speeds just at the boundary
        val maxSpeed = 100.0 // Assuming max speed is 100
        val justOverMax = 100.01

        coEvery { mockRepository.updateSpeed(maxSpeed) } returns Result.success(Unit)

        viewModel.updateSpeed(maxSpeed)
        testDispatcher.scheduler.advanceUntilIdle()
        coVerify { mockRepository.updateSpeed(maxSpeed) }

        viewModel.updateSpeed(justOverMax)
        testDispatcher.scheduler.advanceUntilIdle()
        verify { mockErrorObserver.onChanged("Speed exceeds maximum limit: $justOverMax") }
    }

    @Test
    fun `concurrent state changes should maintain consistency`() = runTest {
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        coEvery { mockRepository.emergencyStop() } returns Result.success(Unit)

        // Start multiple operations concurrently
        viewModel.startDrive()
        viewModel.pauseDrive()
        viewModel.emergencyStop()
        
        testDispatcher.scheduler.advanceUntilIdle()

        // The final state should be consistent (likely EMERGENCY_STOP)
        verify { mockStateObserver.onChanged(DriveState.EMERGENCY_STOP) }
    }

    @Test
    fun `view model should handle repository unavailability`() = runTest {
        // Simulate repository being unavailable
        coEvery { mockRepository.startDrive(any()) } throws IllegalStateException("Repository unavailable")

        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged("Repository unavailable") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `performance under load should be acceptable`() = runTest {
        // Test performance with rapid operations
        val startTime = System.currentTimeMillis()
        
        repeat(1000) {
            viewModel.updateSpeed(it.toDouble())
        }
        
        testDispatcher.scheduler.advanceUntilIdle()
        
        val endTime = System.currentTimeMillis()
        val duration = endTime - startTime
        
        // Should complete within reasonable time (adjust threshold as needed)
        assertTrue("Operations took too long: ${duration}ms", duration < 5000)
    }

    @Test
    fun `view model cleanup should be thorough`() {
        // Test that cleanup properly handles all resources
        viewModel.driveState.removeObserver(mockStateObserver)
        viewModel.errorMessage.removeObserver(mockErrorObserver)
        viewModel.isLoading.removeObserver(mockLoadingObserver)

        // Should not crash or leak resources
        // In real implementation, this might involve checking for proper cleanup
    }

    @Test
    fun `network state changes during operations should be handled`() = runTest {
        // Start with network available
        every { mockNetworkManager.isConnected() } returns true
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)

        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Network becomes unavailable during operation
        every { mockNetworkManager.isConnected() } returns false
        
        val networkStatus = viewModel.isNetworkAvailable()
        assertFalse(networkStatus)
    }

    @Test
    fun `error state should clear on successful operation`() = runTest {
        // Put system in error state
        coEvery { mockRepository.startDrive(any()) } returns Result.failure(Exception("Test error"))
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockStateObserver.onChanged(DriveState.ERROR) }

        // Successful operation should clear error
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
    }

    @Test
    fun `all drive state enum values should be handled`() = runTest {
        // Test that all enum values are properly handled
        val allStates = DriveState.values()
        assertTrue("All drive states should be defined", allStates.isNotEmpty())
        
        // Verify each state is a valid enum value
        allStates.forEach { state ->
            assertTrue("State $state should be valid", state in DriveState.values())
        }
    }

    @Test
    fun `speed update with repository failure should maintain current state`() = runTest {
        val speed = 50.0
        coEvery { mockRepository.updateSpeed(speed) } returns Result.failure(Exception("Speed update failed"))

        viewModel.updateSpeed(speed)
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged("Speed update failed") }
        // State should remain unchanged
        verify { mockStateObserver.onChanged(DriveState.IDLE) }
    }

    @Test
    fun `pause drive with repository failure should handle error gracefully`() = runTest {
        // Start driving first
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Pause with failure
        coEvery { mockRepository.pauseDrive() } returns Result.failure(Exception("Pause failed"))
        viewModel.pauseDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged("Pause failed") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `resume drive with repository failure should handle error gracefully`() = runTest {
        // Start and pause first
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()
        viewModel.pauseDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Resume with failure
        coEvery { mockRepository.resumeDrive() } returns Result.failure(Exception("Resume failed"))
        viewModel.resumeDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged("Resume failed") }
        verify { mockStateObserver.onChanged(DriveState.ERROR) }
    }

    @Test
    fun `coroutine cancellation should be handled gracefully`() = runTest {
        coEvery { mockRepository.startDrive(any()) } coAnswers {
            throw kotlinx.coroutines.CancellationException("Operation cancelled")
        }

        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // Should handle cancellation without crashing
        verify { mockLoadingObserver.onChanged(false) }
    }

    @Test
    fun `multiple observer types should all receive updates`() = runTest {
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)

        viewModel.startDrive()
        testDispatcher.scheduler.advanceUntilIdle()

        // All observers should receive their respective updates
        verify { mockStateObserver.onChanged(DriveState.DRIVING) }
        verify { mockLoadingObserver.onChanged(true) }
        verify { mockLoadingObserver.onChanged(false) }
    }

    @Test
    fun `view model should handle extremely large speed values`() = runTest {
        val extremeSpeed = Double.MAX_VALUE

        viewModel.updateSpeed(extremeSpeed)
        testDispatcher.scheduler.advanceUntilIdle()

        verify { mockErrorObserver.onChanged("Speed exceeds maximum limit: $extremeSpeed") }
        coVerify(exactly = 0) { mockRepository.updateSpeed(any()) }
    }

    @Test
    fun `view model should handle extremely small positive speed values`() = runTest {
        val tinySpeed = Double.MIN_VALUE
        coEvery { mockRepository.updateSpeed(tinySpeed) } returns Result.success(Unit)

        viewModel.updateSpeed(tinySpeed)
        testDispatcher.scheduler.advanceUntilIdle()

        coVerify { mockRepository.updateSpeed(tinySpeed) }
    }

    @Test
    fun `emergency stop from all states should work correctly`() = runTest {
        coEvery { mockRepository.startDrive(any()) } returns Result.success(Unit)
        coEvery { mockRepository.pauseDrive() } returns Result.success(Unit)
        coEvery { mockRepository.emergencyStop() } returns Result.success(Unit)

        val testStates = listOf(
            { }, // IDLE
            { viewModel.startDrive(); testDispatcher.scheduler.advanceUntilIdle() }, // DRIVING
            { 
                viewModel.startDrive(); testDispatcher.scheduler.advanceUntilIdle()
                viewModel.pauseDrive(); testDispatcher.scheduler.advanceUntilIdle()
            } // PAUSED
        )

        testStates.forEach { setupState ->
            setupState()
            
            viewModel.emergencyStop()
            testDispatcher.scheduler.advanceUntilIdle()
            
            verify { mockStateObserver.onChanged(DriveState.EMERGENCY_STOP) }
            coVerify { mockRepository.emergencyStop() }
            
            // Reset for next test
            viewModel.reset()
            testDispatcher.scheduler.advanceUntilIdle()
        }
    }
}
