package dev.aurakai.auraframefx.system.monitor

import android.content.Context
import android.app.ActivityManager
import android.os.Debug
import android.os.Process
import dev.aurakai.auraframefx.utils.AuraFxLogger
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import javax.inject.Inject
import javax.inject.Singleton

/**
 * SystemMonitor provides comprehensive system performance monitoring.
 * Supports Kai's analytical capabilities with real-time system metrics.
 */
@Singleton
class SystemMonitor @Inject constructor(
    private val context: Context,
    private val logger: AuraFxLogger
) {
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var isMonitoring = false
    
    // Performance metrics
    private val _cpuUsage = MutableStateFlow(0.0f)
    val cpuUsage: StateFlow<Float> = _cpuUsage
    
    private val _memoryUsage = MutableStateFlow(0L)
    val memoryUsage: StateFlow<Long> = _memoryUsage
    
    private val _availableMemory = MutableStateFlow(0L)
    val availableMemory: StateFlow<Long> = _availableMemory
    
    private val _networkActivity = MutableStateFlow(NetworkMetrics())
    val networkActivity: StateFlow<NetworkMetrics> = _networkActivity

    /**
     * Initiates periodic system performance monitoring, updating metrics at the specified interval.
     *
     * Has no effect if monitoring is already active.
     *
     * @param intervalMs The interval in milliseconds between metric updates. Defaults to 5000 ms.
     */
    fun startMonitoring(intervalMs: Long = 5000) {
        if (isMonitoring) return
        
        logger.info("SystemMonitor", "Starting system performance monitoring")
        isMonitoring = true
        
        scope.launch {
            while (isMonitoring) {
                try {
                    updateMetrics()
                    delay(intervalMs)
                } catch (e: Exception) {
                    logger.error("SystemMonitor", "Error updating metrics", e)
                    delay(intervalMs * 2) // Longer delay on error
                }
            }
        }
    }

    /**
     * Stops the system performance monitoring process.
     *
     * Disables periodic metric updates by setting the monitoring flag to false.
     */
    fun stopMonitoring() {
        logger.info("SystemMonitor", "Stopping system performance monitoring")
        isMonitoring = false
    }

    /**
     * Returns a map containing the current system performance metrics for the specified component.
     *
     * The map includes CPU usage percentage, memory usage and available memory in bytes, memory usage percentage, network bytes received and transmitted, process ID, thread count, JVM heap size and used heap in bytes, and a timestamp.
     *
     * @param component The name or identifier of the component for which metrics are retrieved.
     * @return A map where each key is a metric name and each value is the current reading for that metric.
     */
    fun getPerformanceMetrics(component: String): Map<String, Any> {
        logger.debug("SystemMonitor", "Getting performance metrics for: $component")
        
        return mapOf(
            "component" to component,
            "cpu_usage_percent" to _cpuUsage.value,
            "memory_usage_bytes" to _memoryUsage.value,
            "available_memory_bytes" to _availableMemory.value,
            "memory_usage_percent" to calculateMemoryUsagePercent(),
            "network_rx_bytes" to _networkActivity.value.receivedBytes,
            "network_tx_bytes" to _networkActivity.value.transmittedBytes,
            "process_id" to Process.myPid(),
            "thread_count" to getThreadCount(),
            "heap_size_bytes" to getHeapSize(),
            "heap_used_bytes" to getUsedHeap(),
            "timestamp" to System.currentTimeMillis()
        )
    }

    /**
     * Calculates a system health score between 0.0 and 1.0 based on current CPU usage and available memory.
     *
     * The score is the average of the inverse CPU usage ratio and the available memory ratio, with the memory ratio floored at 0.1 to prevent extremely low values. A score of 1.0 indicates optimal health, while 0.0 indicates poor health.
     *
     * @return The computed system health score.
     */
    fun getSystemHealthScore(): Float {
        val cpuScore = 1.0f - (_cpuUsage.value / 100f).coerceAtMost(1.0f)
        val memoryScore = (_availableMemory.value.toFloat() / getTotalMemory()).coerceAtLeast(0.1f)
        
        return (cpuScore + memoryScore) / 2.0f
    }

    /**
     * Determines whether the system is currently under stress based on CPU usage, memory usage percentage, or available memory.
     *
     * The system is considered under stress if any of the following conditions are met:
     * - CPU usage exceeds 80%
     * - Memory usage percentage exceeds 85%
     * - Available memory is less than 50 MB
     *
     * @return `true` if the system is under stress; otherwise, `false`.
     */
    fun isSystemUnderStress(): Boolean {
        return _cpuUsage.value > 80f || 
               calculateMemoryUsagePercent() > 85f ||
               _availableMemory.value < (50 * 1024 * 1024) // Less than 50MB available
    }

    /**
     * Returns a snapshot of current system performance metrics.
     *
     * The report includes CPU usage, memory usage, available memory, memory usage percentage, network activity, system health score, stress status, process ID, thread count, JVM heap size, used heap memory, and a timestamp.
     *
     * @return A [SystemPerformanceReport] containing the latest system performance metrics.
     */
    fun getPerformanceReport(): SystemPerformanceReport {
        return SystemPerformanceReport(
            timestamp = System.currentTimeMillis(),
            cpuUsagePercent = _cpuUsage.value,
            memoryUsageBytes = _memoryUsage.value,
            availableMemoryBytes = _availableMemory.value,
            memoryUsagePercent = calculateMemoryUsagePercent(),
            networkMetrics = _networkActivity.value,
            systemHealthScore = getSystemHealthScore(),
            isUnderStress = isSystemUnderStress(),
            processId = Process.myPid(),
            threadCount = getThreadCount(),
            heapSizeBytes = getHeapSize(),
            heapUsedBytes = getUsedHeap()
        )
    }

    /**
     * Refreshes CPU usage, memory usage, and network activity metrics asynchronously on the IO dispatcher.
     *
     * Intended for periodic invocation to update all monitored system metrics.
     */

    private suspend fun updateMetrics() = withContext(Dispatchers.IO) {
        updateCpuUsage()
        updateMemoryMetrics()
        updateNetworkMetrics()
    }

    /**
     * Updates the current CPU usage metric.
     *
     * Attempts to calculate the CPU usage and update the corresponding state. If calculation fails, logs a warning and leaves the previous value unchanged.
     */
    private fun updateCpuUsage() {
        try {
            // Get CPU usage - simplified implementation
            val usage = calculateCpuUsage()
            _cpuUsage.value = usage
        } catch (e: Exception) {
            logger.warn("SystemMonitor", "Failed to update CPU usage", e)
        }
    }

    /**
     * Updates the internal state with the current available and used memory metrics.
     *
     * Retrieves memory information from the Android system and updates the corresponding state flows. If memory information cannot be obtained, logs a warning and retains previous values.
     */
    private fun updateMemoryMetrics() {
        try {
            val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val memoryInfo = ActivityManager.MemoryInfo()
            activityManager.getMemoryInfo(memoryInfo)
            
            _availableMemory.value = memoryInfo.availMem
            _memoryUsage.value = memoryInfo.totalMem - memoryInfo.availMem
        } catch (e: Exception) {
            logger.warn("SystemMonitor", "Failed to update memory metrics", e)
        }
    }

    /**
     * Updates the network activity metrics with placeholder zero values.
     *
     * Currently sets all network metrics to zero, as actual network monitoring is not implemented.
     */
    private fun updateNetworkMetrics() {
        try {
            // Network metrics would require additional permissions and implementation
            // Placeholder for now
            _networkActivity.value = NetworkMetrics(
                receivedBytes = 0L,
                transmittedBytes = 0L,
                receivedPackets = 0L,
                transmittedPackets = 0L
            )
        } catch (e: Exception) {
            logger.warn("SystemMonitor", "Failed to update network metrics", e)
        }
    }

    /**
     * Returns a simulated CPU usage percentage as a random float between 0.0 and 100.0.
     *
     * This function serves as a placeholder for actual CPU usage calculation logic.
     *
     * @return Simulated CPU usage percentage.
     */
    private fun calculateCpuUsage(): Float {
        // Simplified CPU usage calculation
        // In production, this would use /proc/stat or other system calls
        return kotlin.random.Random.nextFloat() * 100f // Placeholder
    }

    /**
     * Calculates the percentage of system memory currently in use.
     *
     * @return The memory usage percentage as a float, or 0.0 if total memory is unavailable.
     */
    private fun calculateMemoryUsagePercent(): Float {
        val total = getTotalMemory()
        return if (total > 0) {
            (_memoryUsage.value.toFloat() / total) * 100f
        } else 0f
    }

    /**
     * Retrieves the total physical memory (RAM) of the device in bytes.
     *
     * Uses the Android `ActivityManager` to obtain the total system memory available.
     *
     * @return The total system memory in bytes.
     */
    private fun getTotalMemory(): Long {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        return memoryInfo.totalMem
    }

    /**
     * Returns the current number of active threads in the JVM process.
     *
     * @return The number of active threads.
     */
    private fun getThreadCount(): Int {
        return Thread.activeCount()
    }

    /**
     * Returns the maximum heap memory available to the JVM in bytes.
     *
     * @return The maximum heap size in bytes.
     */
    private fun getHeapSize(): Long {
        return Runtime.getRuntime().maxMemory()
    }

    /**
     * Returns the current heap memory usage of the JVM in bytes.
     *
     * @return The number of bytes of heap memory currently used by the JVM.
     */
    private fun getUsedHeap(): Long {
        val runtime = Runtime.getRuntime()
        return runtime.totalMemory() - runtime.freeMemory()
    }

    /**
     * Stops system monitoring and cancels the coroutine scope to release resources.
     *
     * Halts all periodic metric updates and ensures that no background monitoring tasks remain active.
     */
    fun cleanup() {
        logger.info("SystemMonitor", "Cleaning up SystemMonitor")
        stopMonitoring()
        scope.cancel()
    }
}

/**
 * Network performance metrics.
 */
data class NetworkMetrics(
    val receivedBytes: Long = 0L,
    val transmittedBytes: Long = 0L,
    val receivedPackets: Long = 0L,
    val transmittedPackets: Long = 0L
)

/**
 * Comprehensive system performance report.
 */
data class SystemPerformanceReport(
    val timestamp: Long,
    val cpuUsagePercent: Float,
    val memoryUsageBytes: Long,
    val availableMemoryBytes: Long,
    val memoryUsagePercent: Float,
    val networkMetrics: NetworkMetrics,
    val systemHealthScore: Float,
    val isUnderStress: Boolean,
    val processId: Int,
    val threadCount: Int,
    val heapSizeBytes: Long,
    val heapUsedBytes: Long
)
