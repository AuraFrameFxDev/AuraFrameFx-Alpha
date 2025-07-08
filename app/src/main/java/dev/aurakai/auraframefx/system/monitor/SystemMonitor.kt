package dev.aurakai.auraframefx.system.monitor

import android.app.ActivityManager
import android.content.Context
import android.os.Process
import dev.aurakai.auraframefx.utils.AuraFxLogger
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
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
     * Initiates periodic monitoring of system performance metrics if monitoring is not already active.
     *
     * Launches a background coroutine that updates CPU, memory, and network metrics at the specified interval.
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
     * The returned map includes CPU usage percentage, memory usage and available memory in bytes, memory usage percentage, network bytes received and transmitted, process ID, thread count, JVM heap size and usage, and a timestamp.
     *
     * @param component The identifier for the component associated with the collected metrics.
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
     * Calculates and returns a normalized score representing the current system health.
     *
     * The score ranges from 0.0 (poor health) to 1.0 (optimal health), based on the average of inverted CPU usage and available memory ratio (with a minimum threshold).
     *
     * @return A float value indicating overall system health, where higher values represent better health.
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
     * Generates a detailed report of the current system performance metrics.
     *
     * The report includes CPU usage, memory usage, available memory, memory usage percentage, network activity, system health score, stress status, process and thread counts, JVM heap size and usage, and a timestamp.
     *
     * @return A `SystemPerformanceReport` containing a snapshot of all monitored system metrics and status.
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
     * Updates CPU usage, memory usage, and network activity metrics asynchronously on the IO dispatcher.
     *
     * This function suspends while collecting the latest system performance data.
     */

    private suspend fun updateMetrics() = withContext(Dispatchers.IO) {
        updateCpuUsage()
        updateMemoryMetrics()
        updateNetworkMetrics()
    }

    /**
     * Updates the CPU usage metric by recalculating and storing the latest value.
     *
     * If CPU usage calculation fails, the previous value is retained.
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
     * Updates the available and used memory metrics by retrieving current memory information from the system.
     *
     * Queries the Android ActivityManager for memory statistics and updates the internal state flows for available and used memory. If retrieval fails, the previous metric values are retained.
     */
    private fun updateMemoryMetrics() {
        try {
            val activityManager =
                context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val memoryInfo = ActivityManager.MemoryInfo()
            activityManager.getMemoryInfo(memoryInfo)

            _availableMemory.value = memoryInfo.availMem
            _memoryUsage.value = memoryInfo.totalMem - memoryInfo.availMem
        } catch (e: Exception) {
            logger.warn("SystemMonitor", "Failed to update memory metrics", e)
        }
    }

    /**
     * Sets the network metrics state flow to zero values as a placeholder.
     *
     * Intended for future implementation of actual network monitoring.
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
     * Generates a simulated CPU usage percentage as a random float between 0 and 100.
     *
     * This is a placeholder implementation and does not reflect actual system CPU usage.
     *
     * @return A random float representing simulated CPU usage percentage.
     */
    private fun calculateCpuUsage(): Float {
        // Simplified CPU usage calculation
        // In production, this would use /proc/stat or other system calls
        return kotlin.random.Random.nextFloat() * 100f // Placeholder
    }

    /**
     * Returns the current memory usage as a percentage of total system memory.
     *
     * If total memory cannot be determined, returns 0f.
     *
     * @return The percentage of memory currently used.
     */
    private fun calculateMemoryUsagePercent(): Float {
        val total = getTotalMemory()
        return if (total > 0) {
            (_memoryUsage.value.toFloat() / total) * 100f
        } else 0f
    }

    /**
     * Returns the total physical memory of the device in bytes.
     *
     * @return Total system memory in bytes.
     */
    private fun getTotalMemory(): Long {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        return memoryInfo.totalMem
    }

    /**
     * Retrieves the current number of active threads in the JVM.
     *
     * @return The number of active threads.
     */
    private fun getThreadCount(): Int {
        return Thread.activeCount()
    }

    /**
     * Retrieves the maximum heap memory available to the JVM in bytes.
     *
     * @return The maximum heap size in bytes.
     */
    private fun getHeapSize(): Long {
        return Runtime.getRuntime().maxMemory()
    }

    /**
     * Retrieves the current amount of heap memory used by the JVM in bytes.
     *
     * @return The number of bytes of heap memory currently in use.
     */
    private fun getUsedHeap(): Long {
        val runtime = Runtime.getRuntime()
        return runtime.totalMemory() - runtime.freeMemory()
    }

    /**
     * Stops monitoring and cancels internal resources used by the system monitor.
     *
     * Halts all ongoing monitoring activities and releases the coroutine scope to free resources.
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
