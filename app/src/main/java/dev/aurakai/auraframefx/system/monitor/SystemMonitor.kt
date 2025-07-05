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
     * Begins periodic system performance monitoring, updating metrics at the specified interval.
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
     * Sets the monitoring flag to false, halting periodic metric updates.
     */
    fun stopMonitoring() {
        logger.info("SystemMonitor", "Stopping system performance monitoring")
        isMonitoring = false
    }

    /**
     * Returns a map containing current system performance metrics for the specified component.
     *
     * The returned map includes CPU usage percentage, memory usage in bytes, available memory, memory usage percentage, network activity (received and transmitted bytes), process ID, thread count, heap size, heap used, and a timestamp.
     *
     * @param component The name or identifier of the component for which metrics are being retrieved.
     * @return A map of performance metric names to their current values.
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
     * Calculates and returns a system health score between 0.0 and 1.0 based on current CPU usage and available memory.
     *
     * The score is an average of the inverse CPU usage ratio and the available memory ratio, providing a simple indicator of overall system health.
     *
     * @return The system health score, where 1.0 indicates optimal health and 0.0 indicates poor health.
     */
    fun getSystemHealthScore(): Float {
        val cpuScore = 1.0f - (_cpuUsage.value / 100f).coerceAtMost(1.0f)
        val memoryScore = (_availableMemory.value.toFloat() / getTotalMemory()).coerceAtLeast(0.1f)
        
        return (cpuScore + memoryScore) / 2.0f
    }

    /**
     * Determines whether the system is currently under stress based on CPU usage, memory usage percentage, or available memory thresholds.
     *
     * @return `true` if CPU usage exceeds 80%, memory usage exceeds 85%, or available memory is less than 50MB; otherwise, `false`.
     */
    fun isSystemUnderStress(): Boolean {
        return _cpuUsage.value > 80f || 
               calculateMemoryUsagePercent() > 85f ||
               _availableMemory.value < (50 * 1024 * 1024) // Less than 50MB available
    }

    /**
     * Returns a comprehensive snapshot of current system performance metrics.
     *
     * The report includes CPU usage, memory usage, available memory, memory usage percentage, network activity, system health score, stress status, process ID, thread count, heap size, and used heap memory, along with a timestamp.
     *
     * @return A `SystemPerformanceReport` containing the latest system performance data.
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
     * Updates all monitored system metrics, including CPU usage, memory usage, and network activity, on the IO dispatcher.
     */

    private suspend fun updateMetrics() = withContext(Dispatchers.IO) {
        updateCpuUsage()
        updateMemoryMetrics()
        updateNetworkMetrics()
    }

    /**
     * Updates the current CPU usage metric by calculating and assigning a new value.
     *
     * If an error occurs during calculation, logs a warning and leaves the previous value unchanged.
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
     * Retrieves memory statistics using the Android ActivityManager and updates the corresponding state flows.
     * Logs a warning if memory information cannot be obtained.
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
     * Updates the network activity metrics with placeholder values.
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
     * Returns a placeholder value representing the current CPU usage percentage.
     *
     * This implementation generates a random float between 0 and 100 as a stub.
     * In a production environment, this should be replaced with actual CPU usage calculation logic.
     *
     * @return Simulated CPU usage as a percentage (0.0 to 100.0).
     */
    private fun calculateCpuUsage(): Float {
        // Simplified CPU usage calculation
        // In production, this would use /proc/stat or other system calls
        return kotlin.random.Random.nextFloat() * 100f // Placeholder
    }

    /**
     * Calculates the current memory usage as a percentage of total system memory.
     *
     * @return The percentage of memory currently used, or 0.0 if total memory is unavailable.
     */
    private fun calculateMemoryUsagePercent(): Float {
        val total = getTotalMemory()
        return if (total > 0) {
            (_memoryUsage.value.toFloat() / total) * 100f
        } else 0f
    }

    /**
     * Retrieves the total physical memory available on the device in bytes.
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
     * Returns the current number of active threads in the JVM.
     *
     * @return The active thread count.
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
     * Returns the amount of heap memory currently used by the JVM, in bytes.
     *
     * @return The number of bytes of heap memory in use.
     */
    private fun getUsedHeap(): Long {
        val runtime = Runtime.getRuntime()
        return runtime.totalMemory() - runtime.freeMemory()
    }

    /**
     * Releases resources used by the system monitor and stops ongoing monitoring.
     *
     * Cancels the monitoring coroutine scope and halts all metric updates.
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
