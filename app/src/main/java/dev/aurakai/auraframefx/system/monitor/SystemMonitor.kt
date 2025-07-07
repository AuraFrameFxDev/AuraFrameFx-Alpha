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
     * Starts periodic system performance monitoring in a background coroutine.
     *
     * If monitoring is not already active, this method begins updating system metrics at the specified interval.
     *
     * @param intervalMs Interval in milliseconds between metric updates. Defaults to 5000 ms.
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
     * Halts periodic metric updates by disabling the monitoring flag.
     */
    fun stopMonitoring() {
        logger.info("SystemMonitor", "Stopping system performance monitoring")
        isMonitoring = false
    }

    /**
     * Retrieves a map of current system performance metrics for the specified component.
     *
     * The map includes CPU usage percentage, memory usage and availability in bytes, memory usage percentage,
     * network bytes received and transmitted, process ID, thread count, JVM heap size and usage, and a timestamp.
     *
     * @param component The identifier for the component associated with these metrics.
     * @return A map where keys are metric names and values are their current readings.
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
     * Computes a normalized system health score based on current CPU usage and available memory.
     *
     * The score ranges from 0.0 (poor health) to 1.0 (optimal health), calculated as the average of an inverted, capped CPU usage score and a minimum-threshold available memory ratio.
     *
     * @return The system health score, where higher values indicate better overall system performance.
     */
    fun getSystemHealthScore(): Float {
        val cpuScore = 1.0f - (_cpuUsage.value / 100f).coerceAtMost(1.0f)
        val memoryScore = (_availableMemory.value.toFloat() / getTotalMemory()).coerceAtLeast(0.1f)
        
        return (cpuScore + memoryScore) / 2.0f
    }

    /**
     * Returns `true` if the system is under stress due to high CPU usage, high memory usage, or low available memory.
     *
     * The system is considered under stress if CPU usage exceeds 80%, memory usage percentage exceeds 85%, or available memory falls below 50 MB.
     *
     * @return `true` if any stress condition is met; otherwise, `false`.
     */
    fun isSystemUnderStress(): Boolean {
        return _cpuUsage.value > 80f || 
               calculateMemoryUsagePercent() > 85f ||
               _availableMemory.value < (50 * 1024 * 1024) // Less than 50MB available
    }

    /**
     * Generates a detailed report of the current system performance metrics.
     *
     * The report includes CPU usage, memory usage, available memory, memory usage percentage, network activity, system health score, stress status, process and thread information, JVM heap size and usage, and a timestamp.
     *
     * @return A `SystemPerformanceReport` containing the latest system metrics and status.
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
     */

    private suspend fun updateMetrics() = withContext(Dispatchers.IO) {
        updateCpuUsage()
        updateMemoryMetrics()
        updateNetworkMetrics()
    }

    /**
     * Calculates and updates the current CPU usage metric.
     *
     * If CPU usage calculation fails, logs a warning and leaves the previous value unchanged.
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
     * Retrieves current memory information from the system and updates available and used memory metrics.
     *
     * If retrieval fails, logs a warning and retains previous metric values.
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
     * This is a placeholder implementation and should be replaced with actual CPU usage calculation in production environments.
     *
     * @return A random float representing simulated CPU usage percentage.
     */
    private fun calculateCpuUsage(): Float {
        // Simplified CPU usage calculation
        // In production, this would use /proc/stat or other system calls
        return kotlin.random.Random.nextFloat() * 100f // Placeholder
    }

    /**
     * Returns the percentage of system memory currently in use.
     *
     * @return Memory usage as a percentage of total memory, or 0f if total memory cannot be determined.
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
     * @return The total system memory in bytes.
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
     * Terminates ongoing monitoring activities and releases the coroutine scope to free associated resources.
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
