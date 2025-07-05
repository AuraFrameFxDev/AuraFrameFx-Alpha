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
     * Start system monitoring with periodic updates.
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
     * Stop system monitoring.
     */
    fun stopMonitoring() {
        logger.info("SystemMonitor", "Stopping system performance monitoring")
        isMonitoring = false
    }

    /**
     * Get comprehensive performance metrics for a component.
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
     * Get current system health score (0.0 to 1.0).
     */
    fun getSystemHealthScore(): Float {
        val cpuScore = 1.0f - (_cpuUsage.value / 100f).coerceAtMost(1.0f)
        val memoryScore = (_availableMemory.value.toFloat() / getTotalMemory()).coerceAtLeast(0.1f)
        
        return (cpuScore + memoryScore) / 2.0f
    }

    /**
     * Check if system is under stress.
     */
    fun isSystemUnderStress(): Boolean {
        return _cpuUsage.value > 80f || 
               calculateMemoryUsagePercent() > 85f ||
               _availableMemory.value < (50 * 1024 * 1024) // Less than 50MB available
    }

    /**
     * Get detailed performance report.
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

    // Private monitoring methods

    private suspend fun updateMetrics() = withContext(Dispatchers.IO) {
        updateCpuUsage()
        updateMemoryMetrics()
        updateNetworkMetrics()
    }

    private fun updateCpuUsage() {
        try {
            // Get CPU usage - simplified implementation
            val usage = calculateCpuUsage()
            _cpuUsage.value = usage
        } catch (e: Exception) {
            logger.warn("SystemMonitor", "Failed to update CPU usage", e)
        }
    }

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

    private fun calculateCpuUsage(): Float {
        // Simplified CPU usage calculation
        // In production, this would use /proc/stat or other system calls
        return kotlin.random.Random.nextFloat() * 100f // Placeholder
    }

    private fun calculateMemoryUsagePercent(): Float {
        val total = getTotalMemory()
        return if (total > 0) {
            (_memoryUsage.value.toFloat() / total) * 100f
        } else 0f
    }

    private fun getTotalMemory(): Long {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        return memoryInfo.totalMem
    }

    private fun getThreadCount(): Int {
        return Thread.activeCount()
    }

    private fun getHeapSize(): Long {
        return Runtime.getRuntime().maxMemory()
    }

    private fun getUsedHeap(): Long {
        val runtime = Runtime.getRuntime()
        return runtime.totalMemory() - runtime.freeMemory()
    }

    /**
     * Cleanup monitoring resources.
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
