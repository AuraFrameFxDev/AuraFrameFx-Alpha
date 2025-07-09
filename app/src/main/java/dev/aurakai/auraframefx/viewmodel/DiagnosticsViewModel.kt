package dev.aurakai.auraframefx.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import dev.aurakai.auraframefx.data.logging.AuraFxLogger
import dev.aurakai.auraframefx.data.offline.OfflineDataManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*
import javax.inject.Inject

@HiltViewModel
class DiagnosticsViewModel @Inject constructor(
    private val auraFxLogger: AuraFxLogger,
    private val offlineDataManager: OfflineDataManager
) : ViewModel() {

    private val _diagnosticsState = MutableStateFlow(DiagnosticsState())
    val diagnosticsState: StateFlow<DiagnosticsState> = _diagnosticsState.asStateFlow()

    private val _systemStatus = MutableStateFlow<Map<String, String>>(emptyMap())
    val systemStatus: StateFlow<Map<String, String>> = _systemStatus.asStateFlow()

    init {
        loadSystemStatus()
    }

    /**
     * Loads system status information including offline data status.
     */
    private fun loadSystemStatus() {
        viewModelScope.launch {
            try {
                val offlineData = offlineDataManager.loadCriticalOfflineData()

                _systemStatus.value = buildMap {
                    put("System Status", "Online")
                    put("Logger Status", "Active")
                    put(
                        "Last Full Sync",
                        if (offlineData?.lastFullSyncTimestamp != null && offlineData.lastFullSyncTimestamp != 0L) {
                            SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(
                                Date(offlineData.lastFullSyncTimestamp)
                            )
                        } else {
                            "N/A"
                        }
                    )
                    put("Offline Data Status", if (offlineData != null) "Available" else "Not Available")
                    // Add more status items as needed
                }
            } catch (e: Exception) {
                auraFxLogger.e("DiagnosticsVM", "Failed to load system status: ${e.message}")
                _systemStatus.value = mapOf(
                    "System Status" to "Error",
                    "Error" to (e.message ?: "Unknown error")
                )
            }
        }
    }

    /**
     * Retrieves logs for a specific date with error handling.
     */
    suspend fun getLogsForDate(date: String): List<String> {
        return try {
            auraFxLogger.getLogsForDate(date)
        } catch (e: Exception) {
            auraFxLogger.e("DiagnosticsVM", "Failed to get logs for date $date: ${e.message}")
            listOf("Error retrieving logs for $date: ${e.message}")
        }
    }

    /**
     * Clears all logs with error handling.
     */
    suspend fun clearAllLogs(): Boolean {
        return try {
            auraFxLogger.clearAllLogs()
            true
        } catch (e: Exception) {
            auraFxLogger.e("DiagnosticsVM", "Failed to clear logs: ${e.message}")
            false
        }
    }

    /**
     * Retrieves all logs and converts Map to List format.
     */
    suspend fun getAllLogs(): List<String> {
        return try {
            val logsMap = auraFxLogger.getAllLogs()
            logsMap.values.flatMap { content ->
                content.split("\n").filter { it.isNotBlank() }
            }.take(500) // Limit to 500 lines
        } catch (e: Exception) {
            auraFxLogger.e("DiagnosticsVM", "Failed to get all logs: ${e.message}")
            listOf("Error retrieving all logs: ${e.message}")
        }
    }

    /**
     * Filters logs by severity level.
     */
    suspend fun getLogsByLevel(level: String): List<String> {
        return try {
            val logsMap = auraFxLogger.getAllLogs()
            val allLogLines = logsMap.values.flatMap { content ->
                content.split("\n").filter { it.isNotBlank() }
            }
            allLogLines.filter { log ->
                log.contains("[$level]", ignoreCase = true)
            }
        } catch (e: Exception) {
            auraFxLogger.e("DiagnosticsVM", "Failed to filter logs by level: ${e.message}")
            listOf("Error filtering logs: ${e.message}")
        }
    }

    /**
     * Loads and displays detailed configuration from offline data manager.
     */
    suspend fun loadDetailedConfig() {
        try {
            val offlineData = offlineDataManager.loadCriticalOfflineData()

            _diagnosticsState.value = _diagnosticsState.value.copy(
                configLoaded = true,
                lastSyncTime = offlineData?.lastFullSyncTimestamp ?: 0L
            )
        } catch (e: Exception) {
            auraFxLogger.e("DiagnosticsVM", "Failed to load detailed config: ${e.message}")
            _diagnosticsState.value = _diagnosticsState.value.copy(
                configLoaded = false,
                error = e.message
            )
        }
    }

    /**
     * Refreshes all diagnostic data.
     */
    fun refreshDiagnostics() {
        viewModelScope.launch {
            loadSystemStatus()
            loadDetailedConfig()
        }
    }
}

data class DiagnosticsState(
    val configLoaded: Boolean = false,
    val lastSyncTime: Long = 0L,
    val error: String? = null
)
