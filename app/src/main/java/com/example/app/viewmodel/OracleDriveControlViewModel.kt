package com.example.app.viewmodel

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * ViewModel for Oracle Drive Control Screen
 *
 * Note: This is a stub implementation for beta. All methods are no-ops
 * and return placeholder data to prevent crashes during UI testing.
 */
class OracleDriveControlViewModel : ViewModel() {

    // State properties expected by the UI
    private val _isServiceConnected = MutableStateFlow(false)
    val isServiceConnected: StateFlow<Boolean> = _isServiceConnected.asStateFlow()

    private val _status = MutableStateFlow("Service not connected (Beta Mode)")
    val status: StateFlow<String> = _status.asStateFlow()

    private val _detailedStatus = MutableStateFlow("Oracle Drive is disabled in beta version")
    val detailedStatus: StateFlow<String> = _detailedStatus.asStateFlow()

    private val _diagnosticsLog = MutableStateFlow("Beta mode: No diagnostics available")
    val diagnosticsLog: StateFlow<String> = _diagnosticsLog.asStateFlow()

    /**
     * Simulates binding to the Oracle Drive service in beta mode.
     *
     * Updates internal state to indicate that service binding is disabled. Does not perform any real service connection.
     */
    fun bindService() {
        // Beta stub: No actual binding
        _isServiceConnected.value = false
        _status.value = "Service binding disabled in beta"
    }

    /**
     * Simulates unbinding from the Oracle Drive service by updating internal state to reflect a disconnected status.
     *
     * This method does not perform any real unbinding operation and is intended as a placeholder for beta mode.
     */
    fun unbindService() {
        // Beta stub: No actual unbinding
        _isServiceConnected.value = false
        _status.value = "Service unbound (beta mode)"
    }

    /**
     * Updates status, detailed status, and diagnostics log with placeholder messages indicating Oracle Drive is disabled in beta mode.
     *
     * This stub method is intended for UI testing and does not retrieve real status or diagnostics information.
     */
    fun refreshStatus() {
        // Beta stub: Update with fake status
        _status.value = "Beta Mode - Oracle Drive Status Simulation"
        _detailedStatus.value = "All Oracle Drive features are disabled in beta for safety"
        _diagnosticsLog.value = "Beta Log: No real diagnostics in this version"
    }

    /**
     * Simulates toggling a module by updating the diagnostics log with a placeholder message.
     *
     * This method does not affect any real module state and is intended for UI testing in beta mode.
     *
     * @param packageName The name of the module package to simulate toggling.
     * @param enable If `true`, simulates enabling the module; if `false`, simulates disabling it.
     */
    fun toggleModule(packageName: String, enable: Boolean) {
        // Beta stub: Just log the action without actually doing anything
        val action = if (enable) "enable" else "disable"
        _diagnosticsLog.value = "Beta Mode: Would $action module '$packageName' (no-op)"
    }
}
