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
     * Simulates a service bind operation by updating the connection state to reflect that binding is disabled in beta mode.
     *
     * No actual service connection is performed; this method only updates internal state for UI testing purposes.
     */
    fun bindService() {
        // Beta stub: No actual binding
        _isServiceConnected.value = false
        _status.value = "Service binding disabled in beta"
    }

    /**
     * Simulates unbinding from the Oracle Drive service for UI testing in beta mode.
     *
     * Sets the service connection state to disconnected and updates the status message with a placeholder, without performing any real unbinding operation.
     */
    fun unbindService() {
        // Beta stub: No actual unbinding
        _isServiceConnected.value = false
        _status.value = "Service unbound (beta mode)"
    }

    /**
     * Sets status, detailed status, and diagnostics log to placeholder messages indicating Oracle Drive features are disabled in beta mode.
     *
     * Intended as a no-op for UI testing; does not perform any real status or diagnostics operations.
     */
    fun refreshStatus() {
        // Beta stub: Update with fake status
        _status.value = "Beta Mode - Oracle Drive Status Simulation"
        _detailedStatus.value = "All Oracle Drive features are disabled in beta for safety"
        _diagnosticsLog.value = "Beta Log: No real diagnostics in this version"
    }

    /**
     * Updates the diagnostics log with a simulated message indicating a module enable or disable action.
     *
     * Intended for beta mode UI testing; does not perform any real module operations.
     *
     * @param packageName The module package name to display in the simulated action message.
     * @param enable If `true`, simulates enabling the module; if `false`, simulates disabling it.
     */
    fun toggleModule(packageName: String, enable: Boolean) {
        // Beta stub: Just log the action without actually doing anything
        val action = if (enable) "enable" else "disable"
        _diagnosticsLog.value = "Beta Mode: Would $action module '$packageName' (no-op)"
    }
}
