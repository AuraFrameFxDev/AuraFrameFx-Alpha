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
     * Updates the internal state to reflect that service binding is disabled. No actual service connection is performed.
     */
    fun bindService() {
        // Beta stub: No actual binding
        _isServiceConnected.value = false
        _status.value = "Service binding disabled in beta"
    }
    
    /**
     * Simulates unbinding from the Oracle Drive service in beta mode.
     *
     * Updates the internal state to reflect a disconnected service and sets a status message indicating beta mode. No real unbinding occurs.
     */
    fun unbindService() {
        // Beta stub: No actual unbinding
        _isServiceConnected.value = false
        _status.value = "Service unbound (beta mode)"
    }
    
    /**
     * Sets status, detailed status, and diagnostics log to static messages indicating Oracle Drive features are disabled in beta mode.
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
     * This method provides simulated feedback for UI testing in beta mode and does not alter any real module state.
     *
     * @param packageName The package name of the module to simulate toggling.
     * @param enable If `true`, simulates enabling the module; if `false`, simulates disabling it.
     */
    fun toggleModule(packageName: String, enable: Boolean) {
        // Beta stub: Just log the action without actually doing anything
        val action = if (enable) "enable" else "disable"
        _diagnosticsLog.value = "Beta Mode: Would $action module '$packageName' (no-op)"
    }
}
