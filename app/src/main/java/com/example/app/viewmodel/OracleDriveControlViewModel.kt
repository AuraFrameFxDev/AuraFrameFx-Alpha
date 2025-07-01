package com.example.app.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.app.ipc.OracleDriveServiceConnector
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class OracleDriveControlViewModel @Inject constructor(
    private val oracleDriveServiceConnector: OracleDriveServiceConnector,
) : ViewModel() {
    private val _status = MutableStateFlow<String?>(null)
    val status: StateFlow<String?> = _status.asStateFlow()

    private val _detailedStatus = MutableStateFlow<String?>(null)
    val detailedStatus: StateFlow<String?> = _detailedStatus.asStateFlow()

    private val _diagnosticsLog = MutableStateFlow<String?>(null)
    val diagnosticsLog: StateFlow<String?> = _diagnosticsLog.asStateFlow()

    val isServiceConnected: StateFlow<Boolean> =
        oracleDriveServiceConnector.isServiceConnected

    fun bindService() {
        oracleDriveServiceConnector.bindService()
    }

    fun unbindService() {
        oracleDriveServiceConnector.unbindService()
    }

    fun refreshStatus() {
        viewModelScope.launch {
            _status.value = oracleDriveServiceConnector.getStatusFromOracleDrive()
            _detailedStatus.value = oracleDriveServiceConnector.getDetailedInternalStatus()
            _diagnosticsLog.value = oracleDriveServiceConnector.getInternalDiagnosticsLog()
        }
    }

    fun toggleModule(packageName: String, enable: Boolean) {
        viewModelScope.launch {
            oracleDriveServiceConnector.toggleModuleOnOracleDrive(packageName, enable)
            refreshStatus()
        }
    }

    override fun onCleared() {
        super.onCleared()
        unbindService()
    }
}
