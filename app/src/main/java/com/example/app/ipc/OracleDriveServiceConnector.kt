package com.example.app.ipc

// Explicitly import the AIDL interface
import com.example.app.ipc.IAuraDriveService
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.os.RemoteException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext

class OracleDriveServiceConnector(private val context: Context) {
    private var auraDriveService: IAuraDriveService? = null
    private val _isServiceConnected = MutableStateFlow(false)
    val isServiceConnected: StateFlow<Boolean> = _isServiceConnected.asStateFlow()

    private val serviceConnection = object : ServiceConnection {
        /****
         * Handles successful connection to the AuraDrive service by assigning the remote interface and updating the connection state.
         *
         * @param name The component name of the connected service.
         * @param service The binder interface to the connected service.
         */
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            auraDriveService = IAuraDriveService.Companion.Stub.asInterface(service)
            _isServiceConnected.value = true
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            auraDriveService = null
            _isServiceConnected.value = false
        }
    }

    fun bindService() {
        val intent = Intent().apply {
            component = ComponentName(
                "com.genesis.ai.app",
                "com.genesis.ai.app.service.AuraDriveServiceImpl"
            )
        }
        try {
            context.bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
        } catch (e: SecurityException) {
            _isServiceConnected.value = false
        }
    }

    fun unbindService() {
        try {
            context.unbindService(serviceConnection)
        } catch (_: Exception) {
        }
        auraDriveService = null
        _isServiceConnected.value = false
    }

    /**
     * Retrieves the current status from the remote OracleDrive service.
     *
     * Executes the status request on the IO dispatcher. Returns the status string if available, or null if the service is unavailable or a remote exception occurs.
     *
     * @return The status string from OracleDrive, or null if unavailable.
     */
    suspend fun getStatusFromOracleDrive(): String? = withContext(Dispatchers.IO) {
        try {
            auraDriveService?.getOracleDriveStatus()
        } catch (e: RemoteException) {
            null
        }
    }

    /**
         * Toggles the LSPosed module on the connected Oracle Drive service.
         *
         * The parameters `packageName` and `enable` are currently unused.
         *
         * @return "Success" if the module was toggled successfully, "Failed" if the operation failed, or null if a remote exception occurred or the service is unavailable.
         */
        suspend fun toggleModuleOnOracleDrive(packageName: String, enable: Boolean): String? =
        withContext(Dispatchers.IO) {
            try {
                val result = auraDriveService?.toggleLSPosedModule()
                if (result == true) "Success" else "Failed"
            } catch (e: RemoteException) {
                null
            }
        }

    /**
     * Retrieves a detailed internal status report from the remote AuraDrive service.
     *
     * @return The detailed status string, or null if the service is unavailable or a remote exception occurs.
     */
    suspend fun getDetailedInternalStatus(): String? = withContext(Dispatchers.IO) {
        try {
            auraDriveService?.getDetailedInternalStatus()
        } catch (e: RemoteException) {
            null
        }
    }

    /**
     * Retrieves the internal diagnostics log from the remote AuraDrive service as a single newline-separated string.
     *
     * @return The diagnostics log as a single string, or null if the service is unavailable or a remote exception occurs.
     */
    suspend fun getInternalDiagnosticsLog(): String? = withContext(Dispatchers.IO) {
        try {
            val logs = auraDriveService?.getInternalDiagnosticsLog()
            logs?.joinToString("\n")
        } catch (e: RemoteException) {
            null
        }
    }
}
