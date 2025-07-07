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
        /**
         * Called when the AuraDrive service is connected.
         *
         * Initializes the AuraDrive AIDL interface and updates the connection status to connected.
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

    /**
     * Unbinds from the AuraDrive service and updates the connection state.
     *
     * Safely attempts to unbind from the service, clears the service interface reference,
     * and marks the service as disconnected. Any exceptions during unbinding are ignored.
     */
    fun unbindService() {
        try {
            context.unbindService(serviceConnection)
        } catch (_: Exception) {
        }
        auraDriveService = null
        _isServiceConnected.value = false
    }

    /**
     * Returns the current status string from the AuraDrive service, or null if unavailable or on communication error.
     *
     * Executes the status retrieval on the IO dispatcher and handles RemoteException by returning null.
     *
     * @return The current status from the AuraDrive service, or null if the service is not connected or a RemoteException occurs.
     */
    suspend fun getStatusFromOracleDrive(): String? = withContext(Dispatchers.IO) {
        try {
            auraDriveService?.getOracleDriveStatus()
        } catch (e: RemoteException) {
            null
        }
    }

    /**
         * Requests the AuraDrive service to toggle the LSPosed module state.
         *
         * The `packageName` and `enable` parameters are currently ignored.
         *
         * @return "Success" if the toggle operation succeeds, "Failed" if it does not, or null if the service is unavailable or a remote exception occurs.
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
     * Retrieves a detailed internal status report from the AuraDrive service.
     *
     * @return The detailed status string, or null if the service is unavailable or a RemoteException occurs.
     */
    suspend fun getDetailedInternalStatus(): String? = withContext(Dispatchers.IO) {
        try {
            auraDriveService?.getDetailedInternalStatus()
        } catch (e: RemoteException) {
            null
        }
    }

    /**
     * Retrieves the internal diagnostics log from the AuraDrive service as a single newline-separated string.
     *
     * @return The diagnostics log as a single string, or null if unavailable or if a remote exception occurs.
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
