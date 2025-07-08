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
         * Handles the event when the AuraDrive service is connected.
         *
         * Initializes the AuraDrive AIDL interface from the provided binder and updates the connection state to indicate a successful connection.
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

    /**
     * Attempts to bind to the AuraDrive service using an explicit intent.
     *
     * Updates the connection state to false if binding fails due to a security exception.
     */
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
     * Unbinds from the AuraDrive service and resets the connection state.
     *
     * Attempts to unbind from the AuraDrive service, clears the internal service interface reference,
     * and marks the service as disconnected. Any exceptions during unbinding are silently ignored.
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
     * Retrieves the current status from the AuraDrive service.
     *
     * Executes on the IO dispatcher. Returns the status string if the service is connected, or null if the service is unavailable or a RemoteException occurs.
     *
     * @return The current status string from the AuraDrive service, or null if unavailable or on communication error.
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
         * The `packageName` and `enable` parameters are currently ignored by the service call.
         *
         * @return "Success" if the toggle operation succeeds, "Failed" if it fails, or null if the service is unavailable or a remote exception occurs.
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
     * Executes on the IO dispatcher. Returns the detailed status string if available, or null if the service is not connected or a RemoteException occurs.
     *
     * @return The detailed internal status string, or null if unavailable.
     */
    suspend fun getDetailedInternalStatus(): String? = withContext(Dispatchers.IO) {
        try {
            auraDriveService?.getDetailedInternalStatus()
        } catch (e: RemoteException) {
            null
        }
    }

    /**
     * Retrieves the internal diagnostics log from the AuraDrive service as a single string.
     *
     * Executes on the IO dispatcher. Returns the diagnostics log as a newline-separated string, or null if the service is unavailable or a remote exception occurs.
     *
     * @return The diagnostics log as a single string, or null if not available.
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
