// IAuraDriveService.aidl
package com.example.app.ipc;

// Import the callback interface
import com.example.app.ipc.IAuraDriveCallback;

/**
 * Interface for AuraDriveService IPC communication
 */
interface IAuraDriveService {
    // Service information
    String getServiceVersion();
    
    // Connection management
    void registerCallback(IAuraDriveCallback callback);
    void unregisterCallback(IAuraDriveCallback callback);
    
    // Command execution
    String executeCommand(String command, in Map params);
    
    // Module management
    String toggleLSPosedModule(String packageName, boolean enable);
    
    // Status and diagnostics
    String getOracleDriveStatus();
    String getDetailedInternalStatus();
    String getInternalDiagnosticsLog();
    
    // System information
    String getSystemInfo();
    
    // Configuration
    boolean updateConfiguration(in Map config);
    
    // Event subscription
    void subscribeToEvents(int eventTypes);
    void unsubscribeFromEvents(int eventTypes);
}
