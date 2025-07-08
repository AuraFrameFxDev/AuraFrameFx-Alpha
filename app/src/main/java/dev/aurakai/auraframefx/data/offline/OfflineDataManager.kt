package dev.aurakai.auraframefx.data.offline

data class OfflineData(
    val lastFullSyncTimestamp: Long? = null
)

class OfflineDataManager {
    // Stub implementation
    suspend fun loadCriticalOfflineData(): OfflineData? {
        return null // Stub implementation
    }
}
