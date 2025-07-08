package dev.aurakai.auraframefx.data.offline

data class OfflineData(
    val lastFullSyncTimestamp: Long? = null
)

class OfflineDataManager {
    /**
     * Asynchronously loads essential offline data for the application.
     *
     * @return An [OfflineData] object if available, or `null` if no offline data exists.
     */
    suspend fun loadCriticalOfflineData(): OfflineData? {
        return null // Stub implementation
    }
}
