package dev.aurakai.auraframefx.data.offline

data class OfflineData(
    val lastFullSyncTimestamp: Long? = null
)

class OfflineDataManager {
    /**
     * Attempts to load critical offline data asynchronously.
     *
     * @return An [OfflineData] instance if available, or `null` if no offline data exists.
     */
    suspend fun loadCriticalOfflineData(): OfflineData? {
        return null // Stub implementation
    }
}
