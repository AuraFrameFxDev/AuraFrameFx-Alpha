package dev.aurakai.auraframefx.data.offline

data class OfflineData(
    val lastFullSyncTimestamp: Long? = null
)

class OfflineDataManager {
    /**
     * Loads critical offline data necessary for application operation.
     *
     * @return An [OfflineData] instance if available, or `null` if no offline data exists.
     */
    suspend fun loadCriticalOfflineData(): OfflineData? {
        return null // Stub implementation
    }
}
