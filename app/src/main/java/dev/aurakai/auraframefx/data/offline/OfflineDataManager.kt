package dev.aurakai.auraframefx.data.offline

data class OfflineData(
    val lastFullSyncTimestamp: Long? = null
)

class OfflineDataManager {
    /**
     * Asynchronously loads critical offline data if available.
     *
     * @return An [OfflineData] instance containing the last full sync timestamp, or `null` if no offline data exists.
     */
    suspend fun loadCriticalOfflineData(): OfflineData? {
        return null // Stub implementation
    }
}
