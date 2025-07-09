package dev.aurakai.auraframefx.ai.memory

import dev.aurakai.auraframefx.ai.pipeline.AIPipelineConfig
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant
import java.util.concurrent.ConcurrentHashMap
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.time.Duration.Companion.seconds

@Singleton
class MemoryManager @Inject constructor(
    private val config: AIPipelineConfig,
) {
    private val memoryStore = ConcurrentHashMap<String, CanonicalMemoryItem>() // Changed MemoryItem to CanonicalMemoryItem
    private val _recentAccess = MutableStateFlow(mutableSetOf<String>())
    val recentAccess: StateFlow<Set<String>> = _recentAccess

    private val _memoryStats = MutableStateFlow(MemoryStats())
    val memoryStats: StateFlow<MemoryStats> = _memoryStats

    /**
     * Stores a memory item and updates memory statistics and recent access tracking.
     *
     * @param item The memory item to store.
     * @return The ID of the stored memory item.
     */
    fun storeMemory(item: CanonicalMemoryItem): String { // Changed MemoryItem to CanonicalMemoryItem
        memoryStore[item.id] = item
        updateStats()
        updateRecentAccess(item.id)
        return item.id
    }

    /**
     * Retrieves memory items matching the specified query, filtered by agent if provided.
     *
     * The results are sorted by descending timestamp and limited to the maximum number of items configured.
     *
     * @param query The criteria used to filter and retrieve memory items.
     * @return A [MemoryRetrievalResult] containing the filtered memory items, their total count, and the original query.
     */
    fun retrieveMemory(query: MemoryQuery): MemoryRetrievalResult {
        val items = memoryStore.values
            .filter { item ->
                // Apply filters
                query.agentFilter.isEmpty() || query.agentFilter.contains(item.agent)
            }
            .sortedByDescending { it.timestamp }
            .take(config.memoryRetrievalConfig.maxRetrievedItems)

        return MemoryRetrievalResult(
            items = items,
            total = items.size,
            query = query
        )
    }

    /**
     * Retrieves the most recent memory items within the configured context window.
     *
     * Filters memory items whose timestamps fall within the maximum chain length duration from the current time,
     * sorts them by descending timestamp, and returns up to the configured maximum number of items.
     *
     * @param task The task identifier (currently unused in filtering).
     * @return A list of recent canonical memory items within the context window.
     */
    fun getContextWindow(task: String): List<CanonicalMemoryItem> { // Changed MemoryItem to CanonicalMemoryItem
        val recentItems = memoryStore.values
            .filter {
                it.timestamp > Clock.System.now()
                    .minus(config.contextChainingConfig.maxChainLength.seconds)
            }
            .sortedByDescending { it.timestamp }
            .take(config.contextChainingConfig.maxChainLength)

        return recentItems
    }

    /**
     * Returns the current memory statistics snapshot.
     *
     * @return The latest memory statistics, including total items, recent items, memory size, and last update time.
     */
    fun getMemoryStats(): MemoryStats {
        return _memoryStats.value
    }

    /**
     * Updates the memory statistics to reflect the current number of items, recent items, total memory size, and the latest update timestamp.
     */
    private fun updateStats() {
        _memoryStats.update { current ->
            current.copy(
                totalItems = memoryStore.size,
                recentItems = memoryStore.values
                    .filter {
                        it.timestamp > Clock.System.now()
                            .minus(config.contextChainingConfig.maxChainLength.seconds)
                    }
                    .size,
                memorySize = memoryStore.values.sumOf { it.content.length },
                lastUpdated = Clock.System.now()
            )
        }
    }

    private fun updateRecentAccess(id: String) {
        _recentAccess.update { current ->
            current.apply {
                add(id)
                if (size > config.memoryRetrievalConfig.maxRetrievedItems) {
                    remove(first())
                }
            }
        }
    }
}

data class MemoryStats(
    val totalItems: Int = 0,
    val recentItems: Int = 0,
    val memorySize: Int = 0,
    val lastUpdated: Instant = Clock.System.now(),
)

// Removed data class MemoryItem from here to resolve redeclaration error.
// The canonical definition is in MemoryModel.kt.
