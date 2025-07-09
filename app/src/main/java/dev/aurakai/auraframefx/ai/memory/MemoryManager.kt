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
    private val memoryStore =
        ConcurrentHashMap<String, CanonicalMemoryItem>() // Changed MemoryItem to CanonicalMemoryItem
    private val _recentAccess = MutableStateFlow(mutableSetOf<String>())
    val recentAccess: StateFlow<Set<String>> = _recentAccess

    private val _memoryStats = MutableStateFlow(MemoryStats())
    val memoryStats: StateFlow<MemoryStats> = _memoryStats

    /**
     * Stores a canonical memory item in the memory store.
     *
     * Updates memory statistics and recent access tracking after storing the item.
     *
     * @param item The canonical memory item to store.
     * @return The unique ID of the stored memory item.
     */
    fun storeMemory(item: CanonicalMemoryItem): String { // Changed MemoryItem to CanonicalMemoryItem
        memoryStore[item.id] = item
        updateStats()
        updateRecentAccess(item.id)
        return item.id
    }

    /**
     * Retrieves memory items that match the given query, optionally filtering by agent.
     *
     * Filters stored canonical memory items by the agent(s) specified in the query (if any), sorts them by descending timestamp, and limits the results to the maximum number configured. Returns a [MemoryRetrievalResult] containing the filtered items, their count, and the original query.
     *
     * @param query The memory query specifying agent filters and retrieval criteria.
     * @return A [MemoryRetrievalResult] with the matching memory items, their count, and the query used.
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
     * Returns the most recent canonical memory items within the configured context window.
     *
     * Filters memory items whose timestamps are within the maximum chain length duration from the current time,
     * sorts them by descending timestamp, and returns up to the configured maximum number of items.
     *
     * @param task The task identifier (currently not used for filtering).
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
     * Returns the latest snapshot of memory statistics.
     *
     * @return The current memory statistics, including total item count, recent item count, aggregate memory size, and last updated timestamp.
     */
    fun getMemoryStats(): MemoryStats {
        return _memoryStats.value
    }

    /**
     * Recalculates and updates memory statistics, including total items, recent items within the context window, aggregate memory size, and the last update timestamp.
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
