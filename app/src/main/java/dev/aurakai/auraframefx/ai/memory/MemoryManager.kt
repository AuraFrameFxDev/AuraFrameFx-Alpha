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
    private val memoryStore = ConcurrentHashMap<String, MemoryItem>()
    private val _recentAccess = MutableStateFlow(mutableSetOf<String>())
    val recentAccess: StateFlow<Set<String>> = _recentAccess

    private val _memoryStats = MutableStateFlow(MemoryStats())
    val memoryStats: StateFlow<MemoryStats> = _memoryStats

    /**
     * Stores the given memory item in the in-memory store, updates memory statistics, and tracks recent access.
     *
     * @param item The memory item to be stored.
     * @return The unique ID of the stored memory item.
     */
    fun storeMemory(item: MemoryItem): String {
        memoryStore[item.id] = item
        updateStats()
        updateRecentAccess(item.id)
        return item.id
    }

    /**
     * Retrieves memory items that match the specified query criteria.
     *
     * Filters memory items by agent if an agent filter is provided in the query, sorts them by descending timestamp, and limits the results to the configured maximum. Returns a result containing the filtered items, their count, and the original query.
     *
     * @param query The criteria used to filter and retrieve memory items.
     * @return A result containing the retrieved memory items, their total count, and the query used.
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
     * Returns memory items whose timestamps fall within the configured context window duration.
     *
     * Filters and sorts memory items by descending timestamp, returning up to the maximum number specified in the configuration.
     *
     * @param task The task identifier (currently not used in filtering).
     * @return A list of recent memory items within the context window.
     */
    fun getContextWindow(task: String): List<MemoryItem> {
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
     * Retrieves the latest statistics about the memory store.
     *
     * @return The current `MemoryStats` containing totals for items, recent items, memory size, and last update time.
     */
    fun getMemoryStats(): MemoryStats {
        return _memoryStats.value
    }

    /**
     * Updates the memory statistics state with the current total number of items, the count of recent items within the configured context window, and the aggregate size of all memory item contents.
     *
     * The recent item count is calculated based on items whose timestamps are within the maximum chain length duration specified in the configuration.
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
                memorySize = memoryStore.values.sumOf { it.content.length }
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
