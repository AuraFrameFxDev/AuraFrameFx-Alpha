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

<<<<<<< HEAD
    /**
     * Stores a memory item in the memory store and updates statistics and recent access tracking.
     *
     * @param item The memory item to store.
     * @return The ID of the stored memory item.
     */
=======
>>>>>>> pr458merge
    /**
     * Stores a memory item in the memory store and updates statistics and recent access.
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

<<<<<<< HEAD
    /**
     * Retrieves memory items matching the specified query criteria.
     *
     * Filters memory items by agent if an agent filter is provided in the query, sorts them by descending timestamp, and limits the results to the maximum number configured. Returns a result containing the filtered items, their count, and the original query.
     *
     * @param query The criteria used to filter and retrieve memory items.
     * @return A result object containing the retrieved memory items, their total count, and the query used.
     */
=======
>>>>>>> pr458merge
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

<<<<<<< HEAD
    /**
     * Retrieves recent memory items within the configured context window duration.
     *
     * Filters memory items whose timestamps fall within the maximum chain length duration from the current time,
     * sorts them in descending order by timestamp, and limits the result to the configured maximum number of items.
     *
     * @param task The task identifier (currently unused in filtering).
     * @return A list of recent memory items for the context window.
     */
=======
>>>>>>> pr458merge
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

<<<<<<< HEAD
    /**
     * Returns the current snapshot of memory statistics.
     *
     * @return The latest `MemoryStats` reflecting the state of the memory store.
     */
=======
>>>>>>> pr458merge
    fun getMemoryStats(): MemoryStats {
        return _memoryStats.value
    }

<<<<<<< HEAD
    /**
     * Updates the memory statistics state with the current total item count, recent item count, and aggregate memory size.
     *
     * The recent item count is determined by items whose timestamps fall within the configured maximum chain length duration.
     */
=======
>>>>>>> pr458merge
    /**
     * Refreshes the memory statistics state with the latest total item count, recent item count within the configured duration, and total memory content size.
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
