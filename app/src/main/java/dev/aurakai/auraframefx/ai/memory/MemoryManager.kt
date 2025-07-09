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
     * Stores a memory item in the in-memory store and updates memory statistics and recent access tracking.
     *
     * @param item The memory item to store.
     * @return The ID of the stored memory item.
     */
    fun storeMemory(item: MemoryItem): String {
        memoryStore[item.id] = item
        updateStats()
        updateRecentAccess(item.id)
        return item.id
    }

    /**
     * Retrieves memory items matching the specified query, filtered by agent if provided, sorted by most recent, and limited to the maximum configured number of items.
     *
     * @param query The criteria used to filter and retrieve memory items.
     * @return A result containing the filtered memory items, their count, and the original query.
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
     * Retrieves recent memory items within the configured context window for a given task.
     *
     * Filters memory items whose timestamps fall within the maximum chain length duration from the current time, sorts them by most recent, and limits the result to the configured maximum chain length.
     *
     * @param task The task identifier for which to retrieve the context window.
     * @return A list of recent MemoryItem objects relevant to the specified task.
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
     * Returns the current memory statistics snapshot.
     *
     * @return The latest `MemoryStats` representing the state of the memory store.
     */
    fun getMemoryStats(): MemoryStats {
        return _memoryStats.value
    }

    /**
     * Updates the memory statistics state with the current total item count, recent item count, and aggregate memory size.
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
