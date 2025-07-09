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
     * Stores a canonical memory item and updates memory statistics and recent access tracking.
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
     * Retrieves memory items matching the specified query, filtered by agent and limited to the maximum configured number.
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
     * Retrieves a list of recent memory items within the configured context window.
     *
     * The returned list contains up to the maximum number of items specified by the context chaining configuration, sorted by descending timestamp and limited to those within the recent time window.
     *
     * @param task The task identifier (currently unused in filtering).
     * @return A list of recent canonical memory items for context chaining.
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
     * Returns the current memory statistics.
     *
     * @return The latest snapshot of memory statistics, including total items, recent items, memory size, and last updated time.
     */
    fun getMemoryStats(): MemoryStats {
        return _memoryStats.value
    }

    /**
     * Recalculates and updates memory statistics, including total items, recent items within the configured time window, aggregate memory size, and the current timestamp.
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
