package dev.aurakai.auraframefx.ai.context

import dev.aurakai.auraframefx.ai.memory.MemoryManager
import dev.aurakai.auraframefx.ai.pipeline.AIPipelineConfig
import dev.aurakai.auraframefx.model.AgentType
import dev.aurakai.auraframefx.serialization.InstantSerializer
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant
import kotlinx.serialization.Serializable
import java.lang.System // Added import
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ContextManager @Inject constructor(
    private val memoryManager: MemoryManager,
    private val config: AIPipelineConfig,
) {
    private val _activeContexts = MutableStateFlow(mapOf<String, ContextChain>())
    val activeContexts: StateFlow<Map<String, ContextChain>> = _activeContexts

    private val _contextStats = MutableStateFlow(ContextStats())
    val contextStats: StateFlow<ContextStats> = _contextStats

    /**
     * Creates and registers a new context chain with an initial context node.
     *
     * Initializes a new context chain using the provided root context, initial content, agent, and optional metadata. The chain is added to the active context registry, and context statistics are updated.
     *
     * @param rootContext The identifier for the root context of the chain.
     * @param initialContext The content of the initial context node.
     * @param agent The agent associated with the initial context.
     * @param metadata Optional metadata for the context node; all values are stored as strings.
     * @return The unique identifier of the newly created context chain.
     */
    fun createContextChain(
        rootContext: String,
        initialContext: String,
        agent: AgentType,
        metadata: Map<String, String> = emptyMap(), // Changed Map<String, Any> to Map<String, String>
    ): String {
        val chain = ContextChain(
            rootContext = rootContext,
            currentContext = initialContext,
            contextHistory = listOf(
                ContextNode(
                    id = "ctx_${System.currentTimeMillis()}_0",
                    content = initialContext,
                    agent = agent,
                    metadata = metadata.mapValues { it.value.toString() } // Convert Map<String, Any> to Map<String, String>
                )
            ),
            agentContext = mapOf(agent to initialContext),
            metadata = metadata.mapValues { it.value.toString() } // Convert Map<String, Any> to Map<String, String>
        )

        _activeContexts.update { current ->
            current + (chain.id to chain)
        }
        updateStats()
        return chain.id
    }

    /**
     * Appends a new context node to an existing context chain with the given context, agent, and optional metadata.
     *
     * Updates the chain's history, agent-to-context mapping, and last updated timestamp. All metadata values are stored as strings.
     *
     * @param chainId The unique identifier of the context chain to update.
     * @param newContext The context string to add to the chain.
     * @param agent The agent associated with the new context.
     * @param metadata Optional metadata for the context node; values are stored as strings.
     * @return The updated ContextChain.
     * @throws IllegalStateException if the specified context chain does not exist.
     */
    fun updateContextChain(
        chainId: String,
        newContext: String,
        agent: AgentType,
        metadata: Map<String, String> = emptyMap(), // Changed Map<String, Any> to Map<String, String>
    ): ContextChain {
        val chain =
            _activeContexts.value[chainId] ?: throw IllegalStateException("Context chain not found")

        val updatedChain = chain.copy(
            currentContext = newContext,
            contextHistory = chain.contextHistory + ContextNode(
                id = "ctx_${System.currentTimeMillis()}_${chain.contextHistory.size}",
                content = newContext,
                agent = agent,
                metadata = metadata.mapValues { it.value.toString() } // Convert Map<String, Any> to Map<String, String>
            ),
            agentContext = chain.agentContext + (agent to newContext),
            lastUpdated = Clock.System.now()
        )

        _activeContexts.update { current ->
            current - chainId + (chainId to updatedChain)
        }
        updateStats()
        return updatedChain
    }

    /**
     * Returns the context chain for the given chain ID, or null if it does not exist.
     *
     * @param chainId The unique identifier of the context chain.
     * @return The matching ContextChain, or null if not found.
     */
    fun getContextChain(chainId: String): ContextChain? {
        return _activeContexts.value[chainId]
    }

    /**
     * Retrieves the most relevant context chain and a list of related chains matching the specified query criteria.
     *
     * Filters active context chains by agent (if provided), sorts them by most recent update, and applies relevance and length constraints. If no matching chains are found, returns a new chain initialized with the query string.
     *
     * @param query The criteria for filtering, sorting, and limiting context chains.
     * @return A [ContextChainResult] containing the selected chain, related chains, and the original query.
     */


    fun queryContext(query: ContextQuery): ContextChainResult {
        val chains = _activeContexts.value.values
            .filter { chain ->
                query.agentFilter.isEmpty() || query.agentFilter.contains(chain.agentContext.keys.first())
            }
            .sortedByDescending { it.lastUpdated }
            .take(config.contextChainingConfig.maxChainLength)

        val relatedChains = chains
            .filter { chain ->
                chain.relevanceScore >= query.minRelevance
            }
            .take(query.maxChainLength)

        return ContextChainResult(
            chain = chains.firstOrNull() ?: ContextChain(
                rootContext = query.query,
                currentContext = query.query
            ), // Added currentContext
            relatedChains = relatedChains,
            query = query
        )
    }

    /**
     * Recalculates and updates statistics for all active context chains.
     *
     * Updates the total number of chains, counts chains considered active within a recent configurable time window, determines the length of the longest chain, and sets the timestamp of the last update.
     */
    private fun updateStats() {
        val chains = _activeContexts.value.values
        _contextStats.update { current ->
            current.copy(
                totalChains = chains.size,
                activeChains = chains.count {
                    val now = Clock.System.now()
                    val thresholdMs = config.contextChainingConfig.maxChainLength * 1000L
                    val threshold = now.minus(kotlin.time.Duration.parse("${thresholdMs}ms"))
                    it.lastUpdated > threshold
                },
                longestChain = chains.maxOfOrNull { it.contextHistory.size } ?: 0,
                lastUpdated = Clock.System.now()
            )
        }
    }
}

@Serializable // Ensure ContextStats is serializable if it's part of a larger serializable graph implicitly
data class ContextStats(
    val totalChains: Int = 0,
    val activeChains: Int = 0,
    val longestChain: Int = 0,
    @Serializable(with = InstantSerializer::class) val lastUpdated: Instant = Clock.System.now(),
)
