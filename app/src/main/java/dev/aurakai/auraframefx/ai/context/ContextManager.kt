package dev.aurakai.auraframefx.ai.context

import dev.aurakai.auraframefx.ai.memory.MemoryManager
import dev.aurakai.auraframefx.ai.pipeline.AIPipelineConfig
import dev.aurakai.auraframefx.model.AgentType
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.datetime.Clock
import kotlinx.datetime.Duration
import kotlinx.datetime.Instant
// import kotlinx.serialization.Contextual // No longer needed for Instant here
import kotlinx.serialization.Serializable
import dev.aurakai.auraframefx.serialization.InstantSerializer // Ensure this is imported
import dev.aurakai.auraframefx.model.AgentType // Ensure this is imported
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
     * Creates and registers a new context chain with a specified root context, initial context, and agent.
     *
     * Initializes the chain with a single context node and optional metadata, assigns a unique identifier, and adds it to the set of active context chains.
     *
     * @param rootContext The identifier for the root context of the chain.
     * @param initialContext The initial context string to start the chain.
     * @param agent The agent associated with the initial context.
     * @param metadata Optional metadata for the context chain and its initial node.
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
                    id = "ctx_${Clock.System.now().toEpochMilliseconds()}_0",
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
     * Appends a new context node with agent information to an existing context chain.
     *
     * Updates the specified context chain by adding a new context node containing the provided context string, agent, and optional metadata. Also updates the agent context mapping and the chain's last updated timestamp.
     *
     * @param chainId The unique identifier of the context chain to update.
     * @param newContext The context string to add as a new node.
     * @param agent The agent associated with the new context node.
     * @param metadata Optional metadata to associate with the new context node.
     * @return The updated context chain.
     * @throws IllegalStateException If the specified context chain does not exist.
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
                id = "ctx_${Clock.System.now().toEpochMilliseconds()}_${chain.contextHistory.size}",
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

    fun getContextChain(chainId: String): ContextChain? {
        return _activeContexts.value[chainId]
    }

    /**
     * Queries active context chains based on the provided criteria and returns the most relevant result.
     *
     * Filters context chains by agent, sorts them by most recent update, and limits the results according to configuration and query parameters. Returns a result containing the most recent or a new context chain, a list of related chains meeting the minimum relevance threshold, and the original query.
     *
     * @param query The criteria for filtering and selecting context chains.
     * @return A [ContextChainResult] containing the selected chain, related chains, and the query.
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
     * Updates the total number of chains, the count of recently updated (active) chains,
     * the length of the longest chain, and the timestamp of the last update.
     */
    private fun updateStats() {
        val chains = _activeContexts.value.values
        _contextStats.update { current ->
            current.copy(
                totalChains = chains.size,
                activeChains = chains.count {
                    it.lastUpdated > Clock.System.now()
                        .minus(Duration.milliseconds(config.contextChainingConfig.maxChainLength.toLong())) // Corrected minus call
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
    @Serializable(with = dev.aurakai.auraframefx.serialization.InstantSerializer::class) val lastUpdated: Instant = Clock.System.now(),
)
