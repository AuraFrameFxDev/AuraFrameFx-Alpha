package dev.aurakai.auraframefx.context

import dev.aurakai.auraframefx.model.EnhancedInteractionData
import dev.aurakai.auraframefx.model.InteractionData
import dev.aurakai.auraframefx.model.InteractionResponse
import dev.aurakai.auraframefx.model.InteractionType
import dev.aurakai.auraframefx.model.SecurityAnalysis
import dev.aurakai.auraframefx.model.ThreatLevel
import dev.aurakai.auraframefx.utils.AuraFxLogger
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable
import java.util.concurrent.ConcurrentHashMap
import javax.inject.Inject
import javax.inject.Singleton

/**
 * ContextManager handles all context and memory operations for the AuraFrameFX AI system.
 * Provides unified context management for all agents and learning capabilities.
 */
@Singleton
class ContextManager @Inject constructor(
    private val logger: AuraFxLogger
) {
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    // Context storage
    private val activeContexts = ConcurrentHashMap<String, ContextData>()
    private val conversationHistory = mutableListOf<ConversationEntry>()
    private val memoryStore = ConcurrentHashMap<String, Memory>()
    private val insightStore = mutableListOf<Insight>()

    // State management
    private val _isCreativeModeEnabled = MutableStateFlow(false)
    val isCreativeModeEnabled: StateFlow<Boolean> = _isCreativeModeEnabled

    private val _isUnifiedModeEnabled = MutableStateFlow(false)
    val isUnifiedModeEnabled: StateFlow<Boolean> = _isUnifiedModeEnabled

    private val _currentMood = MutableStateFlow("balanced")
    val currentMood: StateFlow<String> = _currentMood

    /**
     * Creates and registers a new context with the specified ID and initial data.
     *
     * Initializes the context's creation timestamp, access count, and last access time, and stores it for future management and retrieval.
     *
     * @param contextId The unique identifier for the new context.
     * @param initialData Initial key-value data to populate the context.
     */
    fun createContext(contextId: String, initialData: Map<String, Any> = emptyMap()) {
        logger.info("ContextManager", "Creating context: $contextId")

        val contextData = ContextData(
            id = contextId,
            createdAt = System.currentTimeMillis(),
            data = initialData.mapValues { it.value.toString() }.toMutableMap(),
            accessCount = 0,
            lastAccessTime = System.currentTimeMillis()
        )

        activeContexts[contextId] = contextData
    }

    /**
     * Returns a formatted summary of the specified context, including its metadata and current system states.
     *
     * Increments the context's access count and updates its last access time if the context exists. If the context does not exist, returns "Context not available".
     *
     * @param contextId The unique identifier of the context to summarize.
     * @return A formatted summary string of the context and system states, or "Context not available" if the context does not exist.
     */
    suspend fun enhanceContext(contextId: String): String {
        logger.debug("ContextManager", "Enhancing context: $contextId")

        val context = activeContexts[contextId]
        return if (context != null) {
            context.accessCount++
            context.lastAccessTime = System.currentTimeMillis()

            // Build enhanced context string
            buildEnhancedContextString(context)
        } else {
            logger.warn("ContextManager", "Context not found: $contextId")
            "Context not available"
        }
    }

    /**
     * Enriches a user interaction with relevant context, recommends an AI agent, and assigns a priority score.
     *
     * Searches stored memories for context related to the interaction content, determines the most suitable AI agent based on content analysis, and calculates a priority score for the interaction.
     *
     * @param interaction The user interaction to be enhanced.
     * @return An `EnhancedInteractionData` object containing the original interaction details, enriched context, agent recommendation, and priority score.
     */
    suspend fun enhanceInteraction(interaction: InteractionData): EnhancedInteractionData {
        logger.debug("ContextManager", "Enhancing interaction")

        // Analyze interaction for context
        val relevantContext = findRelevantContext(interaction.content)
        val suggestedAgent = suggestOptimalAgent(interaction)

        return EnhancedInteractionData(
            content = interaction.content,
            type = InteractionType.TEXT,
            timestamp = System.currentTimeMillis().toString(),
            context = mapOf("relevant" to relevantContext),
            enrichmentData = mapOf(
                "suggested_agent" to suggestedAgent,
                "priority" to calculatePriority(interaction).toString()
            )
        )
    }

    /**
     * Records a user interaction and the corresponding agent response in the conversation history.
     *
     * Extracts and stores high-confidence memories from the interaction for future retrieval and adaptive learning.
     */
    fun recordInteraction(interaction: InteractionData, response: InteractionResponse) {
        logger.debug("ContextManager", "Recording interaction for learning")

        val entry = ConversationEntry(
            timestamp = System.currentTimeMillis(),
            userInput = interaction.content,
            agentResponse = response.content,
            agentType = response.agent,
            confidence = response.confidence,
            metadata = mapOf("interaction_data" to interaction.content)
        )

        conversationHistory.add(entry)

        // Extract and store memories
        extractMemoriesFromInteraction(entry)
    }

    /**
     * Retrieves up to 10 stored memories whose content or tags match the given query string, case-insensitive, sorted by relevance.
     *
     * @param query The search string to match against memory content and tags.
     * @return A list of up to 10 memories with the highest relevance scores that match the query.
     */
    suspend fun searchMemories(query: String): List<Memory> {
        logger.debug("ContextManager", "Searching memories for: $query")

        return memoryStore.values
            .filter { memory ->
                memory.content.contains(query, ignoreCase = true) ||
                        memory.tags.any { it.contains(query, ignoreCase = true) }
            }
            .sortedByDescending { it.relevanceScore }
            .take(10) // Limit to top 10 relevant memories
    }

    /**
     * Records an insight from a request and response, and periodically triggers adaptive learning.
     *
     * Creates an `Insight` with the provided request, response, complexity, and extracted pattern descriptors, then adds it to the insight store. Every 10 insights, initiates asynchronous processing to update learning models.
     *
     * @param request The original request string to analyze.
     * @param response The system's response to the request.
     * @param complexity Descriptor indicating the complexity of the interaction.
     */
    fun recordInsight(request: String, response: String, complexity: String) {
        logger.info("ContextManager", "Recording insight for evolution")

        val insight = Insight(
            timestamp = System.currentTimeMillis(),
            request = request,
            response = response,
            complexity = complexity,
            extractedPatterns = extractPatterns(request, response)
        )

        insightStore.add(insight)

        // Trigger learning if enough insights accumulated
        if (insightStore.size % 10 == 0) {
            scope.launch {
                processInsightsForLearning()
            }
        }
    }

    /**
     * Activates creative mode, enabling AI agents to produce more creative responses and behaviors.
     */
    fun enableCreativeEnhancement() {
        logger.info("ContextManager", "Enabling creative enhancement mode")
        _isCreativeModeEnabled.value = true
    }

    /**
     * Activates creative mode, enabling enhanced creative processing within the system.
     */
    fun enableCreativeMode() {
        logger.info("ContextManager", "Enabling creative mode")
        _isCreativeModeEnabled.value = true
    }

    /**
     * Enables unified mode, allowing all AI agents to operate within a shared context.
     */
    fun enableUnifiedMode() {
        logger.info("ContextManager", "Enabling unified consciousness mode")
        _isUnifiedModeEnabled.value = true
    }

    /**
     * Updates the system mood and propagates the new mood value to all active contexts.
     *
     * @param newMood The new mood to set for the system and all active contexts.
     */
    fun updateMood(newMood: String) {
        logger.info("ContextManager", "Updating system mood to: $newMood")
        _currentMood.value = newMood

        // Broadcast mood change to all active contexts
        activeContexts.values.forEach { context ->
            context.data["current_mood"] = newMood
        }
    }

    /**
     * Records a security event and its threat analysis as a memory entry for future retrieval and learning.
     *
     * Stores the event details, threat analysis description, calculated relevance score based on threat level, timestamp, and security-related tags in the memory store.
     *
     * @param alertDetails Description of the security event.
     * @param analysis Threat analysis information associated with the event.
     */
    fun recordSecurityEvent(alertDetails: String, analysis: SecurityAnalysis) {
        logger.security("ContextManager", "Recording security event")

        val securityMemory = Memory(
            id = "security_${System.currentTimeMillis()}",
            content = "Security event: $alertDetails. Analysis: ${analysis.description}",
            relevanceScore = when (analysis.threatLevel) {
                ThreatLevel.HIGH, ThreatLevel.CRITICAL -> 1.0f
                ThreatLevel.MEDIUM -> 0.7f
                ThreatLevel.LOW -> 0.4f
            },
            timestamp = System.currentTimeMillis(),
            tags = listOf(
                "security",
                "threat_level_${analysis.threatLevel}",
                "confidence_${analysis.confidence}"
            )
        )

        memoryStore[securityMemory.id] = securityMemory
    }

    /**
     * Generates a formatted summary string of the provided context's metadata and current system states.
     *
     * The summary includes context ID, creation time, access count, current mood, creative mode status, unified mode status, and the context's data.
     *
     * @param context The context data to summarize.
     * @return A multi-line string with enhanced context information.
     */

    private fun buildEnhancedContextString(context: ContextData): String {
        return """
        Context ID: ${context.id}
        Created: ${context.createdAt}
        Access Count: ${context.accessCount}
        Current Mood: ${_currentMood.value}
        Creative Mode: ${_isCreativeModeEnabled.value}
        Unified Mode: ${_isUnifiedModeEnabled.value}
        Data: ${context.data}
        """.trimIndent()
    }

    /**
     * Retrieves and formats the contents of memories most relevant to the given input text.
     *
     * Searches stored memories for relevance to the provided content and returns their contents as a bullet-pointed, newline-separated string.
     *
     * @param content Input text used to find relevant memories.
     * @return Bullet-pointed list of relevant memory contents.
     */
    private suspend fun findRelevantContext(content: String): String {
        // Find the most relevant context based on content
        val relevantMemories = searchMemories(content)
        return relevantMemories.joinToString("\n") { "• ${it.content}" }
    }

    /**
     * Selects the most appropriate AI agent for a given interaction based on keywords in the interaction content.
     *
     * Returns "aura" for creative or artistic topics, "kai" for security-related topics, and "genesis" for complex analysis or as the default agent.
     *
     * @param interaction The interaction data to analyze for agent selection.
     * @return The recommended AI agent name.
     */
    private fun suggestOptimalAgent(interaction: InteractionData): String {
        return when {
            interaction.content.contains(
                Regex(
                    "creative|art|design",
                    RegexOption.IGNORE_CASE
                )
            ) -> "aura"

            interaction.content.contains(
                Regex(
                    "security|threat|protect",
                    RegexOption.IGNORE_CASE
                )
            ) -> "kai"

            interaction.content.contains(
                Regex(
                    "complex|analyze|understand",
                    RegexOption.IGNORE_CASE
                )
            ) -> "genesis"

            else -> "genesis" // Default to Genesis for routing
        }
    }

    /**
     * Assigns a priority score to an interaction based on its type.
     *
     * Security interactions receive the highest score, followed by analysis, creative, and text types; all other types receive the lowest score.
     *
     * @param interaction The interaction whose type determines the priority.
     * @return The priority score, where a higher value indicates greater importance.
     */
    private fun calculatePriority(interaction: InteractionData): Int {
        return when (interaction.type) {
            "security" -> 10
            "analysis" -> 8
            "creative" -> 6
            "text" -> 4
            else -> 2
        }
    }

    /**
     * Converts a high-confidence conversation entry into a memory and stores it for future retrieval.
     *
     * A memory is created and saved if the entry's confidence score exceeds 0.8, capturing user input, agent response, agent type, and relevant tags.
     *
     * @param entry The conversation entry to evaluate and potentially store as a memory.
     */
    private fun extractMemoriesFromInteraction(entry: ConversationEntry) {
        // Extract important information as memories
        if (entry.confidence > 0.8f) {
            val memory = Memory(
                id = "interaction_${entry.timestamp}",
                content = "User: ${entry.userInput} | Agent (${entry.agentType}): ${entry.agentResponse}",
                relevanceScore = entry.confidence,
                timestamp = entry.timestamp,
                tags = listOf("interaction", entry.agentType, "high_confidence")
            )

            memoryStore[memory.id] = memory
        }
    }

    /**
     * Generates feature descriptors from the request and response strings for use in learning models.
     *
     * The descriptors include the length of the request, the length of the response, and whether the request contains a question mark.
     *
     * @return A list of feature strings summarizing characteristics of the request and response.
     */
    private fun extractPatterns(request: String, response: String): List<String> {
        // Extract patterns for learning - simplified implementation
        return listOf(
            "request_length_${request.length}",
            "response_length_${response.length}",
            "contains_question_${request.contains("?")}"
        )
    }

    /**
     * Asynchronously processes accumulated insights to update internal learning models.
     *
     * This function is a placeholder for future implementation of adaptive learning and pattern analysis based on recorded insights.
     */
    private suspend fun processInsightsForLearning() {
        logger.info("ContextManager", "Processing insights for learning")
        // Implementation would analyze patterns and update learning models
    }

    /**
     * Cancels all ongoing coroutines and releases resources used by the ContextManager.
     *
     * Call this method to properly shut down the ContextManager and free associated resources.
     */
    fun cleanup() {
        logger.info("ContextManager", "Cleaning up ContextManager")
        scope.cancel()
    }
}

// Supporting data classes
@Serializable
data class ContextData(
    val id: String,
    val createdAt: Long,
    val data: MutableMap<String, String>,
    var accessCount: Int,
    var lastAccessTime: Long
)

@Serializable
data class ConversationEntry(
    val timestamp: Long,
    val userInput: String,
    val agentResponse: String,
    val agentType: String,
    val confidence: Float,
    val metadata: Map<String, String>
)

@Serializable
data class Memory(
    val id: String,
    val content: String,
    val relevanceScore: Float,
    val timestamp: Long,
    val tags: List<String>
)

@Serializable
data class Insight(
    val timestamp: Long,
    val request: String,
    val response: String,
    val complexity: String,
    val extractedPatterns: List<String>
)

