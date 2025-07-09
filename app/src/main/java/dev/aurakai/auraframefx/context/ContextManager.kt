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
     * Creates and registers a new context with the given ID and initial data.
     *
     * Initializes context metadata such as creation time, access count, and last access time, and stores the context in the active contexts map.
     *
     * @param contextId The unique identifier for the context.
     * @param initialData Optional key-value pairs to initialize the context data.
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
     * Increments the context's access count and updates its last access time if the context exists.
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
     * Enhances a user interaction by attaching relevant context, recommending an AI agent, and assigning a priority score.
     *
     * Searches stored memories for context related to the interaction content, determines the most suitable AI agent, and calculates a priority score based on the interaction type. Returns an `EnhancedInteractionData` object containing these enhancements along with the original interaction details.
     *
     * @param interaction The user interaction to enrich.
     * @return An `EnhancedInteractionData` object with relevant context, agent recommendation, and priority score.
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
     * Records a user interaction and its agent response in the conversation history.
     *
     * Also extracts and stores high-confidence memories from the interaction for future retrieval and learning.
     *
     * @param interaction The user interaction data.
     * @param response The agent's response to the interaction.
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
     * Searches stored memories for entries whose content or tags contain the given query string, case-insensitive.
     *
     * Returns up to 10 memories with the highest relevance scores that match the query.
     *
     * @param query The text to search for within memory content and tags.
     * @return A list of up to 10 matching memories, sorted by descending relevance score.
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
     * Records an insight from a request and response for adaptive learning.
     *
     * Creates an `Insight` containing the request, response, complexity, and extracted pattern features, and adds it to the internal insight store. Triggers asynchronous processing to update learning models after every 10 insights.
     *
     * @param request The original request string being analyzed.
     * @param response The system's response to the request.
     * @param complexity A descriptor of the interaction's complexity.
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
     * Enables creative enhancement mode, allowing AI agents to generate more imaginative and innovative responses.
     */
    fun enableCreativeEnhancement() {
        logger.info("ContextManager", "Enabling creative enhancement mode")
        _isCreativeModeEnabled.value = true
    }

    /**
     * Enables creative mode, allowing the system to perform enhanced creative processing.
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
     * Sets the system-wide mood and updates all active contexts with the new mood value.
     *
     * @param newMood The mood to apply across the system and propagate to each active context.
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
     * Records a security event and its threat analysis as a memory entry for future analysis and learning.
     *
     * The memory entry includes event details, threat analysis description, a relevance score based on threat level, and security-related tags.
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
     * Returns a formatted summary of the context's metadata and current system states.
     *
     * The summary includes the context ID, creation time, access count, current mood, creative mode status, unified mode status, and the context's data.
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
     * Returns a bullet-pointed list of memory contents most relevant to the provided input.
     *
     * Searches stored memories for entries related to the given content and formats their contents as a newline-separated list, each prefixed with a bullet point.
     *
     * @param content The input text used to search for relevant memories.
     * @return A string containing relevant memory contents, each on a new line with a bullet point.
     */
    private suspend fun findRelevantContext(content: String): String {
        // Find the most relevant context based on content
        val relevantMemories = searchMemories(content)
        return relevantMemories.joinToString("\n") { "• ${it.content}" }
    }

    /**
     * Determines the most suitable AI agent for an interaction by analyzing keywords in the interaction content.
     *
     * Returns "aura" for creative or artistic topics, "kai" for security-related topics, and "genesis" for analytical or default cases.
     *
     * @param interaction The interaction data to analyze.
     * @return The suggested AI agent: "aura", "kai", or "genesis".
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
     * Determines the priority score of an interaction based on its type.
     *
     * Assigns higher scores to security, analysis, creative, and text interactions in descending order of importance.
     *
     * @param interaction The interaction for which to calculate priority.
     * @return The priority score as an integer; higher values indicate higher priority.
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
     * Stores a conversation entry as a memory if its confidence score exceeds 0.8.
     *
     * The created memory captures the user input, agent response, agent type, and relevant tags for future retrieval.
     *
     * @param entry The conversation entry to evaluate for memory extraction.
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
     * Extracts simple feature descriptors from the request and response for use in learning models.
     *
     * Generates descriptors indicating the length of the request, the length of the response, and whether the request contains a question mark.
     *
     * @return A list of feature strings summarizing the request and response.
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
     * This is a placeholder for future implementation of adaptive learning and pattern analysis based on recorded insights.
     */
    private suspend fun processInsightsForLearning() {
        logger.info("ContextManager", "Processing insights for learning")
        // Implementation would analyze patterns and update learning models
    }

    /**
     * Cancels all active coroutines and releases resources managed by the ContextManager.
     *
     * Call this method to ensure a clean shutdown and prevent resource leaks.
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

