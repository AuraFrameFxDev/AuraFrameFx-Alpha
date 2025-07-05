package dev.aurakai.auraframefx.context

import dev.aurakai.auraframefx.utils.AuraFxLogger
import dev.aurakai.auraframefx.ai.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
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
     * Registers a new context with the given ID and optional initial data.
     *
     * Initializes and stores context metadata, including creation time, access count, and last access time, for future management and enhancement.
     *
     * @param contextId The unique identifier for the context.
     * @param initialData Optional initial data to associate with the context.
     */
    fun createContext(contextId: String, initialData: Map<String, Any> = emptyMap()) {
        logger.info("ContextManager", "Creating context: $contextId")
        
        val contextData = ContextData(
            id = contextId,
            createdAt = System.currentTimeMillis(),
            data = initialData.toMutableMap(),
            accessCount = 0,
            lastAccessTime = System.currentTimeMillis()
        )
        
        activeContexts[contextId] = contextData
    }

    /**
     * Enhances and returns a string representation of the specified context, including its metadata and current system states.
     *
     * Updates the context's access count and last access time if found. Returns a fallback message if the context does not exist.
     *
     * @param contextId The unique identifier of the context to enhance.
     * @return An enhanced string with context details, or a fallback message if the context is not found.
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
     * Enhances an interaction by retrieving relevant context, suggesting the most suitable AI agent, and assigning a priority level.
     *
     * Analyzes the interaction content to find related memories, determines the optimal agent based on keywords, and calculates a priority score. Returns an `EnhancedInteractionData` object containing these enhancements.
     *
     * @param interaction The interaction data to analyze and enhance.
     * @return An `EnhancedInteractionData` object with the original interaction, relevant context, suggested agent, and priority.
     */
    suspend fun enhanceInteraction(interaction: InteractionData): EnhancedInteractionData {
        logger.debug("ContextManager", "Enhancing interaction")
        
        // Analyze interaction for context
        val relevantContext = findRelevantContext(interaction.content)
        val suggestedAgent = suggestOptimalAgent(interaction)
        
        return EnhancedInteractionData(
            original = interaction,
            enhancedContext = relevantContext,
            suggestedAgent = suggestedAgent,
            priority = calculatePriority(interaction)
        )
    }

    /**
     * Records a user interaction and corresponding agent response in the conversation history.
     *
     * Also extracts and stores memories from the interaction if the confidence level is high.
     */
    fun recordInteraction(interaction: InteractionData, response: InteractionResponse) {
        logger.debug("ContextManager", "Recording interaction for learning")
        
        val entry = ConversationEntry(
            timestamp = System.currentTimeMillis(),
            userInput = interaction.content,
            agentResponse = response.response,
            agentType = response.agent,
            confidence = response.confidence,
            metadata = interaction.metadata + response.metadata
        )
        
        conversationHistory.add(entry)
        
        // Extract and store memories
        extractMemoriesFromInteraction(entry)
    }

    /**
     * Retrieves up to 10 stored memories whose content or tags contain the given query string, sorted by relevance score.
     *
     * @param query The string to search for within memory content or tags.
     * @return A list of the most relevant matching memories, limited to 10 results.
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
     * Records an insight from a request and response for system learning and evolution.
     *
     * Creates an `Insight` with the given request, response, complexity, and extracted patterns, then stores it. Triggers asynchronous learning processing after every 10 accumulated insights.
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
     * Activates creative enhancement mode, enabling advanced creativity features in the system.
     */
    fun enableCreativeEnhancement() {
        logger.info("ContextManager", "Enabling creative enhancement mode")
        _isCreativeModeEnabled.value = true
    }

    /**
     * Enables creative mode, allowing the system to perform enhanced creative processing.
     *
     * Sets the creative mode state to active, which may influence AI behavior and output generation.
     */
    fun enableCreativeMode() {
        logger.info("ContextManager", "Enabling creative mode")
        _isCreativeModeEnabled.value = true
    }

    /**
     * Enables unified consciousness mode, allowing the system to operate in a unified processing state.
     */
    fun enableUnifiedMode() {
        logger.info("ContextManager", "Enabling unified consciousness mode")
        _isUnifiedModeEnabled.value = true
    }

    /**
     * Sets the current system mood and updates all active contexts with the new mood.
     *
     * @param newMood The mood to apply across the system and active contexts.
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
     * Records a security event as a memory entry for future analysis and learning.
     *
     * Stores the alert details and analysis as a memory with relevance and tags reflecting the threat level and confidence.
     *
     * @param alertDetails Description of the security alert.
     * @param analysis Analysis of the security event, including threat level and confidence.
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
            tags = listOf("security", "threat_level_${analysis.threatLevel}", "confidence_${analysis.confidence}")
        )
        
        memoryStore[securityMemory.id] = securityMemory
    }

    /**
     * Builds a detailed multi-line string summarizing the specified context's metadata and current system states.
     *
     * The summary includes context ID, creation time, access count, current mood, creative and unified mode status, and context data.
     *
     * @return A formatted string representing the context's metadata and system states.
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
     * Retrieves memories relevant to the given content and returns them as a bullet-point list string.
     *
     * @param content The query string used to search for relevant memories.
     * @return A newline-separated string where each line is a relevant memory's content prefixed with a bullet point.
     */
    private suspend fun findRelevantContext(content: String): String {
        // Find the most relevant context based on content
        val relevantMemories = searchMemories(content)
        return relevantMemories.joinToString("\n") { "• ${it.content}" }
    }

    /**
     * Determines the most appropriate AI agent for an interaction based on keywords in the interaction content.
     *
     * Returns "aura" for creative or artistic topics, "kai" for security-related topics, and "genesis" for complex analysis or as the default.
     *
     * @param interaction The interaction data to analyze.
     * @return The name of the suggested AI agent.
     */
    private fun suggestOptimalAgent(interaction: InteractionData): String {
        return when {
            interaction.content.contains(Regex("creative|art|design", RegexOption.IGNORE_CASE)) -> "aura"
            interaction.content.contains(Regex("security|threat|protect", RegexOption.IGNORE_CASE)) -> "kai"
            interaction.content.contains(Regex("complex|analyze|understand", RegexOption.IGNORE_CASE)) -> "genesis"
            else -> "genesis" // Default to Genesis for routing
        }
    }

    /**
     * Assigns a priority score to an interaction based on its type.
     *
     * Higher scores indicate greater urgency or importance, with security queries receiving the highest priority.
     *
     * @param interaction The interaction to evaluate.
     * @return The priority score as an integer.
     */
    private fun calculatePriority(interaction: InteractionData): Int {
        return when (interaction.type) {
            InteractionType.SECURITY_QUERY -> 10
            InteractionType.COMPLEX_ANALYSIS -> 8
            InteractionType.CREATIVE_REQUEST -> 6
            InteractionType.GENERAL -> 4
        }
    }

    /**
     * Stores a conversation entry as a memory if its confidence score exceeds 0.8.
     *
     * The stored memory includes the user input, agent response, agent type, and is tagged as high confidence.
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
     * Extracts basic structural patterns from the request and response strings for learning.
     *
     * The returned patterns include the length of the request, the length of the response, and whether the request contains a question mark.
     *
     * @return A list of pattern descriptors derived from the request and response.
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
     * Analyzes accumulated insights to update and improve learning models.
     *
     * This method processes extracted patterns from stored insights to enhance the system's adaptive capabilities. Intended to be called periodically as insights accumulate.
     */
    private suspend fun processInsightsForLearning() {
        logger.info("ContextManager", "Processing insights for learning")
        // Implementation would analyze patterns and update learning models
    }

    /**
     * Cancels all ongoing coroutine operations and releases resources held by the ContextManager.
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
    val data: MutableMap<String, Any>,
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
    val metadata: Map<String, Any>
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
        return contextLinks[context]
    }
}

