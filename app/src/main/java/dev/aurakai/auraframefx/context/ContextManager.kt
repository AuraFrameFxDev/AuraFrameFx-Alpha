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
     * Creates and registers a new context with the specified ID and optional initial data.
     *
     * Initializes context metadata, including creation timestamp, access count, and last access time, for future tracking and enhancement.
     *
     * @param contextId The unique identifier for the new context.
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
     * Returns a detailed string summary of the specified context, including its metadata and current system states.
     *
     * Increments the context's access count and updates its last access time if the context exists. If the context is not found, returns a fallback message.
     *
     * @param contextId The unique identifier of the context to summarize.
     * @return A multi-line string with context details, or "Context not available" if the context does not exist.
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
     * Enhances an interaction by analyzing its content to retrieve relevant memories, suggest the most suitable AI agent, and assign a priority score.
     *
     * Searches for contextually related memories, determines the optimal agent based on interaction keywords, and calculates a priority level. Returns an `EnhancedInteractionData` object containing the original interaction, enhanced context string, suggested agent, and priority score.
     *
     * @param interaction The interaction data to be enhanced.
     * @return An `EnhancedInteractionData` object with the original interaction, relevant context, suggested agent, and priority score.
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
     * Adds a user interaction and agent response to the conversation history.
     *
     * If the agent's confidence is above 0.8, also extracts and stores a memory from the interaction.
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
     * Searches stored memories for entries whose content or tags match the given query, returning up to 10 results sorted by relevance.
     *
     * @param query The search string to match against memory content or tags.
     * @return A list of up to 10 memories most relevant to the query.
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
     * Records an insight from a request and response for system learning and adaptation.
     *
     * Creates an `Insight` containing the request, response, complexity, and extracted structural patterns, then stores it for future analysis. Initiates asynchronous learning processing after every 10 accumulated insights.
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
     * Enables creative enhancement mode, activating advanced creativity features within the system.
     */
    fun enableCreativeEnhancement() {
        logger.info("ContextManager", "Enabling creative enhancement mode")
        _isCreativeModeEnabled.value = true
    }

    /**
     * Activates creative mode, enabling enhanced creative processing within the system.
     *
     * Sets the creative mode state to active, which may affect AI behavior and output generation.
     */
    fun enableCreativeMode() {
        logger.info("ContextManager", "Enabling creative mode")
        _isCreativeModeEnabled.value = true
    }

    /**
     * Activates unified consciousness mode, enabling the system to operate in a unified processing state.
     */
    fun enableUnifiedMode() {
        logger.info("ContextManager", "Enabling unified consciousness mode")
        _isUnifiedModeEnabled.value = true
    }

    /**
     * Updates the current system mood and propagates the new mood to all active contexts.
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
     * Records a security event as a memory entry for future analysis and learning.
     *
     * Creates and stores a memory containing the alert details and analysis, with relevance and tags based on the threat level and confidence.
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
     * Constructs a formatted multi-line summary of the given context's metadata and current system states.
     *
     * The summary includes the context ID, creation timestamp, access count, current mood, creative mode status, unified mode status, and the context's data map.
     *
     * @return A formatted string detailing the context's metadata and relevant system states.
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
     * Searches for memories relevant to the provided content and formats them as a bullet-point list.
     *
     * @param content The text used to search for related memories.
     * @return A string with each relevant memory's content on a new line, prefixed by a bullet point.
     */
    private suspend fun findRelevantContext(content: String): String {
        // Find the most relevant context based on content
        val relevantMemories = searchMemories(content)
        return relevantMemories.joinToString("\n") { "• ${it.content}" }
    }

    /**
     * Suggests the most suitable AI agent for a given interaction based on content keywords.
     *
     * Returns "aura" for creative or artistic topics, "kai" for security-related topics, and "genesis" for complex analysis or as the default agent.
     *
     * @param interaction The interaction data to evaluate.
     * @return The suggested AI agent's name.
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
     * Determines the priority level of an interaction based on its type.
     *
     * Security queries are assigned the highest priority, followed by complex analysis, creative requests, and general interactions.
     *
     * @param interaction The interaction whose priority is being calculated.
     * @return An integer representing the priority score.
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
     * Saves a conversation entry as a memory if its confidence score is greater than 0.8.
     *
     * The memory captures the user input, agent response, agent type, confidence score, timestamp, and relevant tags.
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
     * Extracts simple structural patterns from the request and response strings.
     *
     * The patterns include the length of the request, the length of the response, and whether the request contains a question mark.
     *
     * @return A list of pattern descriptors representing basic features of the request and response.
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
     * Processes accumulated insights to refine and update internal learning models.
     *
     * Intended to be invoked periodically as new insights are collected, enabling the system to adapt based on extracted patterns and recent interactions.
     */
    private suspend fun processInsightsForLearning() {
        logger.info("ContextManager", "Processing insights for learning")
        // Implementation would analyze patterns and update learning models
    }

    /**
     * Cancels all ongoing coroutine operations and releases resources used by the ContextManager.
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

