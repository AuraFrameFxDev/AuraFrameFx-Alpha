package dev.aurakai.auraframefx.context

import dev.aurakai.auraframefx.utils.AuraFxLogger
import dev.aurakai.auraframefx.ai.*
import dev.aurakai.auraframefx.model.InteractionData
import dev.aurakai.auraframefx.model.EnhancedInteractionData
import dev.aurakai.auraframefx.model.InteractionResponse
import dev.aurakai.auraframefx.model.InteractionType
import dev.aurakai.auraframefx.model.SecurityAnalysis
import dev.aurakai.auraframefx.model.ThreatLevel
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.Contextual
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
     * Creates and registers a new context with the given ID and optional initial data.
     *
     * Initializes context metadata and adds the context to the active contexts map for tracking.
     *
     * @param contextId The unique identifier for the context.
     * @param initialData Optional data to initialize the context.
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
     * Enhances and returns a string summary of the specified context, including its metadata and current system states.
     *
     * Updates the context's access count and last access time if found. Returns a fallback message if the context does not exist.
     *
     * @param contextId The unique identifier of the context to enhance.
     * @return An enhanced string representation of the context, or a fallback message if not found.
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
     * Enhances a user interaction by retrieving relevant context, suggesting an optimal AI agent, and assigning a processing priority.
     *
     * Analyzes the interaction content to find related memories, determines the most suitable agent based on content patterns, and calculates a priority score. Returns an `EnhancedInteractionData` object containing the original interaction content, type, timestamp, relevant context, suggested agent, and priority.
     *
     * @param interaction The user interaction data to be enhanced.
     * @return An `EnhancedInteractionData` object with enriched context, agent suggestion, and priority.
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
            enrichmentData = mapOf("suggested_agent" to suggestedAgent, "priority" to calculatePriority(interaction).toString())
        )
    }

    /**
     * Records a user interaction and corresponding agent response in the conversation history.
     *
     * Also extracts and stores high-confidence memories from the interaction for future retrieval and learning.
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
     * Searches stored memories for entries whose content or tags contain the specified query string, case-insensitive.
     *
     * Returns up to 10 memories with the highest relevance scores that match the query.
     *
     * @param query The string to search for within memory content and tags.
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
     * Records an insight from a request and response, storing it for system learning and triggering learning processing periodically.
     *
     * Creates an `Insight` with the given request, response, and complexity, extracts pattern descriptors, and adds it to the insight store. Initiates asynchronous learning processing every 10 insights.
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
     * Enables creative enhancement mode, allowing AI agents to operate with increased creativity.
     */
    fun enableCreativeEnhancement() {
        logger.info("ContextManager", "Enabling creative enhancement mode")
        _isCreativeModeEnabled.value = true
    }

    /**
     * Enables creative mode, allowing the system to utilize enhanced creative processing capabilities.
     */
    fun enableCreativeMode() {
        logger.info("ContextManager", "Enabling creative mode")
        _isCreativeModeEnabled.value = true
    }

    /**
     * Enables unified consciousness mode, allowing AI agents to operate within a unified context.
     */
    fun enableUnifiedMode() {
        logger.info("ContextManager", "Enabling unified consciousness mode")
        _isUnifiedModeEnabled.value = true
    }

    /**
     * Sets the system mood and updates all active contexts with the new mood.
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
     * Stores the event details and threat analysis as a memory object, assigning relevance based on threat level and tagging for security context.
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
     * Generates a multi-line summary string containing metadata and current system states for the specified context.
     *
     * The summary includes the context ID, creation timestamp, access count, current mood, creative mode status, unified mode status, and context-specific data.
     *
     * @param context The context data to summarize.
     * @return A formatted string representing the enhanced context information.
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
     * Returns a bullet-pointed list of memory contents most relevant to the given content string.
     *
     * Searches stored memories for entries relevant to the input and concatenates their contents into a newline-separated string, each prefixed with a bullet point.
     *
     * @param content The input string to match against stored memories.
     * @return A string containing relevant memory contents, each on a new line with a bullet point.
     */
    private suspend fun findRelevantContext(content: String): String {
        // Find the most relevant context based on content
        val relevantMemories = searchMemories(content)
        return relevantMemories.joinToString("\n") { "• ${it.content}" }
    }

    /**
     * Determines the most appropriate AI agent for an interaction by analyzing keywords in the interaction content.
     *
     * Returns "aura" for creative or artistic topics, "kai" for security-related topics, and "genesis" for complex analysis or as the default.
     *
     * @param interaction The interaction data whose content is evaluated for agent selection.
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
     * Returns a priority score for the given interaction based on its type.
     *
     * Assigns higher scores to interaction types such as "security" and "analysis", with lower scores for general or unknown types.
     *
     * @param interaction The interaction to evaluate.
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
     * Extracts a high-confidence memory from a conversation entry and stores it in the memory store.
     *
     * A memory is created and saved if the entry's confidence score exceeds 0.8, capturing the user input, agent response, agent type, and relevant tags.
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
     * Extracts simple pattern descriptors from the request and response strings for use in learning models.
     *
     * The returned list includes the length of the request, the length of the response, and whether the request contains a question mark.
     *
     * @return A list of strings describing basic features of the request and response.
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
     * Analyzes accumulated insights to update learning models.
     *
     * This is a placeholder for asynchronous processing of extracted patterns from insights to improve system learning.
     */
    private suspend fun processInsightsForLearning() {
        logger.info("ContextManager", "Processing insights for learning")
        // Implementation would analyze patterns and update learning models
    }

    /**
     * Cancels all coroutines and releases resources managed by the ContextManager.
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

