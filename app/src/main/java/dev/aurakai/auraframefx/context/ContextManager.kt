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
     * Creates and registers a new context with the given ID and optional initial data.
     *
     * Initializes context metadata including creation timestamp, access count, and last access time for future tracking and management.
     *
     * @param contextId The unique identifier for the new context.
     * @param initialData Optional data to associate with the context upon creation.
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
     * Returns a detailed summary string for the specified context, including its metadata and current system states.
     *
     * Increments the access count and updates the last access time for the context if it exists. If the context does not exist, returns a fallback message.
     *
     * @param contextId The unique identifier of the context to summarize.
     * @return A string containing the enhanced context details, or a fallback message if the context is not found.
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
     * Enhances an interaction by providing relevant context, suggesting the optimal AI agent, and assigning a priority score.
     *
     * Searches stored memories for information related to the interaction content, determines the most suitable agent, and calculates a priority level. Returns an `EnhancedInteractionData` object containing these enhancements.
     *
     * @param interaction The interaction to be enhanced.
     * @return An `EnhancedInteractionData` object with relevant context, agent suggestion, and priority score.
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
     * Records a user interaction and the corresponding agent response in the conversation history.
     *
     * If the agent's confidence is above 0.8, extracts and stores a memory from the interaction for future context enhancement.
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
     * Searches stored memories for entries matching the query in content or tags and returns up to 10 most relevant results.
     *
     * @param query The search string to match against memory content or tags.
     * @return A list of up to 10 memories sorted by relevance to the query.
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
     * Records a learning insight by storing the request, response, complexity label, and extracted patterns for future system learning.
     *
     * Triggers asynchronous processing of accumulated insights after every 10 entries to support ongoing system evolution.
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
     * Activates creative mode, enabling enhanced creative processing within the system.
     *
     * Sets the system state to creative mode, which may influence AI behavior and output generation.
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
     * Sets the system's current mood and updates all active contexts with the new mood.
     *
     * @param newMood The mood to apply system-wide and to each active context.
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
     * Stores the provided alert details and analysis as a memory, assigning relevance and tags based on the threat level and confidence.
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
     * Returns a detailed multi-line summary of the specified context, including metadata and current system states.
     *
     * The summary contains the context ID, creation time, access count, current mood, creative mode status, unified mode status, and the context's data.
     *
     * @return A formatted string describing the context and system states.
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
     * Retrieves memories relevant to the given content and returns them as a bullet-point list.
     *
     * @param content The text to match against stored memories.
     * @return A string listing each relevant memory's content as a bullet point, or an empty string if none are found.
     */
    private suspend fun findRelevantContext(content: String): String {
        // Find the most relevant context based on content
        val relevantMemories = searchMemories(content)
        return relevantMemories.joinToString("\n") { "• ${it.content}" }
    }

    /**
     * Determines the most suitable AI agent for an interaction based on keywords in the content.
     *
     * Returns "aura" for creative or artistic topics, "kai" for security-related topics, and "genesis" for complex analysis or as the default agent.
     *
     * @param interaction The interaction data to analyze.
     * @return The suggested AI agent: "aura", "kai", or "genesis".
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
     * Returns a priority score for an interaction based on its type.
     *
     * Security queries are assigned the highest priority, followed by complex analyses, creative requests, and general interactions.
     *
     * @param interaction The interaction whose priority is being determined.
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
     * Extracts a memory from a conversation entry and stores it if the confidence score exceeds 0.8.
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
     * Extracts basic structural patterns from the request and response for learning insights.
     *
     * The extracted patterns include the length of the request, the length of the response, and whether the request contains a question mark.
     *
     * @return A list of string descriptors representing the identified patterns.
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
     * Processes accumulated learning insights to update internal adaptive models.
     *
     * Analyzes extracted patterns from stored insights to enhance the system's learning and adaptation capabilities. Intended to be called asynchronously as new insights are accumulated.
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

