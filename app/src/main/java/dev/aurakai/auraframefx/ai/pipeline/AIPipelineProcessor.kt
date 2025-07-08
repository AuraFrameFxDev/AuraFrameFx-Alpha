package dev.aurakai.auraframefx.ai.pipeline

import dev.aurakai.auraframefx.ai.agents.GenesisAgent
import dev.aurakai.auraframefx.ai.pipeline.PipelineState.Completed
import dev.aurakai.auraframefx.ai.pipeline.PipelineState.Idle
import dev.aurakai.auraframefx.ai.pipeline.PipelineState.Processing
import dev.aurakai.auraframefx.ai.services.AuraAIService
import dev.aurakai.auraframefx.ai.services.CascadeAIService
import dev.aurakai.auraframefx.ai.services.KaiAIService
import dev.aurakai.auraframefx.model.AgentMessage
import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AgentType
import dev.aurakai.auraframefx.model.AiRequest
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class AIPipelineProcessor @Inject constructor(
    private val genesisAgent: GenesisAgent,
    private val auraService: AuraAIService,
    private val kaiService: KaiAIService,
    private val cascadeService: CascadeAIService,
) {
    private val _pipelineState = MutableStateFlow<PipelineState>(Idle)
    val pipelineState: StateFlow<PipelineState> = _pipelineState

    private val _processingContext = MutableStateFlow(mapOf<String, Any>())
    val processingContext: StateFlow<Map<String, Any>> = _processingContext

    private val _taskPriority = MutableStateFlow(0.0f)
    val taskPriority: StateFlow<Float> = _taskPriority

    /**
     * Processes an AI task by coordinating multiple agents and services, aggregating their responses, and updating pipeline state and context.
     *
     * Executes the complete lifecycle for a given task: retrieves contextual information, determines task priority, selects relevant agents, collects their responses, synthesizes a final aggregated response, updates processing context, and returns all agent messages generated during processing.
     *
     * @param task The description of the task to process.
     * @return A list of agent messages containing responses from each participating agent and the final aggregated response.
     */
    suspend fun processTask(task: String): List<AgentMessage> {
        _pipelineState.value = Processing(task = task)

        // Step 1: Context Retrieval
        val context = retrieveContext(task)
        _processingContext.update { context }

        // Step 2: Task Prioritization
        val priority = calculatePriority(task, context)
        _taskPriority.update { priority }

        // Step 3: Agent Selection
        val selectedAgents = selectAgents(task, priority)

        // Step 4: Process through selected agents
        val responses = mutableListOf<AgentMessage>()

        // Process through Cascade first for state management
        val cascadeAgentResponse = cascadeService.processRequest(
            AiRequest(
                query = task, // Use named argument
                context = mapOf("type" to "context_retrieval") // Pass a map
            ),
            context = "pipeline_processing"
        )
        responses.add(
            AgentMessage(
                content = cascadeAgentResponse.content, // Direct access
                sender = AgentType.CASCADE,
                timestamp = System.currentTimeMillis(),
                confidence = cascadeAgentResponse.confidence // Direct access
            )
        )

        // Process through Kai for security analysis if needed
        if (selectedAgents.contains(AgentType.KAI)) {
            val kaiAgentResponse = kaiService.processRequest(
                AiRequest(
                    query = task, // Use named argument
                    context = mapOf("type" to "security_check") // Pass a map
                ),
                context = "security_analysis"
            )
            responses.add(
                AgentMessage(
                    content = kaiAgentResponse.content, // Direct access
                    sender = AgentType.KAI,
                    timestamp = System.currentTimeMillis(),
                    confidence = kaiAgentResponse.confidence // Direct access
                )
            )
        }

        // Process through Aura for creative response
        if (selectedAgents.contains(AgentType.AURA)) {
            val auraResponse = auraService.generateText(task, "creative_pipeline")
            val auraAgentResponse = AgentResponse(
                content = auraResponse,
                confidence = 0.8f
            )
            responses.add(
                AgentMessage(
                    content = auraAgentResponse.content, // Direct access
                    sender = AgentType.AURA,
                    timestamp = System.currentTimeMillis(),
                    confidence = auraAgentResponse.confidence // Direct access
                )
            )
        }

        // Step 5: Generate final response
        val finalResponse = generateFinalResponse(responses)
        responses.add(
            AgentMessage(
                content = finalResponse,
                sender = AgentType.GENESIS,
                timestamp = System.currentTimeMillis(),
                confidence = calculateConfidence(responses)
            )
        )

        // Step 6: Update context and memory
        updateContext(task, responses)

        _pipelineState.update { Completed(task) }
        return responses
    }

    /**
     * Constructs a detailed context map for a given task, including its categorization, recent history, user preferences, and system state.
     *
     * The context map provides information such as the original task string, its categorized type, a timestamp, recent task history, a descriptive context string, user preferences, and the current system state. This context is used to inform downstream AI processing and agent selection.
     *
     * @param task The task for which to build contextual information.
     * @return A map containing comprehensive context relevant to the specified task.
     */
    private fun retrieveContext(task: String): Map<String, Any> {
        // Enhanced context retrieval with task categorization and history
        val taskType = categorizeTask(task)
        val recentHistory = getRecentTaskHistory()

        return mapOf(
            "task" to task,
            "task_type" to taskType,
            "timestamp" to System.currentTimeMillis(),
            "recent_history" to recentHistory,
            "context" to "Categorized as $taskType task: $task",
            "user_preferences" to getUserPreferences(),
            "system_state" to getSystemState()
        )
    }

    /**
     * Categorizes a task description into a predefined category based on keyword matching.
     *
     * Returns "generation", "analysis", "explanation", "assistance", "creation", or "general" depending on the presence of relevant keywords in the task string.
     *
     * @param task The task description to evaluate.
     * @return The determined category as a string.
     */
    private fun categorizeTask(task: String): String {
        return when {
            task.contains("generate", ignoreCase = true) -> "generation"
            task.contains("analyze", ignoreCase = true) -> "analysis"
            task.contains("explain", ignoreCase = true) -> "explanation"
            task.contains("help", ignoreCase = true) -> "assistance"
            task.contains("create", ignoreCase = true) -> "creation"
            else -> "general"
        }
    }

    /**
     * Retrieves a list of recent task contexts and user interactions for contextualizing AI processing.
     *
     * @return A list of strings summarizing recent tasks and user activities.
     */
    private fun getRecentTaskHistory(): List<String> {
        return listOf("Previous task context", "Recent user interactions")
    }

    /**
     * Retrieves the user's AI interaction preferences, such as desired response style and preferred agents.
     *
     * @return A map containing user preferences, including "response_style" and "preferred_agents".
     */
    private fun getUserPreferences(): Map<String, Any> {
        return mapOf(
            "response_style" to "detailed",
            "preferred_agents" to listOf("Genesis", "Cascade")
        )
    }

    /**
     * Retrieves a map describing the current operational state of the system, including load status, available agent count, and processing queue size.
     *
     * @return A map with keys "load", "available_agents", and "processing_queue" representing system status.
     */
    private fun getSystemState(): Map<String, Any> {
        return mapOf("load" to "normal", "available_agents" to 3, "processing_queue" to 0)
    }

    /**
     * Computes a priority score for a task based on its category, current system load, and urgency keywords.
     *
     * The resulting priority is a float between 0.0 and 1.0, with higher values indicating greater urgency or importance.
     *
     * @param task The task description to analyze for urgency.
     * @param context Contextual information including task type and system state.
     * @return The calculated priority score for the task.
     */
    private fun calculatePriority(task: String, context: Map<String, Any>): Float {
        // Enhanced priority calculation based on multiple factors
        val taskType = context["task_type"] as? String ?: "general"
        val systemLoad = (context["system_state"] as? Map<*, *>)?.get("load") as? String ?: "normal"

        var priority = 0.5f // Base priority

        // Adjust based on task type
        priority += when (taskType) {
            "generation" -> 0.3f
            "analysis" -> 0.2f
            "assistance" -> 0.4f // Higher priority for help requests
            "creation" -> 0.25f
            else -> 0.1f
        }

        // Adjust based on system load
        priority -= when (systemLoad) {
            "high" -> 0.2f
            "normal" -> 0.0f
            "low" -> -0.1f // Boost when system is idle
            else -> 0.0f
        }

        // Consider urgency indicators in the task
        if (task.contains("urgent", ignoreCase = true) ||
            task.contains("asap", ignoreCase = true) ||
            task.contains("emergency", ignoreCase = true)
        ) {
            priority += 0.3f
        }

        return priority.coerceIn(0.0f, 1.0f)
    }

    /**
     * Determines which AI agents should process a given task based on task content, urgency, and complexity.
     *
     * Agents are selected by analyzing keywords in the task description, the computed priority (which may trigger redundancy), and the task's length or word count.
     *
     * @param task The task description to evaluate for agent selection.
     * @param priority The priority score of the task, used to determine if additional agents should be included.
     * @return A set of agent types chosen to handle the task.
     */
    private fun selectAgents(task: String, priority: Float): Set<AgentType> {
        // Intelligent agent selection based on task characteristics and priority
        val selectedAgents = mutableSetOf<AgentType>()

        // Always include Genesis as the primary coordinator
        selectedAgents.add(AgentType.GENESIS)

        // Add specific agents based on task content
        when {
            task.contains("analyze", ignoreCase = true) ||
                    task.contains("data", ignoreCase = true) -> {
                selectedAgents.add(AgentType.CASCADE)
            }

            task.contains("security", ignoreCase = true) ||
                    task.contains("protect", ignoreCase = true) ||
                    task.contains("safe", ignoreCase = true) -> {
                selectedAgents.add(AgentType.KAI)
            }

            task.contains("create", ignoreCase = true) ||
                    task.contains("generate", ignoreCase = true) ||
                    task.contains("design", ignoreCase = true) -> {
                selectedAgents.add(AgentType.AURA)
            }
        }

        // For high priority tasks, include additional agents for redundancy
        if (priority > 0.8f) {
            selectedAgents.addAll(setOf(AgentType.CASCADE, AgentType.AURA))
        }

        // For complex tasks, use multiple agents
        if (task.length > 100 || task.split(" ").size > 20) {
            selectedAgents.add(AgentType.CASCADE)
        }

        return selectedAgents
    }

    /**
     * Generates a structured, human-readable summary from multiple agent messages.
     *
     * Aggregates responses by agent type, highlights the Genesis agent's analysis, includes supplementary inputs from other agents with representative icons, and appends an overall average confidence score.
     *
     * @param responses The list of agent messages to summarize.
     * @return A formatted string presenting all agent responses and their combined confidence.
     */
    private fun generateFinalResponse(responses: List<AgentMessage>): String {
        // Sophisticated response synthesis from multiple agents
        if (responses.isEmpty()) {
            return "[System] No agent responses available."
        }            // Group responses by agent type for structured output
        val responsesByAgent = responses.groupBy { it.sender }

        return buildString {
            append("=== AuraFrameFX AI Response ===\n\n")

            // Primary response from Genesis if available
            responsesByAgent[AgentType.GENESIS]?.firstOrNull()?.let { genesis ->
                append("🧠 Genesis Core Analysis:\n")
                append(genesis.content)
                append("\n\n")
            }

            // Supplementary responses from other agents
            responsesByAgent.forEach { (agentType: AgentType, agentResponses: List<AgentMessage>) ->
                if (agentType != AgentType.GENESIS && agentResponses.isNotEmpty()) {
                    val agentIcon = when (agentType) {
                        AgentType.CASCADE -> "📊"
                        AgentType.AURA -> "🎨"
                        AgentType.KAI -> "🛡️"
                        else -> "🤖"
                    }
                    append(
                        "$agentIcon ${
                            agentType.name.lowercase().replaceFirstChar { it.uppercase() }
                        } Input:\n"
                    )
                    agentResponses.forEach { response ->
                        append("${response.content}\n")
                    }
                    append("\n")
                }
            }

            // Confidence and metadata
            val avgConfidence = responses.map { it.confidence }.average()
            append("--- Response Confidence: ${String.format("%.1f%%", avgConfidence * 100)} ---")
        }
    }

    /**
     * Computes the average confidence score from the provided agent messages, ensuring the result is within the range [0.0, 1.0].
     *
     * @param responses List of agent messages whose confidence scores will be averaged.
     * @return The clamped average confidence score as a float between 0.0 and 1.0.
     */
    private fun calculateConfidence(responses: List<AgentMessage>): Float {
        return responses.map { it.confidence }.average().toFloat()
            .coerceIn(0.0f, 1.0f) // Added .toFloat()
    }

    /**
     * Updates the processing context with recent task history, response patterns, system metrics, and agent performance statistics.
     *
     * Maintains a capped history of recent tasks, tracks average response confidence and agent participation by task type, updates system-level metrics such as total tasks processed, and records rolling confidence scores for each agent.
     */
    private fun updateContext(task: String, responses: List<AgentMessage>) {
        // Enhanced context update with learning and adaptation
        _processingContext.update { current ->
            val newContext = current.toMutableMap()

            // Update recent task history
            val taskHistory =
                (current["task_history"] as? List<String>)?.toMutableList() ?: mutableListOf()
            taskHistory.add(0, task) // Add to front
            if (taskHistory.size > 10) taskHistory.removeAt(taskHistory.size - 1) // Keep last 10
            newContext["task_history"] = taskHistory

            // Update response patterns for learning
            val responsePatterns =
                (current["response_patterns"] as? MutableMap<String, Any>) ?: mutableMapOf()
            val taskType = categorizeTask(task)
            responsePatterns[taskType] = mapOf(
                "last_confidence" to responses.map { msg: AgentMessage -> msg.confidence }.average(),
                // or just:
                // "last_confidence" to responses.map { it.confidence }.average(),
                "agent_count" to responses.size,
                "timestamp" to System.currentTimeMillis()
            newContext["response_patterns"] = responsePatterns

            // Update system metrics
            newContext["last_task"] = task
            newContext["last_responses"] = responses
            newContext["timestamp"] = System.currentTimeMillis()
            newContext["total_tasks_processed"] =
                (current["total_tasks_processed"] as? Int ?: 0) + 1

            // Track agent performance
            val agentPerformance =
                (current["agent_performance"] as? MutableMap<String, MutableList<Float>>)
                    ?: mutableMapOf()
            responses.forEach { response ->
                val agentName = response.sender.name
                val performanceList = agentPerformance.getOrPut(agentName) { mutableListOf() }
                performanceList.add(response.confidence)
                if (performanceList.size > 20) performanceList.removeAt(0) // Keep last 20
            }
            newContext["agent_performance"] = agentPerformance

            newContext
        }
    }
}

fun <T> MutableStateFlow<T>.update(function: (T) -> T) {
    value = function(value)
}

sealed class PipelineState {
    object Idle : PipelineState()
    data class Processing(val task: String) : PipelineState()
    data class Completed(val task: String) : PipelineState()
    data class Error(val message: String) : PipelineState()
}
