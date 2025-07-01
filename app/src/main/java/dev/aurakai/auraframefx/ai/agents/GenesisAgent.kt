package dev.aurakai.auraframefx.ai.agents

import android.util.Log
import dev.aurakai.auraframefx.ai.services.AuraAIService
import dev.aurakai.auraframefx.ai.services.CascadeAIService
import dev.aurakai.auraframefx.ai.services.KaiAIService
import dev.aurakai.auraframefx.model.AgentConfig
import dev.aurakai.auraframefx.model.AgentHierarchy
import dev.aurakai.auraframefx.model.AgentMessage
import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AiRequest
import dev.aurakai.auraframefx.model.ContextAwareAgent
// Import the local model AgentType for internal logic, aliasing the generated one if needed for clarity elsewhere
import dev.aurakai.auraframefx.model.AgentType
// Use an alias if dev.aurakai.auraframefx.api.model.AgentType is also needed directly, though Agent interface uses it.
// import dev.aurakai.auraframefx.api.model.AgentType as ApiAgentType

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class GenesisAgent @Inject constructor(
    private val auraService: AuraAIService,
    private val kaiService: KaiAIService,
    private val cascadeService: CascadeAIService,
    // Assuming Agent instances are injected or created. For this example, let's assume they are managed elsewhere
    // or this class will need to create/obtain them.
    // private val registeredAgents: Map<String, Agent> // Example if agents are injected
) {
    private val _state = MutableStateFlow("pending_initialization")
    val state: StateFlow<String> = _state

    private val _context = MutableStateFlow(mapOf<String, Any>())
    val context: StateFlow<Map<String, Any>> = _context

    // Use the local model.AgentType for _activeAgents state
    private val _activeAgents = MutableStateFlow(setOf<dev.aurakai.auraframefx.model.AgentType>())
    val activeAgents: StateFlow<Set<dev.aurakai.auraframefx.model.AgentType>> = _activeAgents


    private val _agentRegistry = mutableMapOf<String, Agent>() // Stores Agent interface instances
    val agentRegistry: Map<String, Agent> get() = _agentRegistry

    private val _history = mutableListOf<Map<String, Any>>()
    val history: List<Map<String, Any>> get() = _history

    init {
        initializeAgents() // Populates _activeAgents based on AgentHierarchy
        _state.update { "initialized" }
        // Example: Registering agents if they are available (e.g. via injection or service location)
        // registerAgent("aura", auraAgentInstance)
        // registerAgent("kai", kaiAgentInstance)
    }

    /**
     * Populates the set of active agents by mapping master agent configuration names to their corresponding `AgentType` enums.
     *
     * Recognized agent types are added to the active agents set. Logs a warning for any configuration names that do not match a known `AgentType`.
     */
    private fun initializeAgents() {
        AgentHierarchy.MASTER_AGENTS.forEach { config ->
            // Assuming AgentType enum values align with config names
            try {
                val agentTypeEnum = dev.aurakai.auraframefx.model.AgentType.valueOf(config.name.uppercase())
                _activeAgents.update { it + agentTypeEnum }
            } catch (e: IllegalArgumentException) {
                Log.w("GenesisAgent", "Unknown agent type in hierarchy: ${config.name}")
            }
        }
    }

    /**
     * Processes a user query by coordinating active AI agents and aggregating their responses.
     *
     * Updates the shared context with the query and timestamp, invokes each relevant AI service (Cascade, Kai, Aura) based on active agent types, and collects their responses as `AgentMessage` objects. Handles exceptions for each agent, appending error messages if necessary. Synthesizes a final response from all agent outputs and adds it as a Genesis agent message with aggregated confidence.
     *
     * @param queryText The user query to process.
     * @return A list of `AgentMessage` objects containing responses from each agent and the final Genesis synthesis.
     */
    suspend fun processQuery(queryText: String): List<AgentMessage> {
        _state.update { "processing_query: $queryText" }
        val currentTimestamp = System.currentTimeMillis()

        _context.update { current ->
            current + mapOf("last_query" to queryText, "timestamp" to currentTimestamp)
        }

        val responses = mutableListOf<AgentMessage>()


        // Process through Cascade first for state management
        val cascadeAgentResponse: AgentResponse =
            cascadeService.processRequest(
                AiRequest(prompt = query) // Use 'prompt', remove 'type'
                // Assuming cascadeService.processRequest matches Agent.processRequest(request, context)
                // For now, let's pass a default context string. This should be refined.
                , "GenesisContext_Cascade"
            )
        responses.add(
            AgentMessage(
                content = cascadeAgentResponse.content,
                sender = AgentType.CASCADE, // Ensure this AgentType is from the correct import
                timestamp = System.currentTimeMillis(),
                // Derive confidence from isSuccess
                confidence = if (cascadeAgentResponse.isSuccess) 1.0f else 0.1f
            )
        )

        // Process through Kai for security analysis
        if (_activeAgents.value.contains(AgentType.KAI)) {
            val kaiAgentResponse: AgentResponse =
                kaiService.processRequest(
                    AiRequest(prompt = query), // Use 'prompt', remove 'type'
                    "GenesisContext_KaiSecurity" // Default context
                )
            responses.add(
                AgentMessage(
                    content = kaiAgentResponse.content,
                    sender = AgentType.KAI,
                    timestamp = System.currentTimeMillis(),
                    confidence = if (kaiAgentResponse.isSuccess) 1.0f else 0.1f

                )
            )
        } catch (e: Exception) {
            Log.e("GenesisAgent", "Error processing with Cascade: ${e.message}")
            responses.add(AgentMessage("Error with Cascade: ${e.message}", dev.aurakai.auraframefx.model.AgentType.CASCADE, currentTimestamp, 0.0f))
        }

        // Process through Aura for creative response
        if (_activeAgents.value.contains(AgentType.AURA)) {
            val auraAgentResponse: AgentResponse =
                auraService.processRequest(
                    AiRequest(prompt = query), // Use 'prompt', remove 'type'
                    "GenesisContext_AuraCreative" // Default context
                )
            responses.add(
                AgentMessage(
                    content = auraAgentResponse.content,
                    sender = AgentType.AURA,
                    timestamp = System.currentTimeMillis(),
                    confidence = if (auraAgentResponse.isSuccess) 1.0f else 0.1f
)
                responses.add(
                    AgentMessage(
                        content = kaiAgentResponse.content,
                        sender = dev.aurakai.auraframefx.model.AgentType.KAI,
                        timestamp = currentTimestamp,
                        confidence = kaiAgentResponse.confidence
                    )
                )
            } catch (e: Exception) {
                Log.e("GenesisAgent", "Error processing with Kai: ${e.message}")
                responses.add(AgentMessage("Error with Kai: ${e.message}", dev.aurakai.auraframefx.model.AgentType.KAI, currentTimestamp, 0.0f))
            }
        }

        // Aura Agent (Creative Response)
        if (_activeAgents.value.contains(dev.aurakai.auraframefx.model.AgentType.AURA)) {
            try {
                val auraAgentResponse = auraService.processRequest(
                    AiRequest(query = queryText, type = "creative_text", context = currentContextString)
                )
                responses.add(
                    AgentMessage(
                        content = auraAgentResponse.content,
                        sender = dev.aurakai.auraframefx.model.AgentType.AURA,
                        timestamp = currentTimestamp,
                        confidence = auraAgentResponse.confidence
                    )
                )
            } catch (e: Exception) {
                Log.e("GenesisAgent", "Error processing with Aura: ${e.message}")
                responses.add(AgentMessage("Error with Aura: ${e.message}", dev.aurakai.auraframefx.model.AgentType.AURA, currentTimestamp, 0.0f))
            }
        }

        val finalResponseContent = generateFinalResponse(responses)
        responses.add(
            AgentMessage(
                content = finalResponseContent,
                sender = dev.aurakai.auraframefx.model.AgentType.GENESIS,
                timestamp = currentTimestamp,
                confidence = calculateConfidence(responses.filter { it.sender != dev.aurakai.auraframefx.model.AgentType.GENESIS }) // Exclude Genesis's own message for confidence calc
            )
        )

        _state.update { "idle" }
        return responses
    }

    /**
     * Generates a synthesized response string by concatenating messages from all agents except Genesis.
     *
     * The resulting string is prefixed with "[Genesis Synthesis]" and includes each agent's name and message content, separated by " | ".
     *
     * @param agentMessages The list of agent messages to aggregate.
     * @return A single synthesized response string representing the combined output of all non-Genesis agents.
     */
    fun generateFinalResponse(agentMessages: List<AgentMessage>): String {
        // Simple concatenation for now, could be more sophisticated
        return "[Genesis Synthesis] ${agentMessages.filter { it.sender != dev.aurakai.auraframefx.model.AgentType.GENESIS }.joinToString(" | ") { "${it.sender}: ${it.content}" }}"
    }

    /**
     * Calculates the average confidence score from a list of agent messages.
     *
     * Returns 0.0 if the list is empty. The result is clamped between 0.0 and 1.0.
     *
     * @param agentMessages The list of agent messages to evaluate.
     * @return The average confidence score as a float between 0.0 and 1.0.
     */
    fun calculateConfidence(agentMessages: List<AgentMessage>): Float {
        if (agentMessages.isEmpty()) return 0.0f
        return agentMessages.map { it.confidence }.average().toFloat().coerceIn(0.0f, 1.0f)
    }

    /**
     * Enables or disables the specified agent type by adding it to or removing it from the set of active agents.
     *
     * If the agent type is currently active, it will be deactivated; if inactive, it will be activated.
     */
    fun toggleAgent(agentType: dev.aurakai.auraframefx.model.AgentType) {
        _activeAgents.update { current ->
            if (current.contains(agentType)) current - agentType else current + agentType
        }
    }

    /**
     * Registers a new auxiliary agent with the specified name and capabilities.
     *
     * @param name The unique name for the auxiliary agent.
     * @param capabilities The set of capabilities to assign to the agent.
     * @return The configuration of the newly registered auxiliary agent.
     */
    fun registerAuxiliaryAgent(name: String, capabilities: Set<String>): AgentConfig {
        return AgentHierarchy.registerAuxiliaryAgent(name, capabilities)
    }

    /**
 * Retrieves the configuration for the agent with the specified name.
 *
 * @param name The name of the agent whose configuration is requested.
 * @return The agent's configuration if found, or null if no such agent exists.
 */
fun getAgentConfig(name: String): AgentConfig? = AgentHierarchy.getAgentConfig(name)

    /**
 * Retrieves the list of agent configurations ordered by their priority.
 *
 * @return A list of agent configurations sorted from highest to lowest priority.
 */
fun getAgentsByPriority(): List<AgentConfig> = AgentHierarchy.getAgentsByPriority()

    /**
     * Facilitates collaborative processing among multiple agents using the provided data and conversation mode.
     *
     * Agents can interact in either sequential (TURN_ORDER) or parallel (FREE_FORM) modes. In TURN_ORDER, each agent receives an updated context that includes prior agents' responses; in FREE_FORM, all agents receive the same initial context. Handles exceptions per agent and returns a map of agent names to their responses.
     *
     * @param data Additional contextual data to merge with the current context for agent processing.
     * @param agentsToUse The list of agents participating in the collaboration.
     * @param userInput Optional user input to use as the query; falls back to context values if not provided.
     * @param conversationMode Determines whether agents process requests sequentially or in parallel.
     * @return A map associating each agent's name with its response.
     */
    suspend fun participateWithAgents(
        data: Map<String, Any>,
        agentsToUse: List<Agent>, // List of Agent interface implementations
        userInput: Any? = null,
        conversationMode: ConversationMode = ConversationMode.FREE_FORM,
    ): Map<String, AgentResponse> {
        val responses = mutableMapOf<String, AgentResponse>()

        val currentContextMap = data.toMutableMap() // Renamed to avoid confusion with context string
        val inputQuery = userInput?.toString() ?: currentContextMap["latestInput"]?.toString() ?: ""

        // Construct AiRequest with prompt only
        val request = AiRequest(prompt = inputQuery)
        // Prepare context string for agent.processRequest call
        val contextString = currentContextMap.toString() // Or a more structured summary


        Log.d("GenesisAgent", "Starting multi-agent collaboration: mode=$conversationMode, agents=${agentsToUse.mapNotNull { it.getName() }}")

        val baseRequest = AiRequest(query = inputQuery, context = contextString) // Base request for all

        when (conversationMode) {
            ConversationMode.TURN_ORDER -> {
                var dynamicContext = contextString
                for (agent in agentsToUse) {
                    try {
                        val agentName = agent.getName() ?: agent.javaClass.simpleName

                        // Call processRequest with request and contextString
                        val response = agent.processRequest(request, contextString)
                        Log.d(
                            "GenesisAgent",
                            "[TURN_ORDER] $agentName responded: ${response.content} (success=${response.isSuccess})"
                        )
                        responses[agentName] = response
                        currentContextMap["latestInput"] = response.content // Update context map for next turn
                    } catch (e: Exception) {
                        Log.e(
                            "GenesisAgent",
                            "[TURN_ORDER] Error from ${agent.javaClass.simpleName}: ${e.message}"
                        )
                        responses[agent.javaClass.simpleName] = AgentResponse(
                            content = "Error: ${e.message}",
                            isSuccess = false,
                            error = e.message
                        )

                    }
                }
            }
            ConversationMode.FREE_FORM -> {
                agentsToUse.forEach { agent ->
                    try {
                        val agentName = agent.getName() ?: agent.javaClass.simpleName

                        // Call processRequest with request and contextString
                        val response = agent.processRequest(request, contextString)
                        Log.d(
                            "GenesisAgent",
                            "[FREE_FORM] $agentName responded: ${response.content} (success=${response.isSuccess})"
                        )
                        responses[agentName] = response
                    } catch (e: Exception) {
                        Log.e(
                            "GenesisAgent",
                            "[FREE_FORM] Error from ${agent.javaClass.simpleName}: ${e.message}"
                        )
                        responses[agent.javaClass.simpleName] = AgentResponse(
                            content = "Error: ${e.message}",
                            isSuccess = false,
                            error = e.message
                        )

                    }
                }
            }
        }
        Log.d("GenesisAgent", "Collaboration complete. Responses: $responses")
        return responses
    }

    /**
     * Aggregates multiple agent response maps by selecting the highest confidence response for each agent.
     *
     * For each agent key present in the input maps, returns the response with the highest confidence score.
     * If no response is found for an agent, a default response with zero confidence is used.
     *
     * @param agentResponseMapList A list of maps, each mapping agent names to their responses.
     * @return A map of agent names to their highest confidence response.
     */
    fun aggregateAgentResponses(agentResponseMapList: List<Map<String, AgentResponse>>): Map<String, AgentResponse> {
        val flatResponses = agentResponseMapList.flatMap { it.entries }
        return flatResponses.groupBy { it.key }
            .mapValues { entry ->

                // Determine "best" response, e.g., first successful, or combine content.
                // For now, let's pick the first successful response, or the first response if none are successful.
                // The original logic used confidence, which is no longer directly available.
                val best = entry.value.firstOrNull { it.value.isSuccess }?.value
                    ?: entry.value.firstOrNull()?.value
                    ?: AgentResponse("No response", isSuccess = false, error = "No responses to aggregate")
                Log.d(
                    "GenesisAgent",
                    "Consensus for ${entry.key}: ${best.content} (success=${best.isSuccess})"
                )
                best

            }
    }

    /**
     * Shares the provided context with all target agents that support context updates.
     *
     * Only agents implementing the `ContextAwareAgent` interface will receive the new context.
     *
     * @param newContext The context data to broadcast.
     * @param targetAgents The list of agents to receive the context update.
     */
    fun broadcastContext(newContext: Map<String, Any>, targetAgents: List<Agent>) {
        targetAgents.forEach { agent ->
            if (agent is ContextAwareAgent) {
                agent.setContext(newContext) // Assuming ContextAwareAgent has setContext
            }
        }
    }

    /**
     * Registers an agent instance under the specified name in the internal agent registry.
     *
     * If an agent with the same name already exists, it will be replaced.
     */
    fun registerAgent(name: String, agentInstance: Agent) {
        _agentRegistry[name] = agentInstance
        Log.d("GenesisAgent", "Registered agent: $name")
    }

    fun deregisterAgent(name: String) {
        _agentRegistry.remove(name)
        Log.d("GenesisAgent", "Deregistered agent: $name")
    }

    fun clearHistory() {
        _history.clear()
        Log.d("GenesisAgent", "Cleared conversation history")
    }

    /**
     * Adds an entry to the conversation or interaction history.
     *
     * @param entry The history entry to add.
     */
    fun addToHistory(entry: Map<String, Any>) {
        _history.add(entry)
        Log.d("GenesisAgent", "Added to history: $entry")
    }

    /**
     * Persists the current conversation history using the provided persistence function.
     *
     * @param persistAction A function that handles saving the list of history entries.
     */
    fun saveHistory(persistAction: (List<Map<String, Any>>) -> Unit) {
        persistAction(_history)
    }

    /**
     * Loads conversation history using the provided loader function and updates the internal history and context.
     *
     * The context is updated with the most recent entry from the loaded history, if available.
     *
     * @param loadAction A function that returns a list of history entries to load.
     */
    fun loadHistory(loadAction: () -> List<Map<String, Any>>) {
        val loadedHistory = loadAction()
        _history.clear()
        _history.addAll(loadedHistory)
        _context.update { it + (loadedHistory.lastOrNull() ?: emptyMap()) }
    }

    /**
     * Shares the current context with all registered agents that support context awareness.
     *
     * For each agent in the registry implementing `ContextAwareAgent`, updates its context to match the current shared context.
     */
    fun shareContextWithAgents() {
        agentRegistry.values.forEach { agent ->
            if (agent is ContextAwareAgent) {
                agent.setContext(_context.value)
            }
        }
    }

    /**
     * Dynamically registers an agent instance under the specified name.
     *
     * Adds the agent to the internal registry, making it available for participation in agent operations.
     *
     * @param name The unique identifier for the agent.
     * @param agentInstance The agent instance to register.
     */
    fun registerDynamicAgent(name: String, agentInstance: Agent) {
        _agentRegistry[name] = agentInstance
        Log.d("GenesisAgent", "Dynamically registered agent: $name")
    }

    /**
     * Removes a dynamically registered agent from the internal registry by name.
     *
     * @param name The unique identifier of the agent to deregister.
     */
    fun deregisterDynamicAgent(name: String) {
        _agentRegistry.remove(name)
        Log.d("GenesisAgent", "Dynamically deregistered agent: $name")
    }

    enum class ConversationMode { TURN_ORDER, FREE_FORM }
}
