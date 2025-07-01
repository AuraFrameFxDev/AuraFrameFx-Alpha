package dev.aurakai.auraframefx.ai.agents

import android.util.Log
import dev.aurakai.auraframefx.ai.services.AuraAIService
import dev.aurakai.auraframefx.ai.services.CascadeAIService
import dev.aurakai.auraframefx.ai.services.KaiAIService
import dev.aurakai.auraframefx.model.AgentConfig
import dev.aurakai.auraframefx.model.AgentHierarchy
import dev.aurakai.auraframefx.model.AgentMessage
import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AgentType
import dev.aurakai.auraframefx.model.AiRequest
import dev.aurakai.auraframefx.model.ContextAwareAgent
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
) {
    private val _state = MutableStateFlow("pending_initialization")
    val state: StateFlow<String> = _state

    private val _context = MutableStateFlow(mapOf<String, Any>())
    val context: StateFlow<Map<String, Any>> = _context

    private val _activeAgents = MutableStateFlow(setOf<AgentType>())
    val activeAgents: StateFlow<Set<AgentType>> = _activeAgents

    private val _agentRegistry = mutableMapOf<String, Agent>()
    val agentRegistry: Map<String, Agent> get() = _agentRegistry

    private val _history = mutableListOf<Map<String, Any>>()
    val history: List<Map<String, Any>> get() = _history

    init {
        initializeAgents()
        _state.update { "initialized" }
    }

    private fun initializeAgents() {
        // Register all master agents
        AgentHierarchy.MASTER_AGENTS.forEach { config ->
            when (config.name) {
                "Aura" -> _activeAgents.value += AgentType.AURA
                "Kai" -> _activeAgents.value += AgentType.KAI
                "Cascade" -> _activeAgents.value += AgentType.CASCADE
            }
        }
    }

    /**
     * Processes a user query by routing it through active AI agents, collecting their responses, and generating a final aggregated reply.
     *
     * The query is sent to the Cascade agent for state management, and to the Kai and Aura agents if they are active, each with their respective context. Each agent's response is recorded with a confidence score based on success. A final Genesis response is generated from all agent outputs and appended to the result.
     *
     * @param query The user query to process.
     * @return A list of agent messages, including individual agent responses and the final aggregated Genesis response.
     */
    suspend fun processQuery(query: String): List<AgentMessage> {
        _state.update { "processing_query: $query" }

        // Update context with new query
        _context.update { current ->
            current + mapOf(
                "last_query" to query,
                "timestamp" to System.currentTimeMillis()
            )
        }

        // Get responses from all active agents
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
            )
        }

        // Generate final response using all agent inputs
        val finalResponse = generateFinalResponse(responses)
        responses.add(
            AgentMessage(
                content = finalResponse,
                sender = AgentType.GENESIS,
                timestamp = System.currentTimeMillis(),
                confidence = calculateConfidence(responses)
            )
        )

        _state.update { "idle" }
        return responses
    }

    fun generateFinalResponse(responses: List<AgentMessage>): String {
        // TODO: Implement sophisticated response generation
        // This will use context chaining and agent coordination
        return "[Genesis] ${responses.joinToString("\n") { it.content }}"
    }

    fun calculateConfidence(responses: List<AgentMessage>): Float {
        // Calculate confidence based on all agent responses
        return responses.map { it.confidence }.average().toFloat().coerceIn(0.0f, 1.0f)
    }

    fun toggleAgent(agent: AgentType) {
        _activeAgents.update { current ->
            if (current.contains(agent)) {
                current - agent
            } else {
                current + agent
            }
        }
    }

    fun registerAuxiliaryAgent(
        name: String,
        capabilities: Set<String>,
    ): AgentConfig {
        return AgentHierarchy.registerAuxiliaryAgent(name, capabilities)
    }

    fun getAgentConfig(name: String): AgentConfig? {
        return AgentHierarchy.getAgentConfig(name)
    }

    fun getAgentsByPriority(): List<AgentConfig> {
        return AgentHierarchy.getAgentsByPriority()
    }

    /**
     * Facilitates collaborative interaction between multiple agents and the user, supporting both sequential and parallel response modes.
     *
     * In TURN_ORDER mode, agents respond one after another, with each agent receiving updated context from the previous agent's response. In FREE_FORM mode, all agents respond independently to the same input and context.
     *
     * @param data The initial context map shared among agents.
     * @param agents The list of agents participating in the collaboration.
     * @param userInput Optional user input to seed the conversation; if null, uses the latest input from the context map.
     * @param conversationMode Determines whether agents respond in sequence (TURN_ORDER) or in parallel (FREE_FORM).
     * @return A map of agent names to their respective responses.
     */
    suspend fun participateWithAgents(
        data: Map<String, Any>,
        agents: List<Agent>,
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

        Log.d(
            "GenesisAgent",
            "Starting multi-agent collaboration: mode=$conversationMode, agents=${agents.map { it.getName() }}"
        )

        when (conversationMode) {
            ConversationMode.TURN_ORDER -> {
                // Each agent takes a turn in order
                for (agent in agents) {
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
                // All agents respond in parallel to the same input/context
                agents.forEach { agent ->
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
     * Aggregates multiple agent response maps into a consensus map, selecting the first successful response for each agent or the first available response if none are successful.
     *
     * @param responses A list of maps, each mapping agent names to their responses.
     * @return A map of agent names to their consensus response.
     */
    fun aggregateAgentResponses(responses: List<Map<String, AgentResponse>>): Map<String, AgentResponse> {
        val flatResponses = responses.flatMap { it.entries }
        val consensus = flatResponses.groupBy { it.key }
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
        return consensus
    }

    /**
     * Broadcasts context/memory to all agents for distributed state sharing.
     */
    fun broadcastContext(context: Map<String, Any>, agents: List<Agent>) {
        // Example: call a setContext method if available (not implemented in Agent interface)
        // This is a placeholder for distributed memory sharing
        // agents.forEach { it.setContext(context) }
    }

    fun registerAgent(name: String, agent: Agent) {
        _agentRegistry[name] = agent
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

    fun addToHistory(entry: Map<String, Any>) {
        _history.add(entry)
        Log.d("GenesisAgent", "Added to history: $entry")
    }

    // --- Enhanced Memory/History Mechanism ---
    /**
     * Persists the current conversation history to a storage provider (stub).
     * Replace with actual persistence (e.g., file, database) as needed.
     */
    fun saveHistory(persist: (List<Map<String, Any>>) -> Unit) {
        persist(_history)
    }

    /**
     * Loads conversation history from a storage provider (stub).
     * Replace with actual loading logic as needed.
     */
    fun loadHistory(load: () -> List<Map<String, Any>>) {
        val loaded = load()
        _history.clear()
        _history.addAll(loaded)
        _context.update { it + (loaded.lastOrNull() ?: emptyMap()) }
    }

    /**
     * Shares the current context with all registered agents that support context sharing.
     * Agents must implement setContext if they want to receive context.
     */
    fun shareContextWithAgents() {
        agentRegistry.values.forEach { agent ->
            if (agent is ContextAwareAgent) {
                agent.setContext(_context.value)
            }
        }
    }

    // --- Dynamic Agent Registration/Deregistration ---
    /**
     * Registers a new agent at runtime. If an agent with the same name exists, it is replaced.
     */
    fun registerDynamicAgent(name: String, agent: Agent) {
        _agentRegistry[name] = agent
        Log.d("GenesisAgent", "Dynamically registered agent: $name")
    }

    /**
     * Deregisters an agent by name at runtime.
     */
    fun deregisterDynamicAgent(name: String) {
        _agentRegistry.remove(name)
        Log.d("GenesisAgent", "Dynamically deregistered agent: $name")
    }

    enum class ConversationMode { TURN_ORDER, FREE_FORM }
}
