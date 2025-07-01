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

    fun generateFinalResponse(agentMessages: List<AgentMessage>): String {
        // Simple concatenation for now, could be more sophisticated
        return "[Genesis Synthesis] ${agentMessages.filter { it.sender != dev.aurakai.auraframefx.model.AgentType.GENESIS }.joinToString(" | ") { "${it.sender}: ${it.content}" }}"
    }

    fun calculateConfidence(agentMessages: List<AgentMessage>): Float {
        if (agentMessages.isEmpty()) return 0.0f
        return agentMessages.map { it.confidence }.average().toFloat().coerceIn(0.0f, 1.0f)
    }

    fun toggleAgent(agentType: dev.aurakai.auraframefx.model.AgentType) {
        _activeAgents.update { current ->
            if (current.contains(agentType)) current - agentType else current + agentType
        }
    }

    fun registerAuxiliaryAgent(name: String, capabilities: Set<String>): AgentConfig {
        return AgentHierarchy.registerAuxiliaryAgent(name, capabilities)
    }

    fun getAgentConfig(name: String): AgentConfig? = AgentHierarchy.getAgentConfig(name)

    fun getAgentsByPriority(): List<AgentConfig> = AgentHierarchy.getAgentsByPriority()

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

    fun broadcastContext(newContext: Map<String, Any>, targetAgents: List<Agent>) {
        targetAgents.forEach { agent ->
            if (agent is ContextAwareAgent) {
                agent.setContext(newContext) // Assuming ContextAwareAgent has setContext
            }
        }
    }

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

    fun addToHistory(entry: Map<String, Any>) {
        _history.add(entry)
        Log.d("GenesisAgent", "Added to history: $entry")
    }

    fun saveHistory(persistAction: (List<Map<String, Any>>) -> Unit) {
        persistAction(_history)
    }

    fun loadHistory(loadAction: () -> List<Map<String, Any>>) {
        val loadedHistory = loadAction()
        _history.clear()
        _history.addAll(loadedHistory)
        _context.update { it + (loadedHistory.lastOrNull() ?: emptyMap()) }
    }

    fun shareContextWithAgents() {
        agentRegistry.values.forEach { agent ->
            if (agent is ContextAwareAgent) {
                agent.setContext(_context.value)
            }
        }
    }

    fun registerDynamicAgent(name: String, agentInstance: Agent) {
        _agentRegistry[name] = agentInstance
        Log.d("GenesisAgent", "Dynamically registered agent: $name")
    }

    fun deregisterDynamicAgent(name: String) {
        _agentRegistry.remove(name)
        Log.d("GenesisAgent", "Dynamically deregistered agent: $name")
    }

    enum class ConversationMode { TURN_ORDER, FREE_FORM }
}
