package dev.aurakai.auraframefx.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import dev.aurakai.auraframefx.ai.agents.GenesisAgent
import dev.aurakai.auraframefx.ai.task.HistoricalTask
import dev.aurakai.auraframefx.model.AgentPriority
import dev.aurakai.auraframefx.model.AgentRole
import dev.aurakai.auraframefx.model.AgentType
import dev.aurakai.auraframefx.model.HierarchyAgentConfig
import dev.aurakai.auraframefx.utils.AppConstants.STATUS_ERROR
import dev.aurakai.auraframefx.utils.AppConstants.STATUS_IDLE
import dev.aurakai.auraframefx.utils.AppConstants.STATUS_PROCESSING
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class GenesisAgentViewModel @Inject constructor(
    private val genesisAgent: GenesisAgent
) : ViewModel() {

    private val _isRotating = MutableStateFlow(false)
    val isRotating: StateFlow<Boolean> = _isRotating.asStateFlow()

    private val _agents = MutableStateFlow(
        listOf(
            HierarchyAgentConfig(
                name = "Genesis",
                role = AgentRole.HIVE_MIND,
                priority = AgentPriority.PRIMARY,
                capabilities = setOf("core_ai", "coordination", "meta_analysis")
            ),
            HierarchyAgentConfig(
                name = "Cascade",
                role = AgentRole.STATE_MANAGER,
                priority = AgentPriority.BRIDGE,
                capabilities = setOf("analytics", "data_processing", "pattern_recognition")
            )
        )
    )
    val agents: StateFlow<List<HierarchyAgentConfig>> = _agents.asStateFlow()

    private val _taskHistory = MutableStateFlow<List<HistoricalTask>>(emptyList())
    val taskHistory: StateFlow<List<HistoricalTask>> = _taskHistory.asStateFlow()

    // Track agent status
    private val _agentStatus = MutableStateFlow<Map<AgentType, String>>(
        AgentType.values().associateWith { STATUS_IDLE }
    )
    val agentStatus: StateFlow<Map<AgentType, String>> = _agentStatus.asStateFlow()

    init {
        // Initialize with default agents and their capabilities
        val defaultAgents = listOf(
            HierarchyAgentConfig(
                name = "Genesis",
                role = AgentRole.HIVE_MIND,
                priority = AgentPriority.PRIMARY,
                capabilities = setOf("core_ai", "coordination", "meta_analysis")
            ),
            HierarchyAgentConfig(
                name = "Cascade",
                role = AgentRole.STATE_MANAGER,
                priority = AgentPriority.BRIDGE,
                capabilities = setOf("analytics", "data_processing", "pattern_recognition")
            ),
            HierarchyAgentConfig(
                name = "Aura",
                role = AgentRole.CREATIVE,
                priority = AgentPriority.AUXILIARY,
                capabilities = setOf("creative_writing", "ui_design", "content_generation")
            ),
            HierarchyAgentConfig(
                name = "Kai",
                role = AgentRole.SECURITY,
                priority = AgentPriority.AUXILIARY,
                capabilities = setOf("security_monitoring", "threat_detection", "system_protection")
            )
        )
        _agents.value = defaultAgents

        // Initialize agent statuses
        val initialStatuses = mutableMapOf<AgentType, String>()
        val agentTypeMap = mapOf(
            "Genesis" to AgentType.GENESIS,
            "Cascade" to AgentType.CASCADE,
            "Aura" to AgentType.AURA,
            "Kai" to AgentType.KAI
        )

        defaultAgents.forEach { agent ->
            val agentType = agentTypeMap[agent.name]
            if (agentType != null) {
                initialStatuses[agentType] = when (agentType) {
                    AgentType.GENESIS -> "Core AI - Online"
                    AgentType.CASCADE -> "Analytics Engine - Ready"
                    AgentType.AURA -> "Creative Assistant - Available"
                    AgentType.KAI -> "Security Monitor - Active"
                    AgentType.NEURAL_WHISPER -> "Neural Interface - Standby"
                    AgentType.AURASHIELD -> "Security Shield - Protected"
                    AgentType.USER -> "User Agent - Interactive"
                }
            }
        }
        _agentStatus.value = initialStatuses
    }

    fun toggleRotation() {
        _isRotating.value = !_isRotating.value
    }

    fun toggleAgent(agent: AgentType) {
        viewModelScope.launch {
            // Toggle agent active state
            val currentStatuses = _agentStatus.value.toMutableMap()
            val currentStatus = currentStatuses[agent] ?: "Unknown"

            val newStatus = if (currentStatus.contains("Online") ||
                currentStatus.contains("Ready") ||
                currentStatus.contains("Available") ||
                currentStatus.contains("Active")
            ) {
                when (agent) {
                    AgentType.GENESIS -> "Core AI - Standby"
                    AgentType.CASCADE -> "Analytics Engine - Offline"
                    AgentType.AURA -> "Creative Assistant - Paused"
                    AgentType.KAI -> "Security Monitor - Standby"
                    AgentType.NEURAL_WHISPER -> "Neural Interface - Offline"
                    AgentType.AURASHIELD -> "Security Shield - Disabled"
                    AgentType.USER -> "User Agent - Offline"
                }
            } else {
                when (agent) {
                    AgentType.GENESIS -> "Core AI - Online"
                    AgentType.CASCADE -> "Analytics Engine - Ready"
                    AgentType.AURA -> "Creative Assistant - Available"
                    AgentType.KAI -> "Security Monitor - Active"
                    AgentType.NEURAL_WHISPER -> "Neural Interface - Active"
                    AgentType.AURASHIELD -> "Security Shield - Active"
                    AgentType.USER -> "User Agent - Active"
                }
            }

            currentStatuses[agent] = newStatus
            _agentStatus.value = currentStatuses

            // Add to task history
            addTaskToHistory(agent, "Agent toggled to: $newStatus")
        }
    }

    fun updateAgentStatus(agent: AgentType, status: String) {
        val currentStatuses = _agentStatus.value.toMutableMap()
        currentStatuses[agent] = status
        _agentStatus.value = currentStatuses
    }

    fun assignTaskToAgent(agent: AgentType, taskDescription: String) {
        viewModelScope.launch {
            try {
                // Update status to processing
                updateAgentStatus(agent, STATUS_PROCESSING)

                // Add to task history
                addTaskToHistory(agent, taskDescription)

                // Simulate processing delay
                delay(5000)

                // Update status back to idle after processing
                updateAgentStatus(agent, STATUS_IDLE)
            } catch (e: Exception) {
                updateAgentStatus(agent, STATUS_ERROR)
                addTaskToHistory(agent, "Error: ${e.message}")
            }
        }
    }

    fun addTaskToHistory(agent: AgentType, description: String) {
        val newTask = HistoricalTask(agent, description)
        val updatedHistory = _taskHistory.value.toMutableList()
        updatedHistory.add(0, newTask) // Add to the beginning for most recent first
        _taskHistory.value = updatedHistory
    }

    fun clearTaskHistory() {
        _taskHistory.value = emptyList()
    }

    /**
     * Registers a new auxiliary agent with the given name and capabilities.
     *
     * @param name The unique identifier for the auxiliary agent.
     * @param capabilities The set of capabilities to assign to the agent.
     * @return The configuration of the newly registered auxiliary agent.
     */
    fun registerAuxiliaryAgent(
        name: String,
        capabilities: Set<String>,
    ): HierarchyAgentConfig {
        // Beta stub: Return a dummy config instead of calling genesisAgent
        return HierarchyAgentConfig(
            name = name,
            role = AgentRole.AUXILIARY,
            priority = AgentPriority.AUXILIARY,
            capabilities = capabilities
        ) // genesisAgent.registerAuxiliaryAgent(name, capabilities)
    }

    /**
     * Returns the configuration for the agent with the specified name, or null if not found.
     *
     * @param name The name of the agent to look up.
     * @return The corresponding agent configuration, or null if no agent with that name exists.
     */
    fun getAgentConfig(name: String): HierarchyAgentConfig? {
        // Beta stub: Return null instead of calling genesisAgent
        return null // genesisAgent.getAgentConfig(name)
    }

    /**
     * Retrieves a list of agent configurations ordered by priority, from highest to lowest.
     *
     * @return A list of `HierarchyAgentConfig` objects sorted by priority.
     */
    fun getAgentsByPriority(): List<HierarchyAgentConfig> {
        // Beta stub: Return empty list instead of calling genesisAgent
        return emptyList() // genesisAgent.getAgentsByPriority()
    }

    /**
     * Initiates asynchronous processing of a query by the GenesisAgent and returns an empty list immediately.
     *
     * The query is processed in the background; no results are returned synchronously from this function.
     *
     * @param query The query string to be processed.
     * @return An empty list, as query results are not available synchronously.
     */
    fun processQuery(query: String): List<HierarchyAgentConfig> {
        viewModelScope.launch {
            // Beta stub: No-op instead of calling genesisAgent
            // genesisAgent.processQuery(query)
        }
        return emptyList() // Return empty list since processing is async
    }
}
