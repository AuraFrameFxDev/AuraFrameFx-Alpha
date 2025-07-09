package dev.aurakai.auraframefx.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
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

// import javax.inject.Singleton // ViewModels should use @HiltViewModel

class GenesisAgentViewModel @Inject constructor(
    // private val genesisAgent: GenesisAgent
) : ViewModel() {

    // Beta stub: No actual GenesisAgent dependency
    // private val genesisAgent: GenesisAgent? = null

    private val _agents =
        MutableStateFlow<List<HierarchyAgentConfig>>(emptyList()) // Initialize properly
    val agents: StateFlow<List<HierarchyAgentConfig>> = _agents.asStateFlow()

    // Track agent status
    private val _agentStatus = MutableStateFlow<Map<AgentType, String>>(
        AgentType.entries.associateWith { STATUS_IDLE }
    )
    val agentStatus: StateFlow<Map<AgentType, String>> = _agentStatus.asStateFlow()

    // Track task history
    private val _taskHistory = MutableStateFlow<List<HistoricalTask>>(emptyList())
    val taskHistory: StateFlow<List<HistoricalTask>> = _taskHistory.asStateFlow()

    // Track rotation state
    private val _isRotating = MutableStateFlow(true)
    val isRotating: StateFlow<Boolean> = _isRotating.asStateFlow()

    init {
        // Initialize with default agents and their capabilities
        val defaultAgents = listOf(
            HierarchyAgentConfig(
                name = "Genesis",
                role = AgentRole.HIVE_MIND,
                priority = AgentPriority.MASTER,
                capabilities = setOf("core_ai", "coordination", "meta_analysis")
            ),
            HierarchyAgentConfig(
                name = "Cascade",
                role = AgentRole.ANALYTICS,
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

    /**
     * Toggles the rotation state between active and inactive.
     */
    fun toggleRotation() {
        _isRotating.value = !_isRotating.value
    }

    /**
     * Toggles the operational status of the specified agent between active and inactive states.
     *
     * Updates the agent's status in the state flow and records the change in the task history.
     *
     * @param agent The agent whose status should be toggled.
     */
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

    /**
     * Asynchronously assigns a task to the specified agent, updating its status and recording the task in history.
     *
     * The agent's status is set to processing during task execution, then reset to idle upon completion.
     * If an error occurs, the agent's status is set to error and the error is logged in the task history.
     *
     * @param agent The agent to which the task is assigned.
     * @param taskDescription A description of the task to assign.
     */
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

    /**
     * Appends a completed task entry for the specified agent to the task history.
     *
     * @param agent The agent associated with the task.
     * @param description A description of the completed task.
     */
    private fun addTaskToHistory(agent: AgentType, description: String) {
        val newTask = HistoricalTask(
            id = System.currentTimeMillis(),
            agentType = agent,
            description = description,
            timestamp = System.currentTimeMillis(),
            status = "Completed"
        )
        _taskHistory.value = _taskHistory.value + newTask
    }

    /**
     * Removes all entries from the task history.
     */
    fun clearTaskHistory() {
        _taskHistory.value = emptyList()
    }

    /**
     * Returns the current status string of the specified agent.
     *
     * If the agent's status is not found, returns the default idle status.
     *
     * @param agent The agent whose status is to be retrieved.
     * @return The current status string of the agent, or idle status if not set.
     */
    fun getAgentStatus(agent: AgentType): String {
        return _agentStatus.value[agent] ?: STATUS_IDLE
    }

    /**
     * Returns the agent configuration matching the given name, or null if not found.
     *
     * @param name The name of the agent to search for (case-insensitive).
     * @return The corresponding HierarchyAgentConfig if found, otherwise null.
     */
    fun getAgentByName(name: String): HierarchyAgentConfig? {
        return _agents.value.find { it.name.equals(name, ignoreCase = true) }
    }

    /**
     * Resets the status of all agents to the idle state.
     */
    fun clearAllAgentStatuses() {
        val currentStatuses = _agentStatus.value.toMutableMap()
        currentStatuses.keys.forEach { agent ->
            currentStatuses[agent] = STATUS_IDLE
        }
        _agentStatus.value = currentStatuses
    }

    /**
     * Returns a list of agents that possess the specified capability.
     *
     * @param capability The capability to search for (case-insensitive).
     * @return A list of agent configurations whose capabilities include the specified value.
     */
    fun getAgentsByCapability(capability: String): List<HierarchyAgentConfig> {
        return _agents.value.filter { agent ->
            agent.capabilities.any { it.equals(capability, ignoreCase = true) }
        }
    }

    /**
     * Returns a list of agents that have the specified role.
     *
     * @param role The role to filter agents by.
     * @return A list of agent configurations matching the given role.
     */
    fun getAgentsByRole(role: AgentRole): List<HierarchyAgentConfig> {
        return _agents.value.filter { it.role == role }
    }

    /**
     * Returns a list of agents that match the specified priority.
     *
     * @param priority The priority level to filter agents by.
     * @return A list of agent configurations with the given priority.
     */
    fun getAgentsByPriority(priority: AgentPriority): List<HierarchyAgentConfig> {
        return _agents.value.filter { it.priority == priority }
    }

    /**
     * Initiates asynchronous processing of a batch of tasks for the specified agent.
     *
     * Each task is assigned to the agent sequentially with a delay between assignments. Returns immediately with an empty list, as task processing occurs asynchronously.
     *
     * @param agent The agent to which tasks will be assigned.
     * @param tasks The list of task descriptions to process.
     * @return An empty list, as results are not available synchronously.
     */
    fun processBatchTasks(agent: AgentType, tasks: List<String>): List<Boolean> {
        viewModelScope.launch {
            tasks.forEach { task ->
                assignTaskToAgent(agent, task)
                delay(1000) // Delay between tasks
            }
        }
        return emptyList() // Return empty list since processing is async
    }

    /**
     * Retrieves the configuration of an agent by its name.
     *
     * @param name The name of the agent to search for (case-insensitive).
     * @return The agent's configuration if found, or null otherwise.
     */
    fun getAgentConfig(name: String): HierarchyAgentConfig? {
        return _agents.value.find { it.name.equals(name, ignoreCase = true) }
    }

    /**
     * Returns a list of agent configurations sorted by ascending priority.
     *
     * @return A list of `HierarchyAgentConfig` objects ordered from lowest to highest priority value.
     */
    fun getAgentsByPriority(): List<HierarchyAgentConfig> {
        return _agents.value.sortedBy { it.priority }
    }

    /**
     * Starts asynchronous processing of a query and returns immediately with an empty list.
     *
     * The query is handled in the background; this function does not provide synchronous results.
     *
     * @param query The query string to process.
     * @return An empty list, as results are not returned synchronously.
     */
    fun processQuery(query: String): List<HierarchyAgentConfig> {
        viewModelScope.launch {
            // Simulate processing delay
            delay(5000)
        }
        return emptyList() // Return empty list since processing is async
    }
}

/**
 * ViewModel for managing the Genesis Agent state and operations.
 * This ViewModel is designed to work with Hilt for dependency injection.
 * 
 * Key Features:
 * - Manages agent status and state
 * - Handles task assignment and history
 * - Provides agent configuration and capabilities
 * - Supports agent toggling and status updates
 */
