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
     * Toggles the internal rotation state between active and inactive.
     *
     * Flips the rotation flag, which can be used to control agent rotation behavior within the application.
     */
    fun toggleRotation() {
        _isRotating.value = !_isRotating.value
    }

    /**
     * Toggles the operational status of the specified agent between active and inactive states.
     *
     * Updates the agent's status in the state flow and records the status change in the task history.
     *
     * @param agent The agent whose status will be toggled.
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

    /**
     * Sets a new status string for the specified agent.
     *
     * Updates the agent's status in the internal state flow, triggering observers.
     *
     * @param agent The agent whose status will be updated.
     * @param status The status string to assign to the agent.
     */
    fun updateAgentStatus(agent: AgentType, status: String) {
        val currentStatuses = _agentStatus.value.toMutableMap()
        currentStatuses[agent] = status
        _agentStatus.value = currentStatuses
    }

    /**
     * Asynchronously assigns a task to the specified agent, updating the agent's status and recording the task in history.
     *
     * The agent's status is set to processing during task execution and reset to idle upon completion. If an error occurs, the status is set to error and the error is recorded in the task history.
     *
     * @param agent The agent to which the task is assigned.
     * @param taskDescription A description of the task being assigned.
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
     * Records a completed task for the specified agent in the task history.
     *
     * @param agent The agent associated with the completed task.
     * @param description Description of the completed task.
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
     * Removes all tasks from the task history, resetting it to an empty list.
     */
    fun clearTaskHistory() {
        _taskHistory.value = emptyList()
    }

    /**
     * Returns the current status string for the specified agent.
     *
     * If the agent's status is not set, the default idle status is returned.
     *
     * @param agent The agent whose status is being retrieved.
     * @return The current status string for the agent, or the idle status if unset.
     */
    fun getAgentStatus(agent: AgentType): String {
        return _agentStatus.value[agent] ?: STATUS_IDLE
    }

    /**
     * Returns the configuration for the agent with the specified name, ignoring case.
     *
     * @param name The name of the agent to search for.
     * @return The corresponding agent configuration, or null if not found.
     */
    fun getAgentByName(name: String): HierarchyAgentConfig? {
        return _agents.value.find { it.name.equals(name, ignoreCase = true) }
    }

    /**
     * Resets the status of all agents to the idle state.
     *
     * Updates every agent's status to indicate they are not currently processing any tasks.
     */
    fun clearAllAgentStatuses() {
        val currentStatuses = _agentStatus.value.toMutableMap()
        currentStatuses.keys.forEach { agent ->
            currentStatuses[agent] = STATUS_IDLE
        }
        _agentStatus.value = currentStatuses
    }

    /**
     * Returns a list of agents whose capabilities include the specified value, ignoring case.
     *
     * @param capability The capability to search for.
     * @return A list of agent configurations that possess the specified capability.
     */
    fun getAgentsByCapability(capability: String): List<HierarchyAgentConfig> {
        return _agents.value.filter { agent ->
            agent.capabilities.any { it.equals(capability, ignoreCase = true) }
        }
    }

    /**
     * Returns a list of agent configurations matching the specified role.
     *
     * @param role The role to filter agents by.
     * @return A list of agents with the given role.
     */
    fun getAgentsByRole(role: AgentRole): List<HierarchyAgentConfig> {
        return _agents.value.filter { it.role == role }
    }

    /**
     * Returns a list of agent configurations that match the specified priority.
     *
     * @param priority The priority level to filter agents by.
     * @return A list of agents with the given priority.
     */
    fun getAgentsByPriority(priority: AgentPriority): List<HierarchyAgentConfig> {
        return _agents.value.filter { it.priority == priority }
    }

    /**
     * Initiates asynchronous sequential assignment of multiple tasks to the specified agent.
     *
     * Each task in the provided list is assigned to the agent with a delay between assignments. The function returns immediately with an empty list, as task processing occurs asynchronously and results are not available synchronously.
     *
     * @param agent The agent to which tasks will be assigned.
     * @param tasks The list of task descriptions to assign.
     * @return An empty list, as task processing is asynchronous.
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
     * Retrieves the configuration of an agent by name, ignoring case.
     *
     * @param name The name of the agent to search for.
     * @return The agent's configuration if found, or null if no agent matches the name.
     */
    fun getAgentConfig(name: String): HierarchyAgentConfig? {
        return _agents.value.find { it.name.equals(name, ignoreCase = true) }
    }

    /**
     * Returns all agent configurations sorted by ascending priority.
     *
     * @return A list of agent configurations ordered from lowest to highest priority.
     */
    fun getAgentsByPriority(): List<HierarchyAgentConfig> {
        return _agents.value.sortedBy { it.priority }
    }

    /**
     * Starts asynchronous processing of a query and immediately returns an empty list.
     *
     * The query is handled in the background; no results are provided synchronously.
     *
     * @param query The query string to process.
     * @return An empty list, as query results are not returned by this method.
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
