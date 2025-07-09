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
     *
     * Switches the internal rotation flag, which may be used to control agent rotation behavior in the application.
     */
    fun toggleRotation() {
        _isRotating.value = !_isRotating.value
    }

    /**
     * Toggles the specified agent's operational status between active and inactive states.
     *
     * Updates the agent's status in the state flow and records the status change in the task history.
     *
     * @param agent The agent whose operational status will be toggled.
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
     * Updates the status string of the specified agent.
     *
     * @param agent The agent whose status is to be updated.
     * @param status The new status string to assign to the agent.
     */
    fun updateAgentStatus(agent: AgentType, status: String) {
        val currentStatuses = _agentStatus.value.toMutableMap()
        currentStatuses[agent] = status
        _agentStatus.value = currentStatuses
    }

    /**
     * Assigns a task to the specified agent asynchronously, updating the agent's status and recording the task in history.
     *
     * The agent's status is set to processing during task execution and reset to idle upon completion. If an error occurs, the status is set to error and the error is recorded in the task history.
     *
     * @param agent The agent to assign the task to.
     * @param taskDescription The description of the task being assigned.
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
     * Adds a completed task entry for the specified agent to the task history.
     *
     * @param agent The agent for which the task was completed.
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
     * Clears all entries from the task history.
     */
    fun clearTaskHistory() {
        _taskHistory.value = emptyList()
    }

    /**
     * Retrieves the current status string for the specified agent.
     *
     * Returns the agent's status if set; otherwise, returns the default idle status.
     *
     * @param agent The agent whose status is being queried.
     * @return The current status string of the agent, or the idle status if not set.
     */
    fun getAgentStatus(agent: AgentType): String {
        return _agentStatus.value[agent] ?: STATUS_IDLE
    }

    /**
     * Retrieves the configuration of an agent by its name, ignoring case.
     *
     * @param name The name of the agent to look up.
     * @return The agent's configuration if found, or null if no match exists.
     */
    fun getAgentByName(name: String): HierarchyAgentConfig? {
        return _agents.value.find { it.name.equals(name, ignoreCase = true) }
    }

    /**
     * Sets all agent statuses to the idle state.
     *
     * This method updates the status of every agent to indicate they are idle.
     */
    fun clearAllAgentStatuses() {
        val currentStatuses = _agentStatus.value.toMutableMap()
        currentStatuses.keys.forEach { agent ->
            currentStatuses[agent] = STATUS_IDLE
        }
        _agentStatus.value = currentStatuses
    }

    /**
     * Retrieves all agents that have the specified capability.
     *
     * @param capability The capability to match, case-insensitive.
     * @return A list of agent configurations containing the given capability.
     */
    fun getAgentsByCapability(capability: String): List<HierarchyAgentConfig> {
        return _agents.value.filter { agent ->
            agent.capabilities.any { it.equals(capability, ignoreCase = true) }
        }
    }

    /**
     * Retrieves all agents with the specified role.
     *
     * @param role The agent role to filter by.
     * @return A list of agent configurations that match the given role.
     */
    fun getAgentsByRole(role: AgentRole): List<HierarchyAgentConfig> {
        return _agents.value.filter { it.role == role }
    }

    /**
     * Retrieves a list of agent configurations with the specified priority.
     *
     * @param priority The priority level to filter agents.
     * @return A list of agents whose priority matches the given value.
     */
    fun getAgentsByPriority(priority: AgentPriority): List<HierarchyAgentConfig> {
        return _agents.value.filter { it.priority == priority }
    }

    /**
     * Starts asynchronous sequential assignment of multiple tasks to the specified agent.
     *
     * Each task is assigned to the agent with a delay between assignments. The function returns immediately with an empty list, as task processing is handled asynchronously.
     *
     * @param agent The agent to receive the tasks.
     * @param tasks The list of task descriptions to assign.
     * @return An empty list, since task results are not available synchronously.
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
     * Returns the configuration for the agent with the specified name, or null if not found.
     *
     * @param name The agent's name to search for (case-insensitive).
     * @return The matching agent configuration, or null if no agent matches the name.
     */
    fun getAgentConfig(name: String): HierarchyAgentConfig? {
        return _agents.value.find { it.name.equals(name, ignoreCase = true) }
    }

    /**
     * Retrieves all agent configurations sorted by ascending priority.
     *
     * @return A list of agent configurations ordered from lowest to highest priority.
     */
    fun getAgentsByPriority(): List<HierarchyAgentConfig> {
        return _agents.value.sortedBy { it.priority }
    }

    /**
     * Initiates asynchronous processing of a query and returns immediately with an empty list.
     *
     * The query is processed in the background; no results are returned synchronously.
     *
     * @param query The query string to process.
     * @return An empty list, as results are handled asynchronously.
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
