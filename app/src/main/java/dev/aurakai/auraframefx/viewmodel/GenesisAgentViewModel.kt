package dev.aurakai.auraframefx.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import dev.aurakai.auraframefx.ai.agents.GenesisAgent
import dev.aurakai.auraframefx.ai.task.HistoricalTask
import dev.aurakai.auraframefx.model.AgentConfig
import dev.aurakai.auraframefx.model.AgentType
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

@HiltViewModel
class GenesisAgentViewModel @Inject constructor(
    private val genesisAgent: GenesisAgent,
) : ViewModel() {

    private val _agents = MutableStateFlow<List<AgentConfig>>(emptyList()) // Initialize properly
    val agents: StateFlow<List<AgentConfig>> = _agents.asStateFlow()

    // Track agent status
    private val _agentStatus = MutableStateFlow<Map<AgentType, String>>(
        AgentType.values().associateWith { STATUS_IDLE }
    )
    val agentStatus: StateFlow<Map<AgentType, String>> = _agentStatus.asStateFlow()

    // Track task history
    private val _taskHistory = MutableStateFlow<List<HistoricalTask>>(emptyList())
    val taskHistory: StateFlow<List<HistoricalTask>> = _taskHistory.asStateFlow()

    // Track rotation state
    private val _isRotating = MutableStateFlow(true)
    val isRotating: StateFlow<Boolean> = _isRotating.asStateFlow()

    init { // Initialize agents in init block
        _agents.value = genesisAgent.getAgentsByPriority()
    }

    fun toggleRotation() {
        _isRotating.value = !_isRotating.value
    }

    fun toggleAgent(agent: AgentType) {
        viewModelScope.launch {
            genesisAgent.toggleAgent(agent)
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
     * Registers a new auxiliary agent with the specified name and capabilities.
     *
     * @param name The unique name for the auxiliary agent.
     * @param capabilities The set of capabilities assigned to the agent.
     * @return The configuration of the newly registered agent.
     */
    fun registerAuxiliaryAgent(
        name: String,
        capabilities: Set<String>,
    ): AgentConfig {
        return genesisAgent.registerAuxiliaryAgent(name, capabilities)
    }

    /**
     * Retrieves the configuration for an agent by its name.
     *
     * @param name The name of the agent to retrieve.
     * @return The agent's configuration if found, or null if no agent with the specified name exists.
     */
    fun getAgentConfig(name: String): AgentConfig? {
        return genesisAgent.getAgentConfig(name)
    }

    /**
     * Returns a list of agent configurations ordered by priority, from highest to lowest.
     *
     * @return The list of agent configurations sorted by priority.
     */
    fun getAgentsByPriority(): List<AgentConfig> {
        return genesisAgent.getAgentsByPriority()
    }

    /**
     * Triggers asynchronous processing of the given query by the GenesisAgent and returns immediately.
     *
     * The function launches background processing for the provided query string. No results are returned synchronously; the returned list is always empty.
     *
     * @param query The query string to process.
     * @return An empty list, as results are handled asynchronously.
     */
    fun processQuery(query: String): List<AgentConfig> {
        viewModelScope.launch {
            genesisAgent.processQuery(query)
        }
        return emptyList() // Return empty list since processing is async
    }
}
// Note: This ViewModel is designed to be used with Hilt for dependency injection.
// If you're not using Hilt, you can remove the @Inject annotation and manually instantiate it
// in your activity or fragment. The ViewModel should be scoped to the lifecycle of the activity
// or fragment that uses it, typically using ViewModelProvider.Factory or HiltViewModelFactory
// if you're using Hilt.
// Ensure you have the necessary dependencies for ViewModel and Hilt in your build.gradle file:
// implementation "androidx.lifecycle:lifecycle-viewmodel-ktx:2.3.1
// implementation "androidx.hilt:hilt-lifecycle-viewmodel:1.0.0"
// kapt "androidx.hilt:hilt-compiler:1.0.0"
// implementation "com.google.dagger:hilt-android:2.28-alpha"
// kapt "com.google.dagger:hilt-android-compiler:2.28-alpha"
// Also, ensure you have the necessary imports for ViewModel, StateFlow, and other components used in this ViewModel.
// If you're using Hilt, annotate this class with @HiltViewModel and use @Inject constructor for dependencies.
// If you're not using Hilt, you can remove the @Inject annotation and manually instantiate it
// in your activity or fragment. The ViewModel should be scoped to the lifecycle of the activity
// or fragment that uses it, typically using ViewModelProvider.Factory or ViewModelProvider.NewInstance
// if you're using ViewModelProvider directly.
// Ensure you have the necessary dependencies for ViewModel and StateFlow in your build.gradle file:
// implementation "androidx.lifecycle:lifecycle-viewmodel-ktx:2.3.1
// implementation "org.jetbrains.kotlinx:kotlinx-coroutines-core:1.4.2
// implementation "org.jetbrains.kotlinx:kotlinx-coroutines-android:1.4.
// Also, ensure you have the necessary imports for ViewModel, StateFlow, and other components
// used in this ViewModel.
// If you're using Hilt, annotate this class with @HiltViewModel and use @Inject constructor for dependencies.
// If you're not using Hilt, you can remove the @Inject annotation and manually instantiate it
// in your activity or fragment. The ViewModel should be scoped to the lifecycle of the activity
// or fragment that uses it, typically using ViewModelProvider.Factory or ViewModelProvider.NewInstance
// if you're using ViewModelProvider directly.
// Ensure you have the necessary dependencies for ViewModel and StateFlow in your build.gradle file:
// implementation "androidx.lifecycle:lifecycle-viewmodel-ktx:2.3.1"
// implementation "org.jetbrains.kotlinx:kotlinx-coroutines-core:1.4.2"
// implementation "org.jetbrains.kotlinx:kotlinx-coroutines-android:1.4.2"
