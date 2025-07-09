package dev.aurakai.auraframefx.ai.agents

import dev.aurakai.auraframefx.model.agent_states.ProcessingState
import dev.aurakai.auraframefx.model.agent_states.VisionState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import javax.inject.Inject
import javax.inject.Singleton

/**
 * CascadeAgent is a stateful, collaborative agent in AuraFrameFX.
 *
 * Cascade acts as a bridge and orchestrator between Aura (creativity/UI) and Kai (security/automation).
 * Responsibilities:
 *  - Vision management and stateful processing
 *  - Multi-agent collaboration and context sharing
 *  - Synchronizing and coordinating actions between Aura and Kai
 *  - Advanced context chaining and persistent memory
 *
 * Contributors: Please keep Cascade's logic focused on agent collaboration, state management, and bridging creative and security domains.
 */
@Singleton
class CascadeAgent @Inject constructor(
    private val auraAgent: AuraAgent, // Now using the actual AuraAgent type
    private val kaiAgent: KaiAgent,   // Now using the actual KaiAgent type
) {
    private val _visionState = MutableStateFlow(VisionState())
    val visionState: StateFlow<VisionState> = _visionState.asStateFlow()

    private val _processingState = MutableStateFlow(ProcessingState())
    val processingState: StateFlow<ProcessingState> = _processingState.asStateFlow()

    /**
     * Interface for agents that need to be aware of vision state changes
     */
    interface VisionStateAware {
        fun onVisionUpdate(newState: VisionState)
    }

    /**
     * Interface for agents that need to be aware of processing state changes
     */
    interface ProcessingStateAware {
        fun onProcessingStateChange(newState: ProcessingState)
    }

    // Add stubs for agent collaboration methods expected by CascadeAgent
    // These should be implemented in AuraAgent and KaiAgent as well
    open fun onVisionUpdate(newState: VisionState) {
        // Default no-op. Override in AuraAgent/KaiAgent for custom behavior.
    }

    open fun onProcessingStateChange(newState: ProcessingState) {
        // Default no-op. Override in AuraAgent/KaiAgent for custom behavior.
    }

    fun shouldHandleSecurity(prompt: String): Boolean = false
    fun shouldHandleCreative(prompt: String): Boolean = false
    fun processRequest(prompt: String): String = ""

    /**
     * Updates the vision state with a new observation.
     * @param observation The new observation to record
     * @param objectsDetected Optional list of detected objects
     */
    fun updateVisionState(observation: String, objectsDetected: List<String> = emptyList()) {
        _visionState.update { current ->
            current.withNewObservation(observation).copy(objectsDetected = objectsDetected)
        }
        // Notify other agents of the vision update
        notifyAgentsOfVisionUpdate()
    }

    /**
     * Internal method to notify all agents of vision state changes
     */
    private fun notifyAgentsOfVisionUpdate() {
        val currentState = _visionState.value
        CoroutineScope(Dispatchers.Default).launch {
            // Notify Aura if it has the handler
            (auraAgent as? VisionStateAware)?.onVisionUpdate(currentState)
            // Notify Kai if it has the handler
            (kaiAgent as? VisionStateAware)?.onVisionUpdate(currentState)
        }
    }

    /**
     * Updates the processing state with a new step.
     * @param step The current processing step
     * @param progress The progress percentage (0.0 to 1.0)
     * @param isError Whether an error occurred
     */
    fun updateProcessingState(step: String, progress: Float = 0f, isError: Boolean = false) {
        _processingState.update { current ->
            current.withNewStep(step, progress, isError)
        }
        // Notify other agents of the processing state change
        notifyAgentsOfProcessingStateChange()
    }

    /**
     * Internal method to notify all agents of processing state changes
     */
    private fun notifyAgentsOfProcessingStateChange() {
        val currentState = _processingState.value
        CoroutineScope(Dispatchers.Default).launch {
            // Notify Aura if it has the handler
            (auraAgent as? ProcessingStateAware)?.onProcessingStateChange(currentState)
            // Notify Kai if it has the handler
            (kaiAgent as? ProcessingStateAware)?.onProcessingStateChange(currentState)
        }
    }

    // TODO: Implement processRequest, getCapabilities, getContinuousMemory as needed, based on available agent and state APIs.
}
