package dev.aurakai.auraframefx.model.agent_states

// TODO: Define actual properties for these states.
// TODO: Classes reported as unused or need implementation. Ensure these are utilized by CascadeAgent.

data class VisionState(
    val lastObservation: String? = null,
    val objectsDetected: List<String> = emptyList(),
    val history: List<String> = emptyList(),
    // Add other relevant vision state properties
) {
    /**
     * Creates a new VisionState with an updated observation and adds the previous state to history
     */
    fun withNewObservation(observation: String): VisionState {
        val newHistory = history.takeLast(9) + (lastObservation ?: "Initial state")
        return copy(
            lastObservation = observation,
            history = newHistory
        )
    }
}

data class ProcessingState(
    val currentStep: String? = null,
    val progressPercentage: Float = 0.0f,
    val isError: Boolean = false,
    val history: List<String> = emptyList(),
    // Add other relevant processing state properties
) {
    /**
     * Creates a new ProcessingState with an updated step and adds the previous state to history
     */
    fun withNewStep(step: String, progress: Float = 0f, error: Boolean = false): ProcessingState {
        val newHistory = history.takeLast(9) + (currentStep ?: "Initial state")
        return copy(
            currentStep = step,
            progressPercentage = progress,
            isError = error,
            history = newHistory
        )
    }
}
