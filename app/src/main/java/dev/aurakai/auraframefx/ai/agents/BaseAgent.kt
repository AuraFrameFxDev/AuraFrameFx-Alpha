package dev.aurakai.auraframefx.ai.agents

import dev.aurakai.auraframefx.model.AgentResponse

import dev.aurakai.auraframefx.api.model.AgentType // Corrected import
import dev.aurakai.auraframefx.model.AiRequest
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
 

/**
 * Base implementation of the [Agent] interface.
 * @param agentName The name of the agent.

 * @param agentType The string representation of the agent type, to be mapped to [AgentType].
 */
open class BaseAgent(
    private val _agentName: String,
    private val _agentType: String,

) : Agent {

    /**
     * Returns the name of the agent.
     *
     * @return The agent's name, or null if not set.
     */
    override fun getName(): String? {
        return _agentName
    }

    /**
     * Returns the agent's type as an `AgentType` (from api.model) by mapping the internal type string,
     * defaulting to `AgentType.AURA` if no match is found.
     *
     * If the internal type string does not correspond to any known `AgentType`,
     * the method returns `AgentType.AURA` as a fallback.
     *
     * @return The mapped `dev.aurakai.auraframefx.api.model.AgentType` for this agent.
     */
    override fun getType(): dev.aurakai.auraframefx.api.model.AgentType { // Corrected return type
        // Map string to the generated dev.aurakai.auraframefx.api.model.AgentType
        // Fallback to a default or throw an error if mapping fails
        return dev.aurakai.auraframefx.api.model.AgentType.values().firstOrNull { it.value.equals(_agentType, ignoreCase = true) }
            ?: dev.aurakai.auraframefx.api.model.AgentType.AURA // Defaulting to Aura, consider a more robust fallback or error
    }

    /**
     * Processes an AI request with the provided context and returns a default agent response.
     *
     * Subclasses should override this method to provide custom request handling logic.
     *
     * @param request The AI request to process.
     * @param context Additional context for the request.
     * @return A default `AgentResponse` containing a message referencing the request, context, and agent name, with fixed confidence.
     */

    override suspend fun processRequest(request: AiRequest, context: String): AgentResponse {
        // Default implementation for base agent, override in subclasses
        return AgentResponse(
            content = "BaseAgent response to '${request.query}' for agent $_agentName with context '$context'",
            confidence = 1.0f
        )
    }
    /**
     * Returns a flow emitting a default agent response for the given request.
     *
     * The response includes the request query and agent name with a fixed confidence score. Intended to be overridden by subclasses for custom streaming behavior.
     *
     * @return A flow containing a single default `AgentResponse`.
     */
    // Stray '=' was removed from here
    override fun processRequestFlow(request: AiRequest): Flow<AgentResponse> {
        // Basic implementation, can be overridden for more complex streaming logic
        return flow {
            // For simplicity, using a dummy context. Subclasses should provide meaningful context.
            emit(processRequest(request, "DefaultContext_BaseAgentFlow"))
        }
    }

    
}
