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
     * Retrieves the agent's name.
     *
     * @return The name of the agent, or null if not set.

     */
    override fun getName(): String? {
        return _agentName
    }

    /**
     * Returns the agent's type as an `AgentType` enum value.
     *
     * @return The `AgentType` corresponding to the agent's internal type string.
     * @throws IllegalArgumentException If the internal type string does not match any valid `AgentType`.
     */
    override fun getType(): AgentType { // Return non-nullable AgentType from api.model
        return try {
            AgentType.valueOf(_agentType.uppercase())
        } catch (e: IllegalArgumentException) {
            // Or handle error more gracefully, e.g., map to a default or throw
            throw IllegalArgumentException("Invalid agent type string: $_agentType", e)
        }

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

    /**
     * Processes an AI request and returns a default successful response.
     *
     * Generates a generic agent response containing the request prompt, agent name, and provided context.
     * Intended to be overridden by subclasses for custom request handling.
     *
     * @param request The AI request to process.
     * @param context Additional context information for the request.
     * @return An [AgentResponse] with a default message and success status.
     */
    override suspend fun processRequest(request: AiRequest, context: String): AgentResponse {
        // Default implementation for base agent, override in subclasses
        return AgentResponse(
            content = "BaseAgent response to '${request.query}' for agent $_agentName with context '$context'",
            confidence = 1.0f
        )
    }

    /**
     * Returns a flow emitting a single agent response for the given AI request.
     *
     * This default implementation calls `processRequest` with a placeholder context. Subclasses may override to provide streaming or multi-part responses.
     *
     * @return A flow emitting one `AgentResponse` for the provided request.
     */

    override fun processRequestFlow(request: AiRequest): Flow<AgentResponse> {
        // Basic implementation, can be overridden for more complex streaming logic
        return flow {
            // For simplicity, using a dummy context. Subclasses should provide meaningful context.
            emit(processRequest(request, "DefaultContext_BaseAgentFlow"))
        }
    }

    
}
