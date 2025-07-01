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

    override fun getName(): String? {
        return _agentName
    }

    override fun getType(): AgentType { // Return non-nullable AgentType from api.model
        return try {
            AgentType.valueOf(_agentType.uppercase())
        } catch (e: IllegalArgumentException) {
            // Or handle error more gracefully, e.g., map to a default or throw
            throw IllegalArgumentException("Invalid agent type string: $_agentType", e)
        }
    }


    override suspend fun processRequest(request: AiRequest, context: String): AgentResponse {
        // Default implementation for base agent, override in subclasses
        // Added 'context' parameter to match interface
        // Used request.prompt instead of request.query
        // Used isSuccess instead of confidence
        return AgentResponse(

            content = "BaseAgent response to '${request.prompt}' for agent $_agentName with context '$context'",
            isSuccess = true
        )
    }

    override fun processRequestFlow(request: AiRequest): Flow<AgentResponse> {
        // Basic implementation, can be overridden for more complex streaming logic
        return flow {
            // For simplicity, using a dummy context. Subclasses should provide meaningful context.
            emit(processRequest(request, "DefaultContext_BaseAgentFlow"))
        }
    }

    
}
