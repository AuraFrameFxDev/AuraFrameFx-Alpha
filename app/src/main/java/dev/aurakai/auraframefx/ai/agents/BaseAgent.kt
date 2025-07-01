package dev.aurakai.auraframefx.ai.agents

import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AiRequest
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import dev.aurakai.auraframefx.api.model.AgentType as ApiAgentType // Use alias for generated type

/**
 * Base implementation of the [Agent] interface.
 * @param agentName The name of the agent.
 * @param agentType The type or model of the agent (string representation).
 */
open class BaseAgent(
    private val _agentName: String,
    private val _agentType: String, // This will be mapped to ApiAgentType
) : Agent {

    override fun getName(): String? {
        return _agentName
    }

    override fun getType(): ApiAgentType {
        // Map string to the generated ApiAgentType
        // Fallback to a default or throw an error if mapping fails
        return ApiAgentType.values().firstOrNull { it.value.equals(_agentType, ignoreCase = true) }
            ?: ApiAgentType.Aura // Defaulting to Aura, consider a more robust fallback or error
    }

    /**
     * Default implementation for processing a request. Subclasses should override this.
     */
    override suspend fun processRequest(request: AiRequest, context: String): AgentResponse {
        // Default implementation for base agent, override in subclasses
        return AgentResponse(
            content = "BaseAgent response to '${request.query}' in context '$context' for agent $_agentName",
            confidence = 0.5f
        )
    }

    /**
     * Default implementation for processing a request and returning a flow. Subclasses should override this.
     */
    override fun processRequestFlow(request: AiRequest): Flow<AgentResponse> {
        // Default implementation, consider returning a flow with a single response
        // or an empty flow if not applicable for a base agent.
        return flowOf(
            AgentResponse(
                content = "BaseAgent flow response to '${request.query}' for agent $_agentName",
                confidence = 0.5f
            )
        )
    }

    // Removed getCapabilities, getContinuousMemory, getEthicalGuidelines, getLearningHistory
    // as they are not part of the Agent interface.
}
