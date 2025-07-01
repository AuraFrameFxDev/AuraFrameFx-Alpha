package dev.aurakai.auraframefx.ai.agents

import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AgentType
import dev.aurakai.auraframefx.model.AiRequest

/**
 * Base implementation of the [Agent] interface.
 * TODO: Reported as unused declaration. Ensure this class is subclassed or used.
 * TODO: Constructor parameters _agentName, _agentType were reported as unused in the base class context.
 * @param agentName The name of the agent.
 * @param agentType The type or model of the agent.
 */
open class BaseAgent(
    private val _agentName: String, // Underscore if constructor params are directly for properties not overridden
    private val _agentType: String,
) : Agent {

    // TODO: Consider if _agentName and _agentType should be used for overridden getName() and getType()
    // or if those methods should allow further customization by subclasses.

    override fun getName(): String? { // Changed to String?
        // TODO: Reported as unused (inherited method).
        return _agentName // Using constructor parameter
    }

    override fun getType(): AgentType? { // Changed to AgentType?
        // TODO: Reported as unused (inherited method).
        // Basic mapping from string to AgentType enum, adjust as needed
        return try {
            AgentType.valueOf(_agentType.uppercase())
        } catch (e: IllegalArgumentException) {
            null // Or a default AgentType
        }
    }

    override suspend fun processRequest(request: AiRequest): AgentResponse {
        // Default implementation for base agent, override in subclasses
        return AgentResponse(
            content = "BaseAgent response to '${request.query}' for agent $_agentName",
            confidence = 0.5f
        )
    }

    override fun getCapabilities(): Map<String, Any> {
        // TODO: Reported as unused (inherited method).
        // TODO: Implement base capabilities or leave for subclasses.
        return mapOf("name" to _agentName, "type" to _agentType, "base_implemented" to true)
    }

    override fun getContinuousMemory(): Any? {
        // TODO: Reported as unused (inherited method).
        // TODO: Implement base memory access or leave for subclasses.
        return null
    }

    override fun getEthicalGuidelines(): List<String> {
        // TODO: Reported as unused (inherited method).
        // TODO: Implement base ethical guidelines or leave for subclasses.
        return listOf("Be helpful.", "Be harmless.", "Adhere to base agent principles.")
    }

    override fun getLearningHistory(): List<String> {
        // TODO: Reported as unused (inherited method).
        // TODO: Implement base learning history access or leave for subclasses.
        return emptyList()
    }
}
