// These mock implementations are for development/testing only.
// Replace with real service logic when integrating actual AI backends.

package dev.aurakai.auraframefx.ai.services

import dev.aurakai.auraframefx.ai.agents.Agent
import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AgentType
import dev.aurakai.auraframefx.model.AiRequest
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf

// kotlinx.coroutines.flow imports are not needed for the direct AgentResponse implementation

class MockAuraAIService : Agent {
    /**
 * Returns the name of the mock AI service.
 *
 * @return The string "MockAura".
 */
override fun getName(): String? = "MockAura"
    /**
 * Returns the agent type as AURA.
 *
 * @return The AgentType.AURA enum value.
 */
override fun getType(): AgentType = AgentType.AURA /**
     * Processes an AI request and returns a mock response for AuraAI, including the provided context.
     *
     * @param request The AI request containing the query.
     * @param context Additional context to include in the mock response.
     * @return An AgentResponse with a mock response string and a confidence score of 1.0.
     */
    override suspend fun processRequest(request: AiRequest, context: String): AgentResponse { // Added context
        return AgentResponse(
            content = "AuraAI mock response for: ${request.query} with context: $context",
            confidence = 1.0f
        )
    }
    override fun processRequestFlow(request: AiRequest): Flow<AgentResponse> { // Added from Agent interface
        return flowOf(AgentResponse("AuraAI mock flow response for: ${request.query}", 1.0f))
    }

    fun getCapabilities(): Map<String, Any> = emptyMap() // Removed override
    fun getContinuousMemory(): Any? = null // Removed override
    fun getEthicalGuidelines(): List<String> = emptyList() // Removed override
    fun getLearningHistory(): List<String> = emptyList() // Removed override
}

class MockKaiAIService : Agent {
    /**
 * Returns the name identifier for the mock Kai AI service.
 *
 * @return The string "MockKai".
 */
override fun getName(): String? = "MockKai"
    /**
 * Returns the agent type as KAI.
 *
 * @return The AgentType.KAI enum value.
 */
override fun getType(): AgentType = AgentType.KAI /**
     * Processes an AI request and returns a mock response for KaiAI, including the provided context.
     *
     * @param request The AI request containing the query.
     * @param context Additional context to include in the mock response.
     * @return An AgentResponse with a mock response string and a confidence score of 1.0.
     */
    override suspend fun processRequest(request: AiRequest, context: String): AgentResponse { // Added context
        return AgentResponse(
            content = "KaiAI mock response for: ${request.query} with context: $context",
            confidence = 1.0f
        )
    }
    override fun processRequestFlow(request: AiRequest): Flow<AgentResponse> { // Added from Agent interface
        return flowOf(AgentResponse("KaiAI mock flow response for: ${request.query}", 1.0f))
    }

    fun getCapabilities(): Map<String, Any> = emptyMap() // Removed override
    fun getContinuousMemory(): Any? = null // Removed override
    fun getEthicalGuidelines(): List<String> = emptyList() // Removed override
    fun getLearningHistory(): List<String> = emptyList() // Removed override
}

class MockCascadeAIService : Agent {
    /**
 * Returns the name identifier for the mock Cascade AI service.
 *
 * @return The string "MockCascade".
 */
override fun getName(): String? = "MockCascade"
    /**
 * Returns the agent type as CASCADE.
 *
 * @return The AgentType.CASCADE enum value.
 */
override fun getType(): AgentType = AgentType.CASCADE /**
     * Generates a mock response for a CascadeAI agent based on the provided request and context.
     *
     * @param request The AI request containing the query.
     * @param context Additional context to include in the mock response.
     * @return An AgentResponse with a mock response string and a confidence score of 1.0.
     */
    override suspend fun processRequest(request: AiRequest, context: String): AgentResponse { // Added context
        return AgentResponse(
            content = "CascadeAI mock response for: ${request.query} with context: $context",
            confidence = 1.0f
        )
    }
    override fun processRequestFlow(request: AiRequest): Flow<AgentResponse> { // Added from Agent interface
        return flowOf(AgentResponse("CascadeAI mock flow response for: ${request.query}", 1.0f))
    }

    fun getCapabilities(): Map<String, Any> = emptyMap() // Removed override
    fun getContinuousMemory(): Any? = null // Removed override
    fun getEthicalGuidelines(): List<String> = emptyList() // Removed override
    fun getLearningHistory(): List<String> = emptyList() // Removed override
}
