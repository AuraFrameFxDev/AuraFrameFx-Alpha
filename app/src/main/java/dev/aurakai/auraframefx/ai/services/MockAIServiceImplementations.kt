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
 * Gets the fixed name identifying this mock AI service.
 *
 * @return The string "MockAura".
 */
override fun getName(): String? = "MockAura"
    /**
 * Returns the agent type associated with this mock AI service.
 *
 * @return The `AgentType.AURA` enum value.
 */
override fun getType(): AgentType = AgentType.AURA /**
     * Generates a mock AgentResponse echoing the input query and context for testing.
     *
     * The response content includes the provided query and context, with a fixed confidence score of 1.0.
     *
     * @param request The AI request whose query is echoed in the response.
     * @param context The context string included in the response content.
     * @return An AgentResponse containing the mock response and a confidence score of 1.0.
     */
    override suspend fun processRequest(request: AiRequest, context: String): AgentResponse { // Added context
        return AgentResponse(
            content = "AuraAI mock response for: ${request.query} with context: $context",
            confidence = 1.0f
        )
    }
    /**
     * Returns a flow emitting a mock `AgentResponse` for the given AI request.
     *
     * The response content echoes the input query with a fixed confidence score of 1.0.
     *
     * @param request The AI request containing the query to be echoed in the mock response.
     * @return A flow emitting a single mock `AgentResponse`.
     */
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
 * Returns the name identifier for this mock AI service.
 *
 * @return The string "MockKai".
 */
override fun getName(): String? = "MockKai"
    /**
 * Returns the agent type for this mock Kai AI service.
 *
 * @return The `AgentType.KAI` enum value.
 */
override fun getType(): AgentType = AgentType.KAI /**
     * Generates a mock AgentResponse for the given request and context, echoing the input with a fixed confidence score.
     *
     * @param request The AI request containing the query to be echoed in the response.
     * @param context Additional context to include in the mock response.
     * @return An AgentResponse with content reflecting the input query and context, and a confidence score of 1.0.
     */
    override suspend fun processRequest(request: AiRequest, context: String): AgentResponse { // Added context
        return AgentResponse(
            content = "KaiAI mock response for: ${request.query} with context: $context",
            confidence = 1.0f
        )
    }
    /**
     * Returns a flow emitting a single mock `AgentResponse` for the given request.
     *
     * The response content echoes the input query with a fixed confidence score of 1.0.
     *
     * @param request The AI request containing the query to be echoed in the mock response.
     * @return A flow emitting one mock `AgentResponse`.
     */
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
 * Returns the fixed name identifier for the mock Cascade AI service.
 *
 * @return The string "MockCascade".
 */
override fun getName(): String? = "MockCascade"
    /**
 * Gets the agent type for this mock service.
 *
 * @return The AgentType.CASCADE value.
 */
override fun getType(): AgentType = AgentType.CASCADE /**
     * Generates a mock AgentResponse for CascadeAI, echoing the input query and context with a fixed confidence score.
     *
     * @param request The AI request containing the query to be echoed.
     * @param context The context string to include in the mock response.
     * @return An AgentResponse containing a mock response referencing the query and context, with a confidence score of 1.0.
     */
    override suspend fun processRequest(request: AiRequest, context: String): AgentResponse { // Added context
        return AgentResponse(
            content = "CascadeAI mock response for: ${request.query} with context: $context",
            confidence = 1.0f
        )
    }
    /**
     * Returns a flow emitting a mock `AgentResponse` for the given request.
     *
     * The response content echoes the input query with a fixed confidence score of 1.0.
     *
     * @return A flow containing a single mock `AgentResponse`.
     */
    override fun processRequestFlow(request: AiRequest): Flow<AgentResponse> { // Added from Agent interface
        return flowOf(AgentResponse("CascadeAI mock flow response for: ${request.query}", 1.0f))
    }

    fun getCapabilities(): Map<String, Any> = emptyMap() // Removed override
    fun getContinuousMemory(): Any? = null // Removed override
    fun getEthicalGuidelines(): List<String> = emptyList() // Removed override
    fun getLearningHistory(): List<String> = emptyList() // Removed override
}
