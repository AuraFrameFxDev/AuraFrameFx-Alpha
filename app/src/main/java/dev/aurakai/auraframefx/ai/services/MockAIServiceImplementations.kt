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
 * Returns the fixed name "MockAura" identifying this mock AI service.
 *
 * @return The string "MockAura".
 */
    override fun getName(): String? = "MockAura"

    /**
 * Returns the agent type for this mock AI service.
 *
 * @return The `AgentType.AURA` enum value.
 */
    override fun getType(): AgentType = AgentType.AURA

    /**
     * Returns a mock AgentResponse echoing the input query and context for testing purposes.
     *
     * The response content includes the provided query and context, with a fixed confidence score of 1.0.
     *
     * @param request The AI request whose query is included in the mock response.
     * @param context The context string to be echoed in the response content.
     * @return An AgentResponse containing the mock content and a confidence score of 1.0.
     */
    override suspend fun processRequest(
        request: AiRequest,
        context: String
    ): AgentResponse { // Added context
        return AgentResponse(
            content = "AuraAI mock response for: ${request.query} with context: $context",
            confidence = 1.0f
        )
    }

    /**
     * Returns a flow that emits a mock `AgentResponse` echoing the input query with a fixed confidence score of 1.0.
     *
     * @return A flow containing a single mock `AgentResponse`.
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
 * Returns the fixed name identifier for the mock Kai AI service.
 *
 * @return The string "MockKai".
 */
    override fun getName(): String? = "MockKai"

    /**
 * Returns the agent type associated with the mock Kai AI service.
 *
 * @return The `AgentType.KAI` enum value.
 */
    override fun getType(): AgentType = AgentType.KAI

    /**
     * Returns a mock AgentResponse that echoes the input query and context with a fixed confidence score of 1.0.
     *
     * @param request The AI request containing the query to be included in the mock response.
     * @param context Additional context to be reflected in the mock response.
     * @return An AgentResponse with mock content and a confidence score of 1.0.
     */
    override suspend fun processRequest(
        request: AiRequest,
        context: String
    ): AgentResponse { // Added context
        return AgentResponse(
            content = "KaiAI mock response for: ${request.query} with context: $context",
            confidence = 1.0f
        )
    }

    /**
     * Returns a flow that emits a mock AgentResponse simulating KaiAI behavior for the given request.
     *
     * The response echoes the input query with a fixed confidence score of 1.0.
     *
     * @return A Flow emitting a single mock AgentResponse.
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
 * Returns the agent type for this mock service.
 *
 * @return The `AgentType.CASCADE` enum value.
 */
    override fun getType(): AgentType = AgentType.CASCADE

    /**
     * Returns a mock AgentResponse for CascadeAI, echoing the input query and context with a fixed confidence score.
     *
     * @param request The AI request containing the query to echo.
     * @param context The context string to include in the mock response.
     * @return An AgentResponse with mock content referencing the query and context, and a confidence score of 1.0.
     */
    override suspend fun processRequest(
        request: AiRequest,
        context: String
    ): AgentResponse { // Added context
        return AgentResponse(
            content = "CascadeAI mock response for: ${request.query} with context: $context",
            confidence = 1.0f
        )
    }

    /**
     * Returns a flow that emits a mock `AgentResponse` simulating a CascadeAI response for the provided request.
     *
     * The emitted response echoes the input query with a fixed confidence score of 1.0.
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
