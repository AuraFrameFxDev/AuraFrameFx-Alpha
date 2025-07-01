// These mock implementations are for development/testing only.
// Replace with real service logic when integrating actual AI backends.

package dev.aurakai.auraframefx.ai.services

import dev.aurakai.auraframefx.ai.agents.Agent
import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.api.model.AgentType as ApiAgentType // Use api.model.AgentType
import dev.aurakai.auraframefx.model.AiRequest
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf

// kotlinx.coroutines.flow imports are not needed for the direct AgentResponse implementation

class MockAuraAIService : Agent {
    /**
 * Returns the name of the mock AI service ("MockAura").
 *
 * @return The service name, or null if not set.
 */
override fun getName(): String? = "MockAura"
    /**
 * Returns the agent type as `ApiAgentType.AURA`.
 *
 * @return The `ApiAgentType` representing the Aura agent.
 */
override fun getType(): ApiAgentType = ApiAgentType.AURA /**
     * Returns a mock AgentResponse for the given request and context, simulating an AuraAI agent reply.
     *
     * The response content includes the request query and provided context, with a fixed confidence score of 1.0.
     *
     * @param request The AI request containing the query.
     * @param context Additional context to include in the mock response.
     * @return A mock AgentResponse with the query and context.
     */
    override suspend fun processRequest(request: AiRequest, context: String): AgentResponse { // Added context
        return AgentResponse(
            content = "AuraAI mock response for: ${request.query} with context: $context",
            confidence = 1.0f
        )
    }
    /**
     * Returns a flow emitting a single mock AuraAI agent response for the given request.
     *
     * The response contains a fixed message based on the request query and a confidence score of 1.0.
     *
     * @return A flow emitting one mock AgentResponse.
     */
    override fun processRequestFlow(request: AiRequest): Flow<AgentResponse> { // Added from Agent interface
        return flowOf(AgentResponse("AuraAI mock flow response for: ${request.query}", 1.0f))
    }

    /**
 * Returns an empty map, indicating that this mock agent has no capabilities.
 *
 * @return An empty map of capabilities.
 */
fun getCapabilities(): Map<String, Any> = emptyMap() /**
 * Returns null to indicate that continuous memory is not supported in this mock AI service.
 */
    fun getContinuousMemory(): Any? = null /**
 * Returns an empty list, indicating that no ethical guidelines are defined for this mock AI service.
 */
    fun getEthicalGuidelines(): List<String> = emptyList() /**
 * Returns an empty list, indicating no learning history is available for the mock AI service.
 *
 * @return An empty list of learning history entries.
 */
    fun getLearningHistory(): List<String> = emptyList() // Removed override
}

class MockKaiAIService : Agent {
    /**
 * Returns the name of the mock AI service ("MockKai").
 *
 * @return The string "MockKai".
 */
override fun getName(): String? = "MockKai"
    /**
 * Returns the agent type as `ApiAgentType.KAI`.
 *
 * @return The `ApiAgentType` representing the Kai agent.
 */
override fun getType(): ApiAgentType = ApiAgentType.KAI /**
     * Returns a mock response for a KaiAI agent request, incorporating the provided query and context.
     *
     * @param request The AI request containing the query.
     * @param context Additional context to include in the mock response.
     * @return An AgentResponse with a fixed mock message and confidence score.
     */
    override suspend fun processRequest(request: AiRequest, context: String): AgentResponse { // Added context
        return AgentResponse(
            content = "KaiAI mock response for: ${request.query} with context: $context",
            confidence = 1.0f
        )
    }
    /**
     * Returns a flow emitting a single mock KaiAI agent response for the given request.
     *
     * The response contains a fixed message based on the request query and a confidence score of 1.0.
     *
     * @return A flow emitting one mock AgentResponse.
     */
    override fun processRequestFlow(request: AiRequest): Flow<AgentResponse> { // Added from Agent interface
        return flowOf(AgentResponse("KaiAI mock flow response for: ${request.query}", 1.0f))
    }

    /**
 * Returns an empty map, indicating that this mock agent has no capabilities.
 *
 * @return An empty map of capabilities.
 */
fun getCapabilities(): Map<String, Any> = emptyMap() /**
 * Returns null to indicate that continuous memory is not supported in this mock AI service.
 */
    fun getContinuousMemory(): Any? = null /**
 * Returns an empty list, indicating that no ethical guidelines are defined for this mock AI service.
 */
    fun getEthicalGuidelines(): List<String> = emptyList() /**
 * Returns an empty list, indicating no learning history is available for the mock AI service.
 *
 * @return An empty list of learning history entries.
 */
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
 * Returns the agent type as `ApiAgentType.CASCADE`.
 *
 * @return The `ApiAgentType` representing the Cascade mock agent.
 */
override fun getType(): ApiAgentType = ApiAgentType.CASCADE /**
     * Returns a mock response for a CascadeAI agent request, incorporating the provided query and context.
     *
     * @param request The AI request containing the query.
     * @param context Additional context to include in the mock response.
     * @return An AgentResponse with a fixed mock message and confidence score.
     */
    override suspend fun processRequest(request: AiRequest, context: String): AgentResponse { // Added context
        return AgentResponse(
            content = "CascadeAI mock response for: ${request.query} with context: $context",
            confidence = 1.0f
        )
    }
    /**
     * Returns a Flow emitting a single mock AgentResponse for the given request, simulating a CascadeAI agent response.
     *
     * The response contains a fixed message referencing the request query and a confidence score of 1.0.
     *
     * @return A Flow emitting one mock AgentResponse.
     */
    override fun processRequestFlow(request: AiRequest): Flow<AgentResponse> { // Added from Agent interface
        return flowOf(AgentResponse("CascadeAI mock flow response for: ${request.query}", 1.0f))
    }

    /**
 * Returns an empty map, indicating that this mock agent has no capabilities.
 *
 * @return An empty map of capabilities.
 */
fun getCapabilities(): Map<String, Any> = emptyMap() /**
 * Returns null to indicate that continuous memory is not supported in this mock AI service.
 */
    fun getContinuousMemory(): Any? = null /**
 * Returns an empty list, indicating that no ethical guidelines are defined for this mock AI service.
 */
    fun getEthicalGuidelines(): List<String> = emptyList() /**
 * Returns an empty list, indicating no learning history is available for the mock AI service.
 *
 * @return An empty list of learning history entries.
 */
    fun getLearningHistory(): List<String> = emptyList() // Removed override
}
