package dev.aurakai.auraframefx.ai.services

import dev.aurakai.auraframefx.ai.agents.Agent
import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AgentType
import dev.aurakai.auraframefx.model.AiRequest
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class CascadeAIService @Inject constructor(
    private val auraService: AuraAIService,
    private val kaiService: KaiAIService,
) : Agent {

    private val state = mutableMapOf<String, Any>()

    /**
 * Returns the name of the agent.
 *
 * @return The string "Cascade".
 */
    override fun getName(): String? = "Cascade"

    /**
 * Returns the type of this agent as `AgentType.CASCADE`.
 *
 * @return The agent type, always `AgentType.CASCADE`.
 */
    override fun getType(): AgentType = AgentType.CASCADE

    /**
     * Emits agent responses as a flow by routing the AI request to an internal handler based on the "type" key in the request's context.
     *
     * Supported types are "state", "context", "vision", and "processing". If the type is missing or unrecognized, emits a default response with moderate confidence.
     *
     * @param request The AI request containing the query and context information.
     * @return A flow emitting agent responses determined by the request type.
     */
    override fun processRequestFlow(request: AiRequest): Flow<AgentResponse> {
        // This internal routing can stay if these specific flows are desired for internal logic
        // Assuming 'type' is passed within the context map
        return when (request.context?.get("type")) {
            "state" -> processStateRequestFlowInternal(request)
            "context" -> processContextRequestFlowInternal(request)
            "vision" -> processVisionRequestFlowInternal(request)
            "processing" -> processProcessingRequestFlowInternal(request)
            else -> flow {
                emit(AgentResponse("Cascade flow response for basic query: ${request.query}", 0.7f))
            } // Default flow for basic queries
        }
    }

    /**
     * Generates a direct response to an AI request by combining the request's query with the provided context.
     *
     * The response content includes both the query and the context, and the confidence score is set to 0.75.
     *
     * @param request The AI request containing the query.
     * @param context The context string to be included in the response.
     * @return An [AgentResponse] with the combined content and a fixed confidence score.
     */
    override suspend fun processRequest(
        request: AiRequest,
        context: String
    ): AgentResponse { // Added context
        // Example: collect from the flow, or implement separate direct logic
        return AgentResponse(
            content = "Cascade direct response to '${request.query}' with context '$context'",
            confidence = 0.75f
        )
    }

    /**
     * Returns a flow emitting a single response that summarizes all key-value pairs in the agent's internal state.
     *
     * The response content lists the current state entries with full confidence.
     *
     * @return A flow emitting one AgentResponse describing the current internal state.
     */
    private fun processStateRequestFlowInternal(request: AiRequest): Flow<AgentResponse> {
        return flow {
            emit(
                AgentResponse(
                    // type = "state",
                    content = "Current state: ${state.entries.joinToString { "${it.key}: ${it.value}" }}",
                    confidence = 1.0f
                )
            )
        }
    }

    /**
     * Aggregates responses from Aura and Kai AI services for a context-type AI request.
     *
     * Concurrently retrieves the first response from both services, then emits a single AgentResponse combining their contents and averaging their confidence scores.
     *
     * @return A flow emitting the aggregated AgentResponse.
     */
    private fun processContextRequestFlowInternal(request: AiRequest): Flow<AgentResponse> { // Made internal
        return flow {
            // Coordinate with Aura and Kai
            val auraResponse = auraService.processRequestFlow(request)
                .first() // Assumes AuraAIService has this method matching Agent iface
            val kaiResponse = kaiService.processRequestFlow(request)
                .first()   // Assumes KaiAIService has this method matching Agent iface

            emit(
                AgentResponse(
                    content = "Aura: ${auraResponse.content}, Kai: ${kaiResponse.content}",
                    confidence = (auraResponse.confidence + kaiResponse.confidence) / 2
                )
            )
        }
    }

    /**
     * Emits a single response indicating that a vision state request is being processed.
     *
     * @return A [Flow] emitting one [AgentResponse] with a message about vision state processing and a confidence score of 0.9.
     */
    private fun processVisionRequestFlowInternal(request: AiRequest): Flow<AgentResponse> { // Made internal
        // Process vision state
        return flow {
            emit(
                AgentResponse(
                    content = "Processing vision state...",
                    confidence = 0.9f
                )
            )
        }
    }

    /**
     * Emits a flow with a single response indicating that a state transition is in progress.
     *
     * @return A flow emitting one AgentResponse with a message about processing a state transition and a confidence score of 0.9.
     */
    private fun processProcessingRequestFlowInternal(request: AiRequest): Flow<AgentResponse> { // Made internal
        // Process state transitions
        return flow {
            emit(
                AgentResponse(
                    content = "Processing state transition...",
                    confidence = 0.9f
                )
            )
        }
    }

    /**
     * Emits a flow containing a single response indicating that the agent is retrieving its state history.
     *
     * @return A flow emitting an [AgentResponse] with a retrieval message and a confidence score of 0.95.
     */
    fun retrieveMemoryFlow(request: AiRequest): Flow<AgentResponse> { // Not in Agent interface, removed suspend, kept public if used elsewhere
        // Retrieve state history
        return flow {
            emit(
                AgentResponse(
                    content = "Retrieving state history...",
                    confidence = 0.95f
                )
            )
        }
    }

    // connect and disconnect are not part of Agent interface - removing these methods
    // as they cause unresolved reference errors

    /**
     * Returns a map detailing the agent's capabilities, including its name, type, and implementation status.
     *
     * @return A map with keys "name" (String), "type" (String), and "service_implemented" (Boolean).
     */
    fun getCapabilities(): Map<String, Any> {
        return mapOf(
            "name" to "Cascade",
            "type" to "CASCADE",
            "service_implemented" to true
        )
    }

    fun getContinuousMemory(): Any? {
        return state // Example: Cascade's state can be its continuous memory
    }

    fun getEthicalGuidelines(): List<String> {
        return listOf("Maintain state integrity.", "Process information reliably.")
    }

    fun getLearningHistory(): List<String> {
        return emptyList() // Or logs of state changes
    }
}
