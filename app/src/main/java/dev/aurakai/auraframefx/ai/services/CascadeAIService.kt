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
 * Returns the agent's name, which is "Cascade".
 *
 * @return The name of the agent.
 */
    override fun getName(): String? = "Cascade"

    /**
 * Returns the agent type identifier for this agent.
 *
 * @return `AgentType.CASCADE`, indicating this is a Cascade agent.
 */
    override fun getType(): AgentType = AgentType.CASCADE

    /**
     * Routes an AI request to the appropriate handler and returns a flow emitting agent responses based on the request type.
     *
     * For recognized types ("state", "context", "vision", "processing"), delegates to specialized internal handlers. For unrecognized types, emits a default response with a basic query message and a confidence score of 0.7.
     *
     * @param request The AI request to be processed.
     * @return A flow emitting agent responses determined by the request type.
     */
    override fun processRequestFlow(request: AiRequest): Flow<AgentResponse> {
        // This internal routing can stay if these specific flows are desired for internal logic
        return when (request.type) {
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
     * Returns a direct response to an AI request, incorporating the provided context.
     *
     * The response content combines the request's query and the given context, with a fixed confidence score of 0.75.
     *
     * @param request The AI request to respond to.
     * @param context Contextual information to include in the response.
     * @return An [AgentResponse] containing the combined query and context.
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
     * Returns a flow that emits a single response summarizing all key-value pairs in the agent's internal state.
     *
     * The response content lists the current state entries, and the confidence score is set to 1.0.
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
     * Aggregates the first response from both Aura and Kai AI services for a context-type request into a single AgentResponse.
     *
     * The returned flow emits one AgentResponse that combines the content from both services and averages their confidence scores.
     *
     * @param request The AI request to process.
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
     * Emits a flow containing a single response indicating that vision state processing is underway.
     *
     * @return A [Flow] emitting one [AgentResponse] with a message about ongoing vision state processing and a confidence score of 0.9.
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
     * Emits a flow containing a single response indicating that a state transition is being processed.
     *
     * @return A flow emitting an AgentResponse with a message about state transition processing and a confidence score of 0.9.
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
     * Returns a flow that emits a single response indicating the retrieval of the agent's state history.
     *
     * @return A flow emitting one [AgentResponse] with a message about retrieving state history and a confidence score of 0.95.
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
     * @return A map with keys for "name", "type", and "service_implemented".
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
