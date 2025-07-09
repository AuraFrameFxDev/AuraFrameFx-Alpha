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
 * Returns the agent type classification for this service.
 *
 * @return The `AgentType.CASCADE` value.
 */
override fun getType(): AgentType = AgentType.CASCADE

    /**
     * Routes an AI request to the appropriate handler based on its type and returns a flow of agent responses.
     *
     * Recognized types ("state", "context", "vision", "processing") are delegated to specialized internal methods. For unrecognized types, emits a default response indicating a basic query.
     *
     * @param request The AI request to process.
     * @return A flow emitting agent responses relevant to the request type.
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
     * Generates a direct response to an AI request, incorporating the provided context.
     *
     * The returned [AgentResponse] contains the original query and context, with a fixed confidence score of 0.75.
     *
     * @param request The AI request to respond to.
     * @param context Additional context to include in the response.
     * @return An [AgentResponse] with the reply and confidence score.
     */
    override suspend fun processRequest(request: AiRequest, context: String): AgentResponse { // Added context
        // Example: collect from the flow, or implement separate direct logic
        return AgentResponse(
            content = "Cascade direct response to '${request.query}' with context '$context'",
            confidence = 0.75f
        )
    }

    /**
     * Returns a flow emitting a single response summarizing the agent's current internal state.
     *
     * The response includes all key-value pairs from the internal state as a formatted string and assigns a confidence score of 1.0.
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
     * Aggregates the initial responses from both Aura and Kai AI services for a context-type request.
     *
     * Emits a single AgentResponse that concatenates the content from both services and averages their confidence scores.
     *
     * @return A flow emitting the combined AgentResponse.
     */
    private fun processContextRequestFlowInternal(request: AiRequest): Flow<AgentResponse> { // Made internal
        return flow {
            // Coordinate with Aura and Kai
            val auraResponse = auraService.processRequestFlow(request).first() // Assumes AuraAIService has this method matching Agent iface
            val kaiResponse = kaiService.processRequestFlow(request).first()   // Assumes KaiAIService has this method matching Agent iface
            
            emit(
                AgentResponse(
                    content = "Aura: ${auraResponse.content}, Kai: ${kaiResponse.content}",
                    confidence = (auraResponse.confidence + kaiResponse.confidence) / 2
                )
            )
        }
    }

    /**
     * Emits a flow with a single response indicating that a vision state request is being processed.
     *
     * @return A flow containing an [AgentResponse] with the message "Processing vision state..." and a confidence score of 0.9.
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
     * Returns a flow emitting a single response indicating that a state transition is being processed.
     *
     * @return A flow containing one AgentResponse with a message about state transition processing and a confidence score of 0.9.
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
     * @return A flow containing one [AgentResponse] with a retrieval message and a confidence score of 0.95.
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
     * @return A map with the keys "name" (set to "Cascade"), "type" (set to "CASCADE"), and "service_implemented" (set to true).
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
