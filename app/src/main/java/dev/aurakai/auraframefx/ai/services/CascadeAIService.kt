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
 * Returns the name of this agent.
 *
 * @return The agent name, "Cascade".
 */
override fun getName(): String? = "Cascade"

    /**
 * Gets the type of this agent.
 *
 * @return The agent type, always `AgentType.CASCADE`.
 */
override fun getType(): AgentType = AgentType.CASCADE

    /**
     * Processes an AI request and emits agent responses as a flow, routing by request type.
     *
     * Delegates to specialized internal handlers for "state", "context", "vision", and "processing" request types. Emits a default response for unrecognized types.
     *
     * @param request The AI request to handle.
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

    // Implemented Agent interface method
    override suspend fun processRequest(request: AiRequest, context: String): AgentResponse { // Added context
        // Example: collect from the flow, or implement separate direct logic
        return AgentResponse(
            content = "Cascade direct response to '${request.query}' with context '$context'",
            confidence = 0.75f
        )
    }

    /**
     * Emits a single response containing a string representation of the current internal state.
     *
     * @return A flow emitting an AgentResponse with the current state details and full confidence.
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
     * Processes a context-type AI request by aggregating responses from both Aura and Kai services.
     *
     * Emits a single [AgentResponse] that combines the content and averages the confidence scores from the two services.
     *
     * @param request The AI request to process.
     * @return A [Flow] emitting the aggregated [AgentResponse].
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
     * Returns a flow emitting a single response indicating retrieval of the agent's state history.
     *
     * @return A [Flow] that emits an [AgentResponse] with a message about retrieving state history and a confidence score of 0.95.
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
     * Returns a map describing the agent's capabilities, including its name, type, and implementation status.
     *
     * @return A map containing capability metadata for the Cascade agent.
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
