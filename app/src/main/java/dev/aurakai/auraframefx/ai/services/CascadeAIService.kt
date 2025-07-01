package dev.aurakai.auraframefx.ai.services

import dev.aurakai.auraframefx.ai.agents.Agent
import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.api.model.AgentType as ApiAgentType // Corrected import
import dev.aurakai.auraframefx.model.AiRequest
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first // Keep for existing logic if processRequestFlow uses it
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
 * Returns the name of the agent, which is "Cascade".
 *
 * @return The agent's name.
 */
override fun getName(): String? = "Cascade"

    /**
 * Returns the agent type as `ApiAgentType.CASCADE`.
 *
 * @return The type of this agent.
 */
override fun getType(): ApiAgentType = ApiAgentType.CASCADE // Changed to non-nullable ApiAgentType

    /**
     * Processes an AI request and returns a flow of agent responses based on the request type.
     *
     * Routes the request to specialized internal handlers for "state", "context", "vision", and "processing" types.
     * For unrecognized types, emits a default response containing the query.
     *
     * @param request The AI request to process.
     * @return A flow emitting one or more agent responses relevant to the request type.
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
     * Processes an AI request and returns a direct response incorporating the provided context.
     *
     * @param request The AI request to process.
     * @param context Additional context information to include in the response.
     * @return An [AgentResponse] containing the request query and context with a fixed confidence score.
     */
    override suspend fun processRequest(request: AiRequest, context: String): AgentResponse { // Added context
        // Example: collect from the flow, or implement separate direct logic
        return AgentResponse(
            content = "Cascade direct response to '${request.query}' with context '$context'",
            confidence = 0.75f
        )
    }

    /**
     * Emits an AgentResponse containing the current internal state as a formatted string.
     *
     * @param request The AI request triggering the state retrieval.
     * @return A flow emitting a single AgentResponse with the state details and full confidence.
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
     * Collects the first response from each service, combines their contents, and averages their confidence scores into a single `AgentResponse` emitted as a flow.
     *
     * @return A flow emitting a combined agent response from Aura and Kai services.
     */
    private fun processContextRequestFlowInternal(request: AiRequest): Flow<AgentResponse> { // Made internal
        // Coordinate with Aura and Kai
        val auraResponse = auraService.processRequestFlow(request).first() // Assumes AuraAIService has this method matching Agent iface
        val kaiResponse = kaiService.processRequestFlow(request).first()   // Assumes KaiAIService has this method matching Agent iface

        return flow {
            emit(
                AgentResponse(
                    content = "Aura: ${auraResponse.content}, Kai: ${kaiResponse.content}",
                    confidence = (auraResponse.confidence + kaiResponse.confidence) / 2
                )
            )
        }
    }

    /**
     * Emits a response indicating that a vision state request is being processed.
     *
     * @return A flow emitting a single AgentResponse with a fixed message and confidence score.
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
     * Emits a response indicating that a state transition is being processed.
     *
     * @return A flow emitting a single AgentResponse with a fixed message and confidence score.
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
     * Returns a flow emitting a response indicating retrieval of the agent's state history.
     *
     * @return A flow containing a single AgentResponse about state history retrieval.
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

    /**
     * Establishes connections to both the Aura and Kai AI services.
     *
     * @return `true` if both services connect successfully; otherwise, `false`.
     */
    fun connect(): Boolean { // Removed suspend
        // Assuming auraService and kaiService have connect methods
        return auraService.connect() && kaiService.connect()
    }

    /**
     * Disconnects both the Aura and Kai AI services.
     *
     * @return `true` if both services disconnect successfully; otherwise, `false`.
     */
    fun disconnect(): Boolean { // Removed suspend
        // Assuming auraService and kaiService have disconnect methods
        return auraService.disconnect() && kaiService.disconnect()
    }

    /**
     * Returns a map describing the agent's capabilities, including its name, type, and service implementation status.
     *
     * @return A map containing capability metadata for the agent.
     */
    fun getCapabilities(): Map<String, Any> {
        return mapOf(
            "name" to "Cascade",
            "type" to ApiAgentType.CASCADE,
            "service_implemented" to true
        )
    }

    /**
     * Returns the agent's continuous memory, represented by its internal state map.
     *
     * @return The current state map containing key-value pairs of internal memory, or null if not set.
     */
    fun getContinuousMemory(): Any? {
        return state // Example: Cascade's state can be its continuous memory
    }

    /**
     * Returns a list of ethical guidelines that the agent follows.
     *
     * @return A list of strings describing the agent's ethical principles.
     */
    fun getEthicalGuidelines(): List<String> {
        return listOf("Maintain state integrity.", "Process information reliably.")
    }

    /**
     * Returns the learning or state change history of the agent.
     *
     * Currently returns an empty list as a placeholder for future implementation.
     * @return An empty list representing the absence of learning history.
     */
    fun getLearningHistory(): List<String> {
        return emptyList() // Or logs of state changes
    }
}
