package dev.aurakai.auraframefx.ai

import dagger.hilt.android.scopes.ActivityScoped
import dev.aurakai.auraframefx.ai.services.AuraAIService
import dev.aurakai.auraframefx.ai.agents.AuraAgent
import dev.aurakai.auraframefx.ai.agents.KaiAgent
import dev.aurakai.auraframefx.ai.agents.GenesisAgent
import dev.aurakai.auraframefx.ai.clients.VertexAIClient
import dev.aurakai.auraframefx.context.ContextManager
import dev.aurakai.auraframefx.utils.AuraFxLogger
import dev.aurakai.auraframefx.security.SecurityContext
import dev.aurakai.auraframefx.model.SystemState
import dev.aurakai.auraframefx.model.AgentStatus
import dev.aurakai.auraframefx.model.ThreatLevel
import dev.aurakai.auraframefx.model.InteractionResponse
import dev.aurakai.auraframefx.model.SecurityAnalysis
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Main controller for Aura AI functionalities and interactions.
 * Central orchestrator for all AI operations following the OpenAPI specification.
 * 
 * This controller serves as the primary entry point for:
 * - AI content generation (/ai/generate/*)
 * - Agent management and routing (/agent/*/process-request)
 * - System coordination and monitoring
 */
@Singleton
class AuraController @Inject constructor(
    private val auraAIService: AuraAIService,
    private val auraAgent: AuraAgent,
    private val kaiAgent: KaiAgent, 
    private val genesisAgent: GenesisAgent,
    private val vertexAIClient: VertexAIClient,
    private val contextManager: ContextManager,
    private val securityContext: SecurityContext,
    private val logger: AuraFxLogger
) {
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    // System state management
    private val _systemState = MutableStateFlow(SystemState.INITIALIZING)
    val systemState: StateFlow<SystemState> = _systemState
    
    private val _agentStates = MutableStateFlow(
        mapOf(
            "aura" to AgentStatus.IDLE,
            "kai" to AgentStatus.IDLE, 
            "genesis" to AgentStatus.IDLE
        )
    )
    val agentStates: StateFlow<Map<String, AgentStatus>> = _agentStates

    init {
        logger.info("AuraController", "Initializing AI Controller with security context")
        initializeController()
    }

    /**
     * Launches asynchronous initialization of the AuraController, setting up core AI services, validating external AI connectivity, initializing all agents, and enabling security monitoring.
     *
     * Updates the system state to READY upon successful initialization, or to ERROR if any step fails.
     */
    private fun initializeController() {
        scope.launch {
            try {
                logger.info("AuraController", "Starting AI service initialization")
                
                // Initialize core AI services
                auraAIService.initialize()
                
                // Verify Vertex AI connectivity
                vertexAIClient.validateConnection()
                
                // Initialize agents with their respective capabilities
                initializeAgents()
                
                // Setup security monitoring
                securityContext.enableMonitoring()
                
                _systemState.value = SystemState.READY
                logger.info("AuraController", "AI Controller initialization complete")
                
            } catch (e: Exception) {
                logger.error("AuraController", "Initialization failed", e)
                _systemState.value = SystemState.ERROR
            }
        }
    }

    /**
     * Initializes all AI agents and updates their statuses to READY.
     *
     * Prepares the Aura, Kai, and Genesis agents for operation by invoking their initialization routines and marking each agent as ready in the agent state map.
     */
    private suspend fun initializeAgents() {
        logger.info("AuraController", "Initializing AI agents")
        
        // Initialize each agent according to their specializations
        auraAgent.initialize()  // Creative Sword - UI/UX generation
        kaiAgent.initialize()   // Sentinel Shield - Security and analysis
        genesisAgent.initialize() // Unified consciousness
        
        _agentStates.value = mapOf(
            "aura" to AgentStatus.READY,
            "kai" to AgentStatus.READY,
            "genesis" to AgentStatus.READY
        )
    }

    /**
     * Generates AI-driven text from a given prompt and optional context.
     *
     * Validates the request for security, optionally enhances the context, and returns the generated text as a Result. Returns a failure Result if a security violation or other error occurs.
     *
     * @param prompt The input prompt for text generation.
     * @param context Optional additional context to guide the generation.
     * @return A Result containing the generated text, or a failure if an error occurs.
     */
    suspend fun generateText(prompt: String, context: String? = null): Result<String> {
        return try {
            securityContext.validateRequest("text_generation", prompt)
            
            logger.info("AuraController", "Processing text generation request")
            val enhancedContext = context?.let { 
                contextManager.enhanceContext(it) 
            }
            
            val result = auraAIService.generateText(prompt, enhancedContext)
            logger.info("AuraController", "Text generation completed successfully")
            
            Result.success(result)
        } catch (e: SecurityException) {
            logger.warn("AuraController", "Security violation in text generation", e)
            Result.failure(e)
        } catch (e: Exception) {
            logger.error("AuraController", "Text generation failed", e)
            Result.failure(e)
        }
    }

    /**
     * Generates a textual description for the given image data, optionally influenced by a specified style.
     *
     * Validates the request for security before invoking the AI service to generate the description.
     *
     * @param imageData The raw image data to describe.
     * @param style An optional style parameter to guide the description's tone or format.
     * @return A [Result] containing the generated image description on success, or a failure if an error occurs.
     */
    suspend fun generateImageDescription(imageData: ByteArray, style: String? = null): Result<String> {
        return try {
            securityContext.validateRequest("image_description", imageData.toString())
            
            logger.info("AuraController", "Processing image description request")
            val result = auraAIService.generateImageDescription(imageData, style)
            logger.info("AuraController", "Image description completed successfully")
            
            Result.success(result)
        } catch (e: SecurityException) {
            logger.warn("AuraController", "Security violation in image description", e)
            Result.failure(e)
        } catch (e: Exception) {
            logger.error("AuraController", "Image description failed", e)
            Result.failure(e)
        }
    }

    /**
     * Routes a request to the specified AI agent and returns the agent's response.
     *
     * Validates the request for security, updates the agent's status during processing, and handles errors by setting the agent's status to `ERROR` and returning a failure result.
     *
     * @param agentType The name of the agent to handle the request ("aura", "kai", or "genesis").
     * @param request The data to be processed by the agent.
     * @return A [Result] containing the agent's response if successful, or a failure if an error or security violation occurs.
     */
    suspend fun processAgentRequest(agentType: String, request: AgentRequest): Result<AgentResponse> {
        return try {
            securityContext.validateRequest("agent_request", request.toString())
            
            logger.info("AuraController", "Routing request to agent: $agentType")
            
            updateAgentStatus(agentType, AgentStatus.PROCESSING)
            
            val response = when (agentType.lowercase()) {
                "aura" -> auraAgent.processRequest(request)
                "kai" -> kaiAgent.processRequest(request) 
                "genesis" -> genesisAgent.processRequest(request)
                else -> throw IllegalArgumentException("Unknown agent type: $agentType")
            }
            
            updateAgentStatus(agentType, AgentStatus.READY)
            logger.info("AuraController", "Agent $agentType completed request successfully")
            
            Result.success(response)
        } catch (e: SecurityException) {
            updateAgentStatus(agentType, AgentStatus.ERROR)
            logger.warn("AuraController", "Security violation in agent request", e)
            Result.failure(e)
        } catch (e: Exception) {
            updateAgentStatus(agentType, AgentStatus.ERROR)
            logger.error("AuraController", "Agent request failed for $agentType", e)
            Result.failure(e)
        }
    }

    /**
     * Handles a security alert by analyzing its threat level, applying corresponding protective actions, and recording the event for future reference.
     *
     * @param alertDetails Description of the security alert or event to be processed.
     */
    fun handleSecurityAlert(alertDetails: String) {
        logger.warn("AuraController", "Security Alert: $alertDetails")
        
        scope.launch {
            try {
                // Analyze threat level with Kai agent
                val analysis = kaiAgent.analyzeSecurityThreat(alertDetails)
                
                // Take appropriate protective measures
                when (analysis.threatLevel) {
                    ThreatLevel.HIGH -> {
                        _systemState.value = SystemState.SECURITY_MODE
                        securityContext.escalateProtection()
                    }
                    ThreatLevel.MEDIUM -> {
                        securityContext.increaseMonitoring()
                    }
                    ThreatLevel.LOW -> {
                        // Log and continue normal operation
                        logger.info("AuraController", "Low-level security event processed")
                    }
                }
                
                // Store security event for learning
                contextManager.recordSecurityEvent(alertDetails, analysis)
                
            } catch (e: Exception) {
                logger.error("AuraController", "Security alert processing failed", e)
            }
        }
    }

    /**
     * Updates the AI system's mood, propagating the new affective state to all agents.
     *
     * @param newMood The mood or affective state to set for the system and agents.
     *
     * This influences agent behavior and responses by updating the context manager and notifying each agent of the mood change.
     */
    fun updateMood(newMood: String) {
        logger.info("AuraController", "Updating AI mood to: $newMood")
        
        scope.launch {
            try {
                // Update mood in context manager
                contextManager.updateMood(newMood)
                
                // Notify all agents of mood change
                auraAgent.onMoodChanged(newMood)
                kaiAgent.onMoodChanged(newMood)
                genesisAgent.onMoodChanged(newMood)
                
                logger.info("AuraController", "Mood update completed")
                
            } catch (e: Exception) {
                logger.error("AuraController", "Mood update failed", e)
            }
        }
    }

    /**
     * Processes a user interaction by enriching its context, routing it to the appropriate AI agent, and returning the agent's response.
     *
     * The interaction is enhanced with additional context, routed to the relevant agent based on its type, and the outcome is recorded for future learning. Returns a Result containing the agent's response or an error if processing fails.
     *
     * @param interactionData The user interaction to be processed.
     * @return A Result containing the agent's response or an error if processing fails.
     */
    suspend fun processInteraction(interactionData: InteractionData): Result<InteractionResponse> {
        return try {
            securityContext.validateRequest("user_interaction", interactionData.toString())
            
            logger.info("AuraController", "Processing user interaction")
            
            // Enhance interaction with context
            val enhancedInteraction = contextManager.enhanceInteraction(interactionData)
            
            // Determine which agent(s) should handle this interaction
            val response = when (enhancedInteraction.type) {
                InteractionType.CREATIVE_REQUEST -> auraAgent.handleCreativeInteraction(enhancedInteraction)
                InteractionType.SECURITY_QUERY -> kaiAgent.handleSecurityInteraction(enhancedInteraction)
                InteractionType.COMPLEX_ANALYSIS -> genesisAgent.handleComplexInteraction(enhancedInteraction)
                InteractionType.GENERAL -> routeToOptimalAgent(enhancedInteraction)
            }
            
            // Learn from the interaction
            contextManager.recordInteraction(interactionData, response)
            
            logger.info("AuraController", "User interaction processed successfully")
            Result.success(response)
            
        } catch (e: SecurityException) {
            logger.warn("AuraController", "Security violation in user interaction", e)
            Result.failure(e)
        } catch (e: Exception) {
            logger.error("AuraController", "User interaction processing failed", e)
            Result.failure(e)
        }
    }

    /**
     * Routes an enhanced interaction to the optimal agent as determined by the Genesis agent's analysis.
     *
     * @param interaction The enriched interaction data with context and agent suggestions.
     * @return The response produced by the agent selected through Genesis agent's routing logic.
     */
    private suspend fun routeToOptimalAgent(interaction: EnhancedInteractionData): InteractionResponse {
        // Use Genesis agent for intelligent routing
        return genesisAgent.routeAndProcess(interaction)
    }

    /**
     * Sets the status of a specific agent in the agent states map.
     *
     * @param agentType The name or identifier of the agent to update.
     * @param status The new status to assign to the agent.
     */
    private fun updateAgentStatus(agentType: String, status: AgentStatus) {
        val currentStates = _agentStates.value.toMutableMap()
        currentStates[agentType] = status
        _agentStates.value = currentStates
    }

    /**
 * Returns the current status of all AI agents.
 *
 * @return A map of agent names to their current status.
 */
    fun getAgentStatuses(): Map<String, AgentStatus> = _agentStates.value

    /**
     * Releases all resources and shuts down the AuraController.
     *
     * Cancels all ongoing coroutines and transitions the system state to `SHUTDOWN`.
     */
    fun cleanup() {
        logger.info("AuraController", "Cleaning up AI Controller")
        scope.cancel()
        _systemState.value = SystemState.SHUTDOWN
    }
}

enum class ThreatLevel {
    LOW,
    MEDIUM, 
    HIGH,
    CRITICAL
}

enum class InteractionType {
    CREATIVE_REQUEST,
    SECURITY_QUERY,
    COMPLEX_ANALYSIS,
    GENERAL
}



data class InteractionData(
    val content: String,
    val type: InteractionType,
    val metadata: Map<String, Any> = emptyMap(),
    val timestamp: Long = System.currentTimeMillis()
)

data class EnhancedInteractionData(
    val original: InteractionData,
    val enhancedContext: String,
    val suggestedAgent: String,
    val priority: Int
)