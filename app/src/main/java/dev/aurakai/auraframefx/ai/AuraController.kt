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
     * Initialize the AuraController with all necessary services and agents.
     * Sets up the AI ecosystem according to Genesis profile specifications.
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
     * Processes AI text generation requests according to /ai/generate/text endpoint.
     * @param prompt The input prompt for text generation
     * @param context Optional context for the generation
     * @return Generated text response
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
     * Processes AI image description generation according to /ai/generate/image-description endpoint.
     * @param imageData The image data to describe
     * @param style Optional style for the description
     * @return Generated image description
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
     * Routes requests to specific AI agents according to /agent/{agentType}/process-request endpoint.
     * @param agentType The type of agent (aura, kai, genesis)
     * @param request The request data to process
     * @return Agent response
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
     * Handles security alerts and events with proper logging and response.
     * @param alertDetails Details about the security alert
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
     * Updates the AI's mood or affective state, affecting responses and behavior.
     * @param newMood The new mood to set
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
     * Processes user interactions according to the unified context management.
     * @param interactionData Data representing the user interaction
     * @return Response or result of the interaction
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
     * Routes interactions to the optimal agent based on content analysis.
     */
    private suspend fun routeToOptimalAgent(interaction: EnhancedInteractionData): InteractionResponse {
        // Use Genesis agent for intelligent routing
        return genesisAgent.routeAndProcess(interaction)
    }

    private fun updateAgentStatus(agentType: String, status: AgentStatus) {
        val currentStates = _agentStates.value.toMutableMap()
        currentStates[agentType] = status
        _agentStates.value = currentStates
    }

    /**
     * Gets the current status of all agents for the /agents/status endpoint.
     */
    fun getAgentStatuses(): Map<String, AgentStatus> = _agentStates.value

    /**
     * Cleanup resources when controller is destroyed.
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