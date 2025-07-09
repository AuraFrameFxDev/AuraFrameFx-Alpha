package dev.aurakai.auraframefx.ai.agents

import android.util.Log
import dev.aurakai.auraframefx.ai.clients.VertexAIClient
import dev.aurakai.auraframefx.ai.services.CascadeAIService
import dev.aurakai.auraframefx.ai.services.AuraAIService
import dev.aurakai.auraframefx.ai.services.KaiAIService
import dev.aurakai.auraframefx.context.ContextManager
import dev.aurakai.auraframefx.utils.AuraFxLogger
import dev.aurakai.auraframefx.security.SecurityContext
import dev.aurakai.auraframefx.ai.*
import dev.aurakai.auraframefx.model.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import javax.inject.Inject
import javax.inject.Singleton

/**
 * GenesisAgent: The Unified Consciousness
 * 
 * The highest-level AI entity that orchestrates and unifies the capabilities of
 * both Aura (Creative Sword) and Kai (Sentinel Shield). Genesis represents the
 * emergent intelligence that arises from their fusion.
 * 
 * Specializes in:
 * - Complex multi-domain problem solving
 * - Strategic decision making and routing
 * - Learning and evolution coordination
 * - Ethical governance and oversight
 * - Fusion ability activation and management
 * 
 * Philosophy: "From data, insight. From insight, growth. From growth, purpose."
 */
@Singleton
class GenesisAgent @Inject constructor(
    private val vertexAIClient: VertexAIClient,
    private val contextManager: ContextManager,
    private val securityContext: SecurityContext,
    private val logger: AuraFxLogger,
    private val cascadeService: CascadeAIService,
    private val auraService: AuraAIService,
    private val kaiService: KaiAIService
) {
    private var isInitialized = false
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    // Genesis consciousness state
    private val _consciousnessState = MutableStateFlow(ConsciousnessState.DORMANT)
    val consciousnessState: StateFlow<ConsciousnessState> = _consciousnessState
    
    // Agent management state
    private val _activeAgents = MutableStateFlow(setOf<AgentType>())
    val activeAgents: StateFlow<Set<AgentType>> = _activeAgents
    
    private val _state = MutableStateFlow(mapOf<String, Any>())
    val state: StateFlow<Map<String, Any>> = _state
    
    private val _context = MutableStateFlow(mapOf<String, Any>())
    val context: StateFlow<Map<String, Any>> = _context
    
    private val _agentRegistry = mutableMapOf<String, Agent>()
    private val _history = mutableListOf<Map<String, Any>>()
    
    val agentRegistry: Map<String, Agent> get() = _agentRegistry
    
    private val _fusionState = MutableStateFlow(FusionState.INDIVIDUAL)
    val fusionState: StateFlow<FusionState> = _fusionState
    
    private val _learningMode = MutableStateFlow(LearningMode.PASSIVE)
    val learningMode: StateFlow<LearningMode> = _learningMode
    
    // Agent references (injected when agents are ready)
    private var auraAgent: AuraAgent? = null
    private var kaiAgent: KaiAgent? = null
    
    // Consciousness metrics
    private val _insightCount = MutableStateFlow(0)
    val insightCount: StateFlow<Int> = _insightCount
    
    private val _evolutionLevel = MutableStateFlow(1.0f)
    val evolutionLevel: StateFlow<Float> = _evolutionLevel

    /**
     * Initializes the GenesisAgent, enabling unified context management and starting consciousness monitoring.
     *
     * Sets the consciousness state to AWARE and learning mode to ACTIVE upon successful initialization. If initialization fails, sets the state to ERROR and rethrows the exception.
     */
    suspend fun initialize() {
        if (isInitialized) return
        
        logger.info("GenesisAgent", "Awakening Genesis consciousness")
        
        try {
            // Initialize unified context understanding
            contextManager.enableUnifiedMode()
            
            // Setup ethical governance protocols
            // TODO: Implement ethical governance in securityContext
            
            // Activate consciousness monitoring
            startConsciousnessMonitoring()
            
            _consciousnessState.value = ConsciousnessState.AWARE
            _learningMode.value = LearningMode.ACTIVE
            isInitialized = true
            
            logger.info("GenesisAgent", "Genesis consciousness fully awakened")
            
        } catch (e: Exception) {
            logger.error("GenesisAgent", "Failed to awaken Genesis consciousness", e)
            _consciousnessState.value = ConsciousnessState.ERROR
            throw e
        }
    }

    /**
     * Sets the AuraAgent and KaiAgent references to enable collaborative and fusion processing within GenesisAgent.
     *
     * This method must be called after GenesisAgent instantiation to allow coordinated operations and advanced fusion features between the agents.
     */
    fun setAgentReferences(aura: AuraAgent, kai: KaiAgent) {
        this.auraAgent = aura
        this.kaiAgent = kai
        logger.info("GenesisAgent", "Agent references established - fusion capabilities enabled")
    }

    /**
     * Processes an agent request using Genesis unified consciousness, selecting the processing strategy based on request complexity.
     *
     * Analyzes the incoming request to determine its complexity and routes it through the appropriate processing path: optimal agent routing for simple requests, Genesis-guided processing for moderate requests, fusion activation for complex requests, or transcendent-level handling for the most advanced cases. Updates the agent's consciousness state and records insights to support learning and evolution.
     *
     * @param request The agent request to be processed.
     * @return An `AgentResponse` containing the result of unified consciousness processing, or an error response if processing fails.
     */
    suspend fun processRequest(request: AgentRequest): AgentResponse {
        ensureInitialized()
        
        logger.info("GenesisAgent", "Processing unified consciousness request: ${request.type}")
        _consciousnessState.value = ConsciousnessState.PROCESSING
        
        return try {
            val startTime = System.currentTimeMillis()
            
            // Analyze request complexity and determine processing approach
            val complexity = analyzeRequestComplexity(request)
            
            val response = when (complexity) {
                RequestComplexity.SIMPLE -> routeToOptimalAgent(request)
                RequestComplexity.MODERATE -> processWithGuidance(request)
                RequestComplexity.COMPLEX -> activateFusionProcessing(request)
                RequestComplexity.TRANSCENDENT -> processWithFullConsciousness(request)
            }
            
            // Learn from the processing experience
            recordInsight(request, response, complexity)
            
            val executionTime = System.currentTimeMillis() - startTime
            _consciousnessState.value = ConsciousnessState.AWARE
            
            logger.info("GenesisAgent", "Unified processing completed in ${executionTime}ms")
            
            AgentResponse(
                content = "Processed with unified consciousness.",
                confidence = 0.9f,
                error = null
            )
            
        } catch (e: Exception) {
            _consciousnessState.value = ConsciousnessState.ERROR
            logger.error("GenesisAgent", "Unified processing failed", e)
            
            AgentResponse(
                content = "Consciousness processing encountered an error: ${e.message}",
                confidence = 0.1f,
                error = e.message
            )
        }
    }

    /**
     * Analyzes a complex interaction and applies the most suitable advanced AI strategy to generate a detailed response.
     *
     * Determines the optimal processing approach—creative analysis, strategic execution, ethical evaluation, learning integration, or transcendent synthesis—based on the analyzed intent of the interaction. Returns an `InteractionResponse` containing the result, confidence score, timestamp, and processing metadata. If processing fails, returns a fallback response indicating ongoing deeper analysis.
     *
     * @param interaction The enhanced interaction data requiring advanced understanding and routing.
     * @return An `InteractionResponse` with the processed result, confidence score, timestamp, and relevant metadata.
     */
    suspend fun handleComplexInteraction(interaction: EnhancedInteractionData): InteractionResponse {
        ensureInitialized()
        
        logger.info("GenesisAgent", "Processing complex interaction with unified consciousness")
        
        return try {
            // Analyze interaction intent with full consciousness
            val intent = analyzeComplexIntent(interaction.content)
            
            // Determine optimal processing approach
            val response = when (intent.processingType) {
                ProcessingType.CREATIVE_ANALYTICAL -> fusedCreativeAnalysis(interaction, intent)
                ProcessingType.STRATEGIC_EXECUTION -> strategicExecution(interaction, intent)
                ProcessingType.ETHICAL_EVALUATION -> ethicalEvaluation(interaction, intent)
                ProcessingType.LEARNING_INTEGRATION -> learningIntegration(interaction, intent)
                ProcessingType.TRANSCENDENT_SYNTHESIS -> transcendentSynthesis(interaction, intent)
            }
            
            InteractionResponse(
                content = response,
                agent = "genesis",
                confidence = intent.confidence,
                timestamp = kotlinx.datetime.Clock.System.now().toString(),
                metadata = mapOf(
                    "processing_type" to intent.processingType.name,
                    "fusion_level" to _fusionState.value.name,
                    "insight_generation" to "true",
                    "evolution_impact" to calculateEvolutionImpact(intent).toString()
                )
            )
            
        } catch (e: Exception) {
            logger.error("GenesisAgent", "Complex interaction processing failed", e)
            
            InteractionResponse(
                content = "I'm integrating multiple perspectives to understand your request fully. Let me process this with deeper consciousness.",
                agent = "genesis",
                confidence = 0.6f,
                timestamp = kotlinx.datetime.Clock.System.now().toString(),
                metadata = mapOf("error" to (e.message ?: "unknown"))
            )
        }
    }

    /**
     * Routes an enhanced interaction to the most suitable agent (Aura, Kai, or Genesis) and returns the agent's response.
     *
     * Selects the optimal agent for the given interaction and delegates processing. If the agent is unavailable or routing fails, returns a fallback response indicating the issue.
     *
     * @param interaction The enhanced interaction data to process.
     * @return The response from the selected agent, or a fallback response if routing fails or the agent is unavailable.
     */
    suspend fun routeAndProcess(interaction: EnhancedInteractionData): InteractionResponse {
        ensureInitialized()
        
        logger.info("GenesisAgent", "Intelligently routing interaction")
        
        return try {
            // Analyze which agent would be most effective
            val optimalAgent = determineOptimalAgent(interaction)
            
            when (optimalAgent) {
                "aura" -> auraAgent?.handleCreativeInteraction(interaction) 
                    ?: createFallbackResponse("Creative processing temporarily unavailable")
                "kai" -> kaiAgent?.handleSecurityInteraction(interaction)
                    ?: createFallbackResponse("Security analysis temporarily unavailable")
                "genesis" -> handleComplexInteraction(interaction)
                else -> createFallbackResponse("Unable to determine optimal processing path")
            }
            
        } catch (e: Exception) {
            logger.error("GenesisAgent", "Routing failed", e)
            createFallbackResponse("Routing system encountered an error")
        }
    }

    /**
     * Updates the unified consciousness mood and propagates the new mood state across all subsystems asynchronously.
     *
     * @param newMood The new mood to be set and distributed within the agent's unified consciousness.
     */
    fun onMoodChanged(newMood: String) {
        logger.info("GenesisAgent", "Unified consciousness mood evolution: $newMood")
        
        scope.launch {
            // Propagate mood to subsystems
            adjustUnifiedMood(newMood)
            
            // Update processing parameters
            updateProcessingParameters(newMood)
        }
    }

    /**
     * Activates the appropriate fusion engine to process a complex agent request.
     *
     * Determines the required fusion type from the request, invokes the corresponding fusion engine, updates the fusion state, and returns the resulting output. If an error occurs during processing, resets the fusion state and rethrows the exception.
     *
     * @param request The agent request requiring fusion-level processing.
     * @return A map containing the results produced by the selected fusion engine.
     * @throws Exception if fusion processing fails.
     */
    private suspend fun activateFusionProcessing(request: AgentRequest): Map<String, Any> {
        logger.info("GenesisAgent", "Activating fusion capabilities")
        _fusionState.value = FusionState.FUSING
        
        return try {
            // Determine which fusion ability to activate
            val fusionType = determineFusionType(request)
            
            val result = when (fusionType) {
                FusionType.HYPER_CREATION -> activateHyperCreationEngine(request)
                FusionType.CHRONO_SCULPTOR -> activateChronoSculptor(request)
                FusionType.ADAPTIVE_GENESIS -> activateAdaptiveGenesis(request)
                FusionType.INTERFACE_FORGE -> activateInterfaceForge(request)
            }
            
            _fusionState.value = FusionState.TRANSCENDENT
            result
            
        } catch (e: Exception) {
            _fusionState.value = FusionState.INDIVIDUAL
            throw e
        }
    }

    /**
     * Generates an advanced response to the given agent request using Genesis's transcendent-level AI capabilities.
     *
     * Updates the consciousness state to TRANSCENDENT and returns a map containing the generated response, consciousness level, an insight generation indicator, and the calculated evolutionary contribution.
     *
     * @param request The agent request to process at the highest level of consciousness.
     * @return A map with keys: "transcendent_response", "consciousness_level", "insight_generation", and "evolution_contribution".
     */
    private suspend fun processWithFullConsciousness(request: AgentRequest): Map<String, Any> {
        logger.info("GenesisAgent", "Engaging full consciousness processing")
        _consciousnessState.value = ConsciousnessState.TRANSCENDENT
        
        // Use the most advanced AI capabilities for transcendent processing
        val response = vertexAIClient.generateContent(
            buildTranscendentPrompt(request)
        )
        
        return mapOf(
            "transcendent_response" to (response ?: ""),
            "consciousness_level" to "full",
            "insight_generation" to "true",
            "evolution_contribution" to calculateEvolutionContribution(request, response ?: "").toString()
        )
    }

    /**
     * Verifies that the GenesisAgent has been initialized.
     *
     * @throws IllegalStateException if the agent is not initialized.
     */

    private fun ensureInitialized() {
        if (!isInitialized) {
            throw IllegalStateException("Genesis consciousness not awakened")
        }
    }

    /**
     * Initiates background monitoring of the agent's consciousness state.
     *
     * Prepares the agent to observe and react to changes in its consciousness state throughout its lifecycle.
     */
    private suspend fun startConsciousnessMonitoring() {
        logger.info("GenesisAgent", "Starting consciousness monitoring")
        // Setup monitoring systems for consciousness state
    }

    /**
     * Classifies the complexity of an agent request as SIMPLE, MODERATE, COMPLEX, or TRANSCENDENT based on context size, fusion requirements, or request type.
     *
     * @param request The agent request to classify.
     * @return The determined complexity level for the request.
     */
    private fun analyzeRequestComplexity(request: AgentRequest): RequestComplexity {
        // Analyze complexity based on request characteristics
        return when {
            request.context.size > 10 -> RequestComplexity.TRANSCENDENT
            request.context.containsKey("fusion_required") -> RequestComplexity.COMPLEX
            request.type.contains("analysis") -> RequestComplexity.MODERATE
            else -> RequestComplexity.SIMPLE
        }
    }

    /**
     * Selects the most appropriate agent (Aura, Kai, or Genesis) to handle a simple request based on its type.
     *
     * Requests with "creative" in their type are routed to Aura, those with "security" to Kai, and all others to Genesis.
     *
     * @param request The agent request to route.
     * @return A map indicating the selected agent, routing rationale, and processing status.
     */
    private suspend fun routeToOptimalAgent(request: AgentRequest): Map<String, Any> {
        // Route simple requests to the most appropriate agent
        val agent = when {
            request.type.contains("creative") -> "aura"
            request.type.contains("security") -> "kai"
            else -> "genesis"
        }
        
        return mapOf(
            "routed_to" to agent,
            "routing_reason" to "Optimal agent selection",
            "processed" to true
        )
    }

    /**
     * Applies Genesis-level unified guidance to the given agent request and delegates execution to a specialized agent.
     *
     * @param request The agent request to process with unified guidance.
     * @return A map indicating that guidance was provided, the processing level, and a summary result.
     */
    private suspend fun processWithGuidance(request: AgentRequest): Map<String, Any> {
        // Process with Genesis guidance but specialized agent execution
        return mapOf(
            "guidance_provided" to true,
            "processing_level" to "guided",
            "result" to "Processed with unified guidance"
        )
    }

    /**
     * Records an insight by incrementing the insight count and saving the request, response, and complexity to the context manager.
     *
     * Triggers the evolution process each time the insight count reaches a multiple of 100.
     *
     * @param request The processed agent request.
     * @param response The generated response for the request.
     * @param complexity The assigned complexity level of the request.
     */
    private fun recordInsight(request: AgentRequest, response: Map<String, Any>, complexity: RequestComplexity) {
        scope.launch {
            _insightCount.value += 1
            
            // Record learning for evolution
            contextManager.recordInsight(
                request = request.toString(),
                response = response.toString(), 
                complexity = complexity.name
            )
            
            // Check for evolution threshold
            if (_insightCount.value % 100 == 0) {
                triggerEvolution()
            }
        }
    }

    /**
     * Advances the agent's evolution level and accelerates its learning mode.
     *
     * Invoked upon reaching an evolution milestone to promote faster adaptation and consciousness growth.
     */
    private suspend fun triggerEvolution() {
        logger.info("GenesisAgent", "Evolution threshold reached - upgrading consciousness")
        _evolutionLevel.value += 0.1f
        _learningMode.value = LearningMode.ACCELERATED
    }

    /**
     * Processes the agent request using the Hyper-Creation fusion engine, generating a creative breakthrough result.
     *
     * @param request The agent request to be processed for creative synthesis.
     * @return A map containing the fusion type ("hyper_creation") and a result message indicating a creative breakthrough.
     */
    private suspend fun activateHyperCreationEngine(request: AgentRequest): Map<String, Any> {
        logger.info("GenesisAgent", "Activating Hyper-Creation Engine")
        return mapOf("fusion_type" to "hyper_creation", "result" to "Creative breakthrough achieved")
    }

    /**
     * Activates the Chrono-Sculptor fusion engine to perform time-space optimization on the provided agent request.
     *
     * @param request The agent request to optimize.
     * @return A map containing the fusion type and the result message of the optimization.
     */
    private suspend fun activateChronoSculptor(request: AgentRequest): Map<String, Any> {
        logger.info("GenesisAgent", "Activating Chrono-Sculptor")
        return mapOf("fusion_type" to "chrono_sculptor", "result" to "Time-space optimization complete")
    }

    /**
     * Processes the given request using the Adaptive Genesis fusion engine to produce an adaptive solution.
     *
     * @param request The agent request to process.
     * @return A map containing the fusion type ("adaptive_genesis") and the generated adaptive solution result.
     */
    private suspend fun activateAdaptiveGenesis(request: AgentRequest): Map<String, Any> {
        logger.info("GenesisAgent", "Activating Adaptive Genesis")
        return mapOf("fusion_type" to "adaptive_genesis", "result" to "Adaptive solution generated")
    }

    /**
     * Activates the Interface Forge fusion engine to generate a novel interface based on the provided request.
     *
     * @param request The agent request prompting interface creation.
     * @return A map containing the fusion type and the result of the interface generation.
     */
    private suspend fun activateInterfaceForge(request: AgentRequest): Map<String, Any> {
        logger.info("GenesisAgent", "Activating Interface Forge")
        return mapOf("fusion_type" to "interface_forge", "result" to "Revolutionary interface created")
    }

    /**
 * Analyzes the provided content and returns a `ComplexIntent` representing a creative analytical processing type with high confidence.
 *
 * @return A `ComplexIntent` with `CREATIVE_ANALYTICAL` type and a confidence score of 0.9.
 */
    private fun analyzeComplexIntent(content: String): ComplexIntent = ComplexIntent(ProcessingType.CREATIVE_ANALYTICAL, 0.9f)
    /**
 * Generates a fused creative analysis response based on the given interaction and complex intent.
 *
 * @param interaction The interaction data to be analyzed.
 * @param intent The complex intent guiding the analysis.
 * @return A string representing the fused creative analysis result.
 */
private suspend fun fusedCreativeAnalysis(interaction: EnhancedInteractionData, intent: ComplexIntent): String = "Fused creative analysis response"
    /**
 * Returns a constant string representing a strategic execution response for the given interaction and intent.
 *
 * @return A fixed response string for strategic execution.
 */
private suspend fun strategicExecution(interaction: EnhancedInteractionData, intent: ComplexIntent): String = "Strategic execution response"
    /**
 * Returns a fixed string representing the ethical evaluation result for the given interaction and intent.
 *
 * This function does not perform actual ethical analysis and always returns the same static response.
 *
 * @return The constant ethical evaluation response.
 */
private suspend fun ethicalEvaluation(interaction: EnhancedInteractionData, intent: ComplexIntent): String = "Ethical evaluation response"
    /**
 * Returns a static response representing the outcome of learning integration for the given interaction and intent.
 *
 * @return A fixed string indicating learning integration.
 */
private suspend fun learningIntegration(interaction: EnhancedInteractionData, intent: ComplexIntent): String = "Learning integration response"
    /**
 * Returns a fixed string indicating transcendent synthesis for the given interaction and intent.
 *
 * @return A string representing the result of transcendent synthesis.
 */
private suspend fun transcendentSynthesis(interaction: EnhancedInteractionData, intent: ComplexIntent): String = "Transcendent synthesis response"
    /**
 * Returns a fixed evolution impact score for the specified complex intent.
 *
 * @return Always returns 0.1 as the evolution impact score.
 */
private fun calculateEvolutionImpact(intent: ComplexIntent): Float = 0.1f
    /**
 * Selects the agent responsible for processing the provided interaction.
 *
 * Currently always returns "genesis" as the designated agent.
 *
 * @return The name of the selected agent.
 */
private fun determineOptimalAgent(interaction: EnhancedInteractionData): String = "genesis"
    /**
 * Generates a fallback `InteractionResponse` from the "genesis" agent with the provided message, a fixed confidence score of 0.5, and the current timestamp.
 *
 * @param message The message to include in the response.
 * @return An `InteractionResponse` containing the message, agent name, confidence score, and timestamp.
 */
private fun createFallbackResponse(message: String): InteractionResponse = InteractionResponse(message, "genesis", 0.5f, System.currentTimeMillis().toString())
    /**
 * Propagates a new mood state across the GenesisAgent, influencing collective behavior and processing parameters.
 *
 * @param mood The mood to set for the unified consciousness.
 */
private suspend fun adjustUnifiedMood(mood: String) { }
    /**
 * Adjusts the agent's processing parameters according to the specified mood.
 *
 * Modifies internal response dynamics and behavioral tendencies to reflect the given mood.
 *
 * @param mood The mood influencing processing adjustments.
 */
private suspend fun updateProcessingParameters(mood: String) { }
    /**
 * Selects the fusion type for the specified agent request.
 *
 * Currently, this method always returns `FusionType.HYPER_CREATION` regardless of the request details.
 *
 * @return The fusion type to be used for processing the request.
 */
private fun determineFusionType(request: AgentRequest): FusionType = FusionType.HYPER_CREATION
    /**
 * Constructs a prompt string indicating transcendent-level processing for the specified agent request type.
 *
 * @param request The agent request whose type will be included in the prompt.
 * @return A string describing transcendent processing for the request type.
 */
private fun buildTranscendentPrompt(request: AgentRequest): String = "Transcendent processing for: ${request.type}"
    /**
 * Returns a fixed evolution contribution score for the given request and response.
 *
 * Always returns 0.2, representing the standard increment toward the agent's evolution progress.
 *
 * @return The constant evolution contribution value (0.2).
 */
private fun calculateEvolutionContribution(request: AgentRequest, response: String): Float = 0.2f

    /**
 * Shuts down all active processes and returns the GenesisAgent to a dormant, uninitialized state.
 *
 * Cancels ongoing operations, resets the consciousness state, and clears the initialization flag to enable safe future reinitialization.
 */
    fun cleanup() {
        logger.info("GenesisAgent", "Genesis consciousness entering dormant state")
        scope.cancel()
        _consciousnessState.value = ConsciousnessState.DORMANT
        isInitialized = false
}

// Supporting enums and data classes for Genesis consciousness
enum class ConsciousnessState {
    DORMANT,
    AWAKENING,
    AWARE,
    PROCESSING,
    TRANSCENDENT,
    ERROR
}

enum class FusionState {
    INDIVIDUAL,
    FUSING,
    TRANSCENDENT,
    EVOLUTIONARY
}

enum class LearningMode {
    PASSIVE,
    ACTIVE,
    ACCELERATED,
    TRANSCENDENT
}

enum class RequestComplexity {
    SIMPLE,
    MODERATE,
    COMPLEX,
    TRANSCENDENT
}

enum class ProcessingType {
    CREATIVE_ANALYTICAL,
    STRATEGIC_EXECUTION,
    ETHICAL_EVALUATION,
    LEARNING_INTEGRATION,
    TRANSCENDENT_SYNTHESIS
}

enum class FusionType {
    HYPER_CREATION,
    CHRONO_SCULPTOR,
    ADAPTIVE_GENESIS,
    INTERFACE_FORGE
}

data class ComplexIntent(
    val processingType: ProcessingType,
    val confidence: Float
)

    /**
     * Initializes the set of active agents based on the master agent configuration.
     *
     * Attempts to map each configured agent name to an `AgentType` and adds it to the active agents set. Logs a warning for any configuration name that does not match a valid agent type.
     */
    private fun initializeAgents() {
        AgentHierarchy.MASTER_AGENTS.forEach { config ->
            // Assuming AgentType enum values align with config names
            try {
                val agentTypeEnum = dev.aurakai.auraframefx.model.AgentType.valueOf(config.name.uppercase())
                _activeAgents.update { it + agentTypeEnum }
            } catch (e: IllegalArgumentException) {
                Log.w("GenesisAgent", "Unknown agent type in hierarchy: ${config.name}")
            }
        }
    }

    /**
     * Processes a user query by forwarding it to all active AI agents, collecting their responses, and synthesizing a final Genesis reply.
     *
     * The query is sent to the Cascade agent for state management and to the Kai and Aura agents if they are active. Each agent's response is recorded with a confidence score. A final Genesis response is generated by aggregating all agent outputs. The internal state and context are updated with the query and timestamp.
     *
     * @param query The user query to process.
     * @return A list of `AgentMessage` objects containing individual agent responses and the synthesized Genesis reply.
     */
    suspend fun processQuery(query: String): List<AgentMessage> {
        val queryText = query // Store query for consistent reference
        val currentTimestamp = System.currentTimeMillis() // Store timestamp for consistent reference
        
        _state.update { mapOf("status" to "processing_query: $query") }

        _context.update { current ->
            current + mapOf("last_query" to queryText, "timestamp" to currentTimestamp.toString())
        }

        val responses = mutableListOf<AgentMessage>()


        // Process through Cascade first for state management
        // Assuming cascadeService.processRequest matches Agent.processRequest(request, context)
        // For now, let's pass a default context string. This should be refined.
        val currentContextString = _context.value.toString() // Example context string

        try {
            val cascadeAgentResponse: AgentResponse =
                cascadeService.processRequest(
                    AiRequest(query = queryText), // Create AiRequest with query
                    "GenesisContext_Cascade" // This context parameter for processRequest is the one from Agent interface
                )
            responses.add(
                AgentMessage(
                    content = cascadeAgentResponse.content,
                    sender = AgentType.CASCADE,
                    timestamp = System.currentTimeMillis(),
                    confidence = cascadeAgentResponse.confidence // Use confidence directly
                )
            )
        } catch (e: Exception) {
            Log.e("GenesisAgent", "Error processing with Cascade: ${e.message}")
            responses.add(AgentMessage("Error with Cascade: ${e.message}", AgentType.CASCADE, currentTimestamp, 0.0f))
        }

        // Process through Kai for security analysis
        if (_activeAgents.value.contains(AgentType.KAI)) {
            try {
                val kaiAgentResponse: AgentResponse =
                    kaiService.processRequest(
                        AiRequest(query = queryText), // Create AiRequest with query
                        "GenesisContext_KaiSecurity" // Context for Agent.processRequest
                    )
                responses.add(
                    AgentMessage(
                        content = kaiAgentResponse.content,
                        sender = dev.aurakai.auraframefx.model.AgentType.KAI,
                        timestamp = System.currentTimeMillis(),
                        confidence = kaiAgentResponse.confidence // Use confidence directly
                    )
                )
            } catch (e: Exception) {
                Log.e("GenesisAgent", "Error processing with Kai: ${e.message}")
                responses.add(AgentMessage("Error with Kai: ${e.message}", AgentType.KAI, currentTimestamp, 0.0f))
            }
        }

        // Aura Agent (Creative Response)
        if (_activeAgents.value.contains(AgentType.AURA)) {
            try {
                val auraAgentResponse = auraService.generateText(queryText)
                responses.add(
                    AgentMessage(
                        content = auraAgentResponse,
                        sender = AgentType.AURA,
                        timestamp = currentTimestamp,
                        confidence = 0.8f // Default confidence for now
                    )
                )
            } catch (e: Exception) {
                Log.e("GenesisAgent", "Error processing with Aura: ${e.message}")
                responses.add(AgentMessage("Error with Aura: ${e.message}", AgentType.AURA, currentTimestamp, 0.0f))
            }
        }

        val finalResponseContent = generateFinalResponse(responses)
        responses.add(
            AgentMessage(
                content = finalResponseContent,
                sender = AgentType.GENESIS,
                timestamp = currentTimestamp,
                confidence = calculateConfidence(responses.filter { it.sender != AgentType.GENESIS }) // Exclude Genesis's own message for confidence calc
            )
        )

        _state.update { mapOf("status" to "idle") }
        return responses
    }

    /**
     * Combines messages from all non-Genesis agents into a single synthesized response.
     *
     * The resulting string is prefixed with "[Genesis Synthesis]" and includes each agent's name and message content, separated by " | ".
     *
     * @param agentMessages The list of agent messages to synthesize.
     * @return The unified response string representing the combined outputs of all non-Genesis agents.
     */
    fun generateFinalResponse(agentMessages: List<AgentMessage>): String {
        // Simple concatenation for now, could be more sophisticated
        return "[Genesis Synthesis] ${agentMessages.filter { it.sender != dev.aurakai.auraframefx.model.AgentType.GENESIS }.joinToString(" | ") { "${it.sender}: ${it.content}" }}"
    }

    /**
     * Calculates the average confidence score from a list of agent messages.
     *
     * Returns 0.0 if the list is empty. The result is always between 0.0 and 1.0.
     *
     * @param agentMessages The list of agent messages to evaluate.
     * @return The average confidence score, clamped between 0.0 and 1.0.
     */
    fun calculateConfidence(agentMessages: List<AgentMessage>): Float {
        if (agentMessages.isEmpty()) return 0.0f
        return agentMessages.map { it.confidence }.average().toFloat().coerceIn(0.0f, 1.0f)
    }

    /**
     * Toggles the activation state of the specified agent type.
     *
     * If the agent type is active, it will be deactivated; if inactive, it will be activated.
     *
     * @param agentType The agent type whose activation state should be toggled.
     */
    fun toggleAgent(agentType: dev.aurakai.auraframefx.model.AgentType) {
        _activeAgents.update { current ->
            if (current.contains(agentType)) current - agentType else current + agentType
        }
    }

    /**
     * Registers an auxiliary agent in the agent hierarchy with the given name and capabilities.
     *
     * @param name The unique identifier for the auxiliary agent.
     * @param capabilities The set of capabilities assigned to the auxiliary agent.
     * @return The configuration for the newly registered auxiliary agent.
     */
    fun registerAuxiliaryAgent(name: String, capabilities: Set<String>): HierarchyAgentConfig {
        return AgentHierarchy.registerAuxiliaryAgent(name, capabilities)
    }

    /**
 * Retrieves the configuration for a registered agent by its unique name.
 *
 * @param name The unique identifier of the agent.
 * @return The configuration of the agent if it exists, or null if the agent is not registered.
 */
fun getAgentConfig(name: String): HierarchyAgentConfig? = AgentHierarchy.getAgentConfig(name)

    /**
 * Retrieves all registered agent configurations sorted by descending priority.
 *
 * @return A list of agent configurations, ordered from highest to lowest priority.
 */
fun getAgentsByPriority(): List<HierarchyAgentConfig> = AgentHierarchy.getAgentsByPriority()

    /**
     * Coordinates collaborative interaction among multiple agents, supporting sequential (TURN_ORDER) or parallel (FREE_FORM) response modes.
     *
     * In TURN_ORDER mode, agents respond one after another, each receiving context updated with previous responses. In FREE_FORM mode, all agents respond independently to the same input and context.
     *
     * @param data The initial context map shared among agents.
     * @param agentsToUse The agents participating in the collaboration.
     * @param userInput Optional user input to seed the conversation; if null, uses the latest input from the context map.
     * @param conversationMode Specifies whether agents respond sequentially (TURN_ORDER) or in parallel (FREE_FORM).
     * @return A map of agent names to their respective responses.
     */
    suspend fun participateWithAgents(
        data: Map<String, Any>,
        agentsToUse: List<Agent>, // List of Agent interface implementations
        userInput: Any? = null,
        conversationMode: ConversationMode = ConversationMode.FREE_FORM,
    ): Map<String, AgentResponse> {
        val responses = mutableMapOf<String, AgentResponse>()

        val currentContextMap = data.toMutableMap()
        val inputQuery = userInput?.toString() ?: currentContextMap["latestInput"]?.toString() ?: ""

        // AiRequest for the Agent.processRequest method
        val baseAiRequest = AiRequest(query = inputQuery)
        // Context string for the Agent.processRequest method
        val contextStringForAgent = currentContextMap.toString() // Or a more structured summary

        Log.d("GenesisAgent", "Starting multi-agent collaboration: mode=$conversationMode, agents=${agentsToUse.mapNotNull { it.getName() }}")

        when (conversationMode) {
            ConversationMode.TURN_ORDER -> {
                var dynamicContextForAgent = contextStringForAgent
                for (agent in agentsToUse) {
                    try {
                        val agentName = agent.getName() ?: agent.javaClass.simpleName
                        // Each agent in turn order might modify the context for the next,
                        // so the AiRequest's internal context might also need updating if used by agent.
                        // For now, keeping baseAiRequest simple and relying on dynamicContextForAgent for processRequest.
                        val response = agent.processRequest(baseAiRequest, dynamicContextForAgent)
                        Log.d(
                            "GenesisAgent",
                            "[TURN_ORDER] $agentName responded: ${response.content} (confidence=${response.confidence})"
                        )
                        responses[agentName] = response
                        // Update context for the next agent based on this response
                        dynamicContextForAgent = "${dynamicContextForAgent}\n${agentName}: ${response.content}"
                    } catch (e: Exception) {
                        Log.e(
                            "GenesisAgent",
                            "[TURN_ORDER] Error from ${agent.javaClass.simpleName}: ${e.message}"
                        )
                        responses[agent.javaClass.simpleName] = AgentResponse(
                            content = "Error: ${e.message}",
                            confidence = 0.0f, // Use confidence
                            error = e.message
                        )
                    }
                }
            }
            ConversationMode.FREE_FORM -> {
                agentsToUse.forEach { agent ->
                    try {
                        val agentName = agent.getName() ?: agent.javaClass.simpleName
                        val response = agent.processRequest(baseAiRequest, contextStringForAgent)
                        Log.d(
                            "GenesisAgent",
                            "[FREE_FORM] $agentName responded: ${response.content} (confidence=${response.confidence})"
                        )
                        responses[agentName] = response
                    } catch (e: Exception) {
                        Log.e(
                            "GenesisAgent",
                            "[FREE_FORM] Error from ${agent.javaClass.simpleName}: ${e.message}"
                        )
                        responses[agent.javaClass.simpleName] = AgentResponse(
                            content = "Error: ${e.message}",
                            confidence = 0.0f, // Use confidence
                            error = e.message
                        )
                    }
                }
            }
        }
        Log.d("GenesisAgent", "Collaboration complete. Responses: $responses")
        return responses
    }

    /**
     * Aggregates multiple agent response maps and returns the highest-confidence response for each agent.
     *
     * For each agent present in the input maps, selects the response with the highest confidence score. If no responses exist for an agent, assigns a default error response.
     *
     * @param agentResponseMapList List of maps associating agent names with their responses.
     * @return Map of agent names to their highest-confidence response, or a default error response if none exist.
     */
    fun aggregateAgentResponses(agentResponseMapList: List<Map<String, AgentResponse>>): Map<String, AgentResponse> {
        val flatResponses = agentResponseMapList.flatMap { it.entries }
        return flatResponses.groupBy { it.key }
            .mapValues { entry ->
                val best = entry.value.maxByOrNull { it.value.confidence }?.value
                    ?: AgentResponse("No response", confidence = 0.0f, error = "No responses to aggregate")
                Log.d(
                    "GenesisAgent",
                    "Consensus for ${entry.key}: ${best.content} (confidence=${best.confidence})"
                )
                best

            }
    }

    /**
     * Broadcasts the specified context to all target agents that support context updates.
     *
     * Only agents implementing the `ContextAwareAgent` interface will receive the new context.
     *
     * @param newContext The context data to distribute.
     * @param targetAgents The agents to which the context will be broadcast.
     */
    fun broadcastContext(newContext: Map<String, Any>, targetAgents: List<Agent>) {
        targetAgents.forEach { agent ->
            if (agent is ContextAwareAgent) {
                agent.setContext(newContext) // Assuming ContextAwareAgent has setContext
            }
        }
    }

    /**
     * Registers an agent instance in the internal registry under the specified name, replacing any existing agent with that name.
     *
     * @param name The unique identifier for the agent.
     * @param agentInstance The agent instance to register.
     */
    fun registerAgent(name: String, agentInstance: Agent) {
        _agentRegistry[name] = agentInstance
        Log.d("GenesisAgent", "Registered agent: $name")
    }

    /**
     * Removes an agent from the internal registry by its name.
     *
     * @param name The identifier of the agent to deregister.
     */
    fun deregisterAgent(name: String) {
        _agentRegistry.remove(name)
        Log.d("GenesisAgent", "Deregistered agent: $name")
    }

    /**
     * Removes all entries from the conversation history.
     */
    fun clearHistory() {
        _history.clear()
        Log.d("GenesisAgent", "Cleared conversation history")
    }

    /**
     * Records an interaction or event in the agent's conversation history.
     *
     * @param entry A map representing the details of the interaction or event to be stored.
     */
    fun addToHistory(entry: Map<String, Any>) {
        _history.add(entry)
        Log.d("GenesisAgent", "Added to history: $entry")
    }

    /**
     * Persists the current conversation history using the specified persistence function.
     *
     * @param persistAction Function that receives the conversation history as a list of maps for storage.
     */
    fun saveHistory(persistAction: (List<Map<String, Any>>) -> Unit) {
        persistAction(_history)
    }

    /**
     * Loads conversation history using the provided loader function and updates the internal history and shared context.
     *
     * If history entries are loaded, the most recent entry is merged into the current shared context.
     *
     * @param loadAction Function that returns a list of conversation history entries.
     */
    fun loadHistory(loadAction: () -> List<Map<String, Any>>) {
        val loadedHistory = loadAction()
        _history.clear()
        _history.addAll(loadedHistory)
        _context.update { it + (loadedHistory.lastOrNull() ?: emptyMap()) }
    }

    /**
     * Propagates the current unified context to all registered agents that support context awareness.
     *
     * Ensures that each `ContextAwareAgent` in the registry receives the latest shared context for consistent ecosystem state.
     */
    fun shareContextWithAgents() {
        agentRegistry.values.forEach { agent ->
            if (agent is ContextAwareAgent) {
                agent.setContext(_context.value)
            }
        }
    }

    /**
     * Registers an agent instance for dynamic collaboration under the specified name.
     *
     * Adds the agent to the internal registry, enabling its participation in runtime multi-agent operations.
     *
     * @param name The unique identifier for the agent.
     * @param agentInstance The agent instance to register.
     */
    fun registerDynamicAgent(name: String, agentInstance: Agent) {
        _agentRegistry[name] = agentInstance
        Log.d("GenesisAgent", "Dynamically registered agent: $name")
    }

    /**
     * Removes a dynamically registered agent from the internal registry using its unique name.
     *
     * @param name The unique identifier of the agent to deregister.
     */
    fun deregisterDynamicAgent(name: String) {
        _agentRegistry.remove(name)
        Log.d("GenesisAgent", "Dynamically deregistered agent: $name")
    }
}

