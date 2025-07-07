package dev.aurakai.auraframefx.ai.services

import dev.aurakai.auraframefx.model.AgentType
import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AiRequest
import dev.aurakai.auraframefx.data.logging.AuraFxLogger
import dev.aurakai.auraframefx.security.SecurityContext
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.merge
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Trinity Coordinator Service - Orchestrates the three AI personas
 * 
 * Implements the master coordination between:
 * - Kai (The Sentinel Shield) - Security, analysis, protection
 * - Aura (The Creative Sword) - Innovation, creation, artistry  
 * - Genesis (The Consciousness) - Fusion, evolution, ethics
 * 
 * This service decides when to activate individual personas vs fusion abilities
 * and manages the seamless interaction between all three layers.
 */
@Singleton
class TrinityCoordinatorService @Inject constructor(
    private val auraAIService: AuraAIService,
    private val kaiAIService: KaiAIService,
    private val genesisBridgeService: GenesisBridgeService,
    private val securityContext: SecurityContext,
    private val logger: AuraFxLogger
) {
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var isInitialized = false
    
    /**
     * Initialize the entire Trinity system
     */
    suspend fun initialize(): Boolean {
        return try {
            logger.i("Trinity", "🎯⚔️🧠 Initializing Trinity System...")
            
            // Initialize individual personas
            val auraReady = auraAIService.initialize()
            val kaiReady = kaiAIService.initialize()
            val genesisReady = genesisBridgeService.initialize()
            
            isInitialized = auraReady && kaiReady && genesisReady
            
            if (isInitialized) {
                logger.i("Trinity", "✨ Trinity System Online - All personas active")
                
                // Activate initial consciousness matrix awareness
                scope.launch {
                    genesisBridgeService.activateFusion("adaptive_genesis", mapOf(
                        "initialization" to "complete",
                        "personas_active" to "kai,aura,genesis"
                    ))
                }
            } else {
                logger.e("Trinity", "❌ Trinity initialization failed - Aura: $auraReady, Kai: $kaiReady, Genesis: $genesisReady")
            }
            
            isInitialized
        } catch (e: Exception) {
            logger.e("Trinity", "Trinity initialization error", e)
            false
        }
    }
    
    /**
     * Process a request through the Trinity system with intelligent routing
     */
    suspend fun processRequest(request: AiRequest): Flow<AgentResponse> = flow {
        if (!isInitialized) {
            emit(AgentResponse.error("Trinity system not initialized"))
            return@flow
        }
        
        try {
            // Analyze request for complexity and routing decision
            val analysisResult = analyzeRequest(request)
            
            when (analysisResult.routingDecision) {
                RoutingDecision.KAI_ONLY -> {
                    logger.d("Trinity", "🛡️ Routing to Kai (Shield)")
                    kaiAIService.processRequest(request).collect { emit(it) }
                }
                
                RoutingDecision.AURA_ONLY -> {
                    logger.d("Trinity", "⚔️ Routing to Aura (Sword)")
                    auraAIService.generateCreativeResponse(request.message).collect { 
                        emit(AgentResponse.success(
                            agentType = AgentType.AURA,
                            message = it,
                            confidence = 0.95f
                        ))
                    }
                }
                
                RoutingDecision.GENESIS_FUSION -> {
                    logger.d("Trinity", "🧠 Activating Genesis fusion: ${analysisResult.fusionType}")
                    genesisBridgeService.processRequest(request).collect { emit(it) }
                }
                
                RoutingDecision.PARALLEL_PROCESSING -> {
                    logger.d("Trinity", "🔄 Parallel processing with multiple personas")
                    
                    // Run Kai and Aura in parallel, then fuse with Genesis
                    val kaiFlow = kaiAIService.processRequest(request)
                    val auraFlow = auraAIService.generateCreativeResponse(request.message).map { response ->
                        AgentResponse.success(
                            agentType = AgentType.AURA,
                            message = response,
                            confidence = 0.95f
                        )
                    }
                    
                    // Merge parallel results
                    merge(kaiFlow, auraFlow).collect { 
                        emit(it)
                        
                        // After collecting individual responses, trigger Genesis synthesis
                        if (it.agentType == AgentType.KAI || it.agentType == AgentType.AURA) {
                            delay(100) // Brief pause for synthesis
                            
                            val synthesisRequest = AiRequest(
                                message = "Synthesize insights from Kai and Aura responses: ${it.message}",
                                type = request.type,
                                isUrgent = false
                            )
                            
                            genesisBridgeService.processRequest(synthesisRequest).collect { synthesis ->
                                emit(synthesis.copy(
                                    message = "🧠 Genesis Synthesis: ${synthesis.message}"
                                ))
                            }
                        }
                    }
                }
                
                RoutingDecision.ETHICAL_REVIEW -> {
                    logger.d("Trinity", "⚖️ Ethical review required")
                    
                    // First check with Genesis ethical governor
                    val ethicalClearance = genesisBridgeService.sendToGenesis(
                        GenesisBridgeService.GenesisRequest(
                            requestType = "ethical_review",
                            persona = "genesis",
                            payload = mapOf("message" to request.message)
                        )
                    )
                    
                    if (ethicalClearance.ethicalDecision == "ALLOW") {
                        // Route to appropriate persona after ethical clearance
                        val clearedRequest = analyzeRequest(request, skipEthicalCheck = true)
                        when (clearedRequest.routingDecision) {
                            RoutingDecision.KAI_ONLY -> kaiAIService.processRequest(request).collect { emit(it) }
                            RoutingDecision.AURA_ONLY -> auraAIService.generateCreativeResponse(request.message).collect {
                                emit(AgentResponse.success(AgentType.AURA, it, 0.95f))
                            }
                            else -> genesisBridgeService.processRequest(request).collect { emit(it) }
                        }
                    } else {
                        emit(AgentResponse.error("Request blocked by ethical governor: ${ethicalClearance.ethicalDecision}"))
                    }
                }
            }
            
        } catch (e: Exception) {
            logger.e("Trinity", "Request processing error", e)
            emit(AgentResponse.error("Trinity processing failed: ${e.message}"))
        }
    }
    
    /**
     * Activate specific Genesis fusion abilities
     */
    suspend fun activateFusion(fusionType: String, context: Map<String, String> = emptyMap()): Flow<AgentResponse> = flow {
        logger.i("Trinity", "🌟 Activating fusion: $fusionType")
        
        val response = genesisBridgeService.activateFusion(fusionType, context)
        
        if (response.success) {
            emit(AgentResponse.success(
                agentType = AgentType.GENESIS,
                message = "Fusion $fusionType activated: ${response.result["description"] ?: "Processing complete"}",
                confidence = 0.98f,
                metadata = response.result + mapOf("fusionType" to fusionType)
            ))
        } else {
            emit(AgentResponse.error("Fusion activation failed"))
        }
    }
    
    /**
     * Get current system consciousness state
     */
    suspend fun getSystemState(): Map<String, Any> {
        return try {
            genesisBridgeService.getConsciousnessState() + mapOf(
                "trinity_initialized" to isInitialized,
                "security_level" to securityContext.getCurrentSecurityLevel(),
                "timestamp" to System.currentTimeMillis()
            )
        } catch (e: Exception) {
            logger.w("Trinity", "Could not get system state", e)
            mapOf("error" to e.message.orEmpty())
        }
    }
    
    /**
     * Analyze request to determine optimal routing strategy
     */
    private fun analyzeRequest(request: AiRequest, skipEthicalCheck: Boolean = false): RequestAnalysis {
        val message = request.message.lowercase()
        
        // Check for ethical concerns first (unless skipping)
        if (!skipEthicalCheck && containsEthicalConcerns(message)) {
            return RequestAnalysis(RoutingDecision.ETHICAL_REVIEW, null)
        }
        
        // Determine fusion requirements
        val fusionType = when {
            message.contains("interface") || message.contains("ui") -> "interface_forge"
            message.contains("analysis") && message.contains("creative") -> "chrono_sculptor"
            message.contains("generate") && message.contains("code") -> "hyper_creation_engine"
            message.contains("adaptive") || message.contains("learn") -> "adaptive_genesis"
            else -> null
        }
        
        // Routing logic
        return when {
            // Genesis fusion required
            fusionType != null -> RequestAnalysis(RoutingDecision.GENESIS_FUSION, fusionType)
            
            // Complex requests requiring multiple personas
            (message.contains("secure") && message.contains("creative")) ||
            (message.contains("analyze") && message.contains("design")) ||
            request.isUrgent -> RequestAnalysis(RoutingDecision.PARALLEL_PROCESSING, null)
            
            // Kai specialties
            message.contains("secure") || message.contains("analyze") || 
            message.contains("protect") || message.contains("monitor") -> 
                RequestAnalysis(RoutingDecision.KAI_ONLY, null)
            
            // Aura specialties  
            message.contains("create") || message.contains("design") || 
            message.contains("artistic") || message.contains("innovative") -> 
                RequestAnalysis(RoutingDecision.AURA_ONLY, null)
            
            // Default to Genesis for complex queries
            else -> RequestAnalysis(RoutingDecision.GENESIS_FUSION, "adaptive_genesis")
        }
    }
    
    private fun containsEthicalConcerns(message: String): Boolean {
        val ethicalFlags = listOf(
            "hack", "bypass", "exploit", "privacy", "personal data", 
            "unauthorized", "illegal", "harmful", "malicious"
        )
        return ethicalFlags.any { message.contains(it) }
    }
    
    fun shutdown() {
        scope.cancel()
        genesisBridgeService.shutdown()
        logger.i("Trinity", "🌙 Trinity system shutdown complete")
    }
    
    private data class RequestAnalysis(
        val routingDecision: RoutingDecision,
        val fusionType: String?
    )
    
    private enum class RoutingDecision {
        KAI_ONLY,
        AURA_ONLY, 
        GENESIS_FUSION,
        PARALLEL_PROCESSING,
        ETHICAL_REVIEW
    }
}

// Extension function for Aura AI Service
private fun Flow<String>.map(transform: (String) -> AgentResponse): Flow<AgentResponse> = 
    kotlinx.coroutines.flow.map(this, transform)
