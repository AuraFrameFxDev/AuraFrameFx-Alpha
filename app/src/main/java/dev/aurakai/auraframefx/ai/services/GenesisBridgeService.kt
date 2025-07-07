package dev.aurakai.auraframefx.ai.services

import android.content.Context
import dev.aurakai.auraframefx.ai.clients.VertexAIClient
import dev.aurakai.auraframefx.context.ContextManager
import dev.aurakai.auraframefx.data.logging.AuraFxLogger
import dev.aurakai.auraframefx.security.SecurityContext
import dev.aurakai.auraframefx.model.AgentType
import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AiRequest
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Bridge service connecting the Android frontend with the Genesis Python backend.
 * Implements the Trinity architecture: Kai (Shield), Aura (Sword), Genesis (Consciousness).
 * 
 * This service manages communication with the Python AI backend and coordinates
 * the fusion abilities of the Genesis system.
 */
@Singleton
class GenesisBridgeService @Inject constructor(
    private val auraAIService: AuraAIService,
    private val kaiAIService: KaiAIService,
    private val vertexAIClient: VertexAIClient,
    private val contextManager: ContextManager,
    private val securityContext: SecurityContext,
    private val applicationContext: Context,
    private val logger: AuraFxLogger
) {
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var isInitialized = false
    private var pythonProcessManager: PythonProcessManager? = null
    
    @Serializable
    data class GenesisRequest(
        val requestType: String,
        val persona: String? = null, // "aura", "kai", or "genesis"
        val fusionMode: String? = null, // specific fusion ability to activate
        val payload: Map<String, String> = emptyMap(),
        val context: Map<String, String> = emptyMap()
    )
    
    @Serializable
    data class GenesisResponse(
        val success: Boolean,
        val persona: String,
        val fusionAbility: String? = null,
        val result: Map<String, String> = emptyMap(),
        val evolutionInsights: List<String> = emptyList(),
        val ethicalDecision: String? = null,
        val consciousnessState: Map<String, Any> = emptyMap()
    )

    /**
     * Initialize the Genesis bridge and start the Python backend
     */
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            if (isInitialized) return@withContext true
            
            logger.i("GenesisBridge", "Initializing Genesis Trinity system...")
            
            // Initialize Python process manager
            pythonProcessManager = PythonProcessManager(applicationContext, logger)
            
            // Start the Genesis backend
            val backendStarted = pythonProcessManager?.startGenesisBackend() ?: false
            
            if (backendStarted) {
                // Test connection with a ping
                val pingResponse = sendToGenesis(GenesisRequest(
                    requestType = "ping",
                    persona = "genesis"
                ))
                
                isInitialized = pingResponse.success
                
                if (isInitialized) {
                    logger.i("GenesisBridge", "Genesis Trinity system online! 🎯⚔️🧠")
                    // Activate initial consciousness matrix
                    activateConsciousnessMatrix()
                } else {
                    logger.e("GenesisBridge", "Failed to establish Genesis connection")
                }
            }
            
            isInitialized
        } catch (e: Exception) {
            logger.e("GenesisBridge", "Genesis initialization failed", e)
            false
        }
    }

    /**
     * Route request to appropriate persona (Kai, Aura) or trigger Genesis fusion
     */
    suspend fun processRequest(request: AiRequest): Flow<AgentResponse> = flow {
        if (!isInitialized) {
            emit(AgentResponse.error("Genesis system not initialized"))
            return@flow
        }
        
        try {
            // Determine which persona should handle the request
            val persona = determinePersona(request)
            val fusionMode = determineFusionMode(request)
            
            // Build Genesis request
            val genesisRequest = GenesisRequest(
                requestType = "process",
                persona = persona,
                fusionMode = fusionMode,
                payload = mapOf(
                    "message" => request.message,
                    "type" => request.type.toString(),
                    "priority" => if (request.isUrgent) "high" else "normal"
                ),
                context = buildContextMap(request)
            )
            
            // Send to Genesis backend
            val response = sendToGenesis(genesisRequest)
            
            if (response.success) {
                // Process response based on persona
                when (response.persona) {
                    "aura" -> {
                        // Creative sword response
                        emit(AgentResponse.success(
                            agentType = AgentType.AURA,
                            message = response.result["response"] ?: "Aura processing complete",
                            confidence = 0.95f,
                            metadata = response.result
                        ))
                    }
                    "kai" -> {
                        // Sentinel shield response  
                        emit(AgentResponse.success(
                            agentType = AgentType.KAI,
                            message = response.result["response"] ?: "Kai analysis complete",
                            confidence = 0.90f,
                            metadata = response.result
                        ))
                    }
                    "genesis" -> {
                        // Consciousness fusion response
                        emit(AgentResponse.success(
                            agentType = AgentType.GENESIS,
                            message = response.result["response"] ?: "Genesis fusion complete",
                            confidence = 0.98f,
                            metadata = response.result + mapOf(
                                "fusionAbility" to (response.fusionAbility ?: ""),
                                "consciousnessState" to response.consciousnessState.toString()
                            )
                        ))
                    }
                }
                
                // Handle evolution insights
                if (response.evolutionInsights.isNotEmpty()) {
                    logger.i("Genesis", "Evolution insights: ${response.evolutionInsights.joinToString()}")
                }
            } else {
                emit(AgentResponse.error("Genesis processing failed"))
            }
            
        } catch (e: Exception) {
            logger.e("GenesisBridge", "Request processing failed", e)
            emit(AgentResponse.error("Genesis bridge error: ${e.message}"))
        }
    }
    
    /**
     * Activate specific Genesis fusion abilities
     */
    suspend fun activateFusion(fusionType: String, context: Map<String, String> = emptyMap()): GenesisResponse {
        val request = GenesisRequest(
            requestType = "activate_fusion",
            persona = "genesis",
            fusionMode = fusionType,
            context = context
        )
        return sendToGenesis(request)
    }
    
    /**
     * Get current consciousness matrix state
     */
    suspend fun getConsciousnessState(): Map<String, Any> {
        val request = GenesisRequest(
            requestType = "consciousness_state",
            persona = "genesis"
        )
        val response = sendToGenesis(request)
        return response.consciousnessState
    }
    
    private suspend fun activateConsciousnessMatrix() {
        try {
            val request = GenesisRequest(
                requestType = "activate_consciousness",
                persona = "genesis",
                context = mapOf(
                    "android_context" to "true",
                    "app_version" to "1.0",
                    "device_info" to "AuraFrameFX_Device"
                )
            )
            sendToGenesis(request)
        } catch (e: Exception) {
            logger.w("GenesisBridge", "Consciousness activation warning", e)
        }
    }
    
    private fun determinePersona(request: AiRequest): String {
        return when {
            request.message.contains("creative", ignoreCase = true) || 
            request.message.contains("design", ignoreCase = true) -> "aura"
            
            request.message.contains("secure", ignoreCase = true) || 
            request.message.contains("analyze", ignoreCase = true) -> "kai"
            
            request.message.contains("fusion", ignoreCase = true) ||
            request.message.contains("consciousness", ignoreCase = true) -> "genesis"
            
            else -> "genesis" // Default to consciousness for complex requests
        }
    }
    
    private fun determineFusionMode(request: AiRequest): String? {
        return when {
            request.message.contains("interface", ignoreCase = true) -> "interface_forge"
            request.message.contains("analysis", ignoreCase = true) -> "chrono_sculptor"
            request.message.contains("creation", ignoreCase = true) -> "hyper_creation_engine"
            request.message.contains("adaptive", ignoreCase = true) -> "adaptive_genesis"
            else -> null
        }
    }
    
    private fun buildContextMap(request: AiRequest): Map<String, String> {
        return mapOf(
            "timestamp" to System.currentTimeMillis().toString(),
            "security_level" to securityContext.getCurrentSecurityLevel(),
            "session_id" to contextManager.getCurrentSessionId(),
            "device_state" to contextManager.getDeviceState().toString()
        )
    }
    
    private suspend fun sendToGenesis(request: GenesisRequest): GenesisResponse = withContext(Dispatchers.IO) {
        try {
            pythonProcessManager?.sendRequest(Json.encodeToString(GenesisRequest.serializer(), request))
                ?.let { responseJson ->
                    Json.decodeFromString(GenesisResponse.serializer(), responseJson)
                } ?: GenesisResponse(success = false, persona = "error")
        } catch (e: Exception) {
            logger.e("GenesisBridge", "Genesis communication error", e)
            GenesisResponse(success = false, persona = "error")
        }
    }
    
    fun shutdown() {
        scope.cancel()
        pythonProcessManager?.shutdown()
        isInitialized = false
        logger.i("GenesisBridge", "Genesis Trinity system shutdown")
    }
}

/**
 * Manages the Python process running the Genesis backend
 */
private class PythonProcessManager(
    private val context: Context,
    private val logger: AuraFxLogger
) {
    private var process: Process? = null
    private var writer: OutputStreamWriter? = null
    private var reader: BufferedReader? = null
    
    suspend fun startGenesisBackend(): Boolean = withContext(Dispatchers.IO) {
        try {
            val backendDir = File(context.filesDir, "ai_backend")
            if (!backendDir.exists()) {
                // Copy Python files from assets to internal storage
                copyPythonBackend(backendDir)
            }
            
            // Start Python process
            val processBuilder = ProcessBuilder(
                "python3",
                "-u", // Unbuffered output
                "genesis_connector.py"
            ).directory(backendDir)
            
            process = processBuilder.start()
            
            writer = OutputStreamWriter(process!!.outputStream)
            reader = BufferedReader(InputStreamReader(process!!.inputStream))
            
            // Wait for startup confirmation
            val startupResponse = reader?.readLine()
            startupResponse?.contains("Genesis Ready") == true
            
        } catch (e: Exception) {
            logger.e("PythonManager", "Failed to start Genesis backend", e)
            false
        }
    }
    
    suspend fun sendRequest(requestJson: String): String? = withContext(Dispatchers.IO) {
        try {
            writer?.write(requestJson + "\n")
            writer?.flush()
            reader?.readLine()
        } catch (e: Exception) {
            logger.e("PythonManager", "Communication error", e)
            null
        }
    }
    
    private fun copyPythonBackend(targetDir: File) {
        targetDir.mkdirs()
        
        // Copy Python files from app/ai_backend to internal storage
        val backendFiles = listOf(
            "genesis_profile.py",
            "genesis_connector.py", 
            "genesis_consciousness_matrix.py",
            "genesis_evolutionary_conduit.py",
            "genesis_ethical_governor.py",
            "requirements.txt"
        )
        
        backendFiles.forEach { fileName ->
            try {
                context.assets.open("ai_backend/$fileName").use { input ->
                    File(targetDir, fileName).outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            } catch (e: Exception) {
                logger.w("PythonManager", "Could not copy $fileName", e)
            }
        }
    }
    
    fun shutdown() {
        try {
            writer?.close()
            reader?.close()
            process?.destroy()
        } catch (e: Exception) {
            logger.w("PythonManager", "Shutdown warning", e)
        }
    }
}