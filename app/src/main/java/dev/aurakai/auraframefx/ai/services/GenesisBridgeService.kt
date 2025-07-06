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
     * Starts the Genesis Python backend process and verifies connectivity.
     *
     * Launches the backend if not already initialized, checks communication with a ping request, and activates the initial consciousness matrix upon success.
     *
     * @return `true` if the backend is successfully initialized and ready; `false` otherwise.
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
     * Routes an AI request to the appropriate persona (Kai, Aura, or Genesis fusion) and emits the resulting agent response as a flow.
     *
     * Determines the target persona and fusion mode based on the request content, constructs a backend request, and emits a success or error response depending on the backend's result. Emits an error if the Genesis system is not initialized or if processing fails.
     *
     * @param request The AI request to process.
     * @return A flow emitting the corresponding agent response.
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
                    "message" to request.message,
                    "type" to request.type.toString(),
                    "priority" to if (request.isUrgent) "high" else "normal"
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
     * Activates a specified fusion ability in the Genesis backend using the given context.
     *
     * @param fusionType The fusion ability to activate.
     * @param context Additional parameters to provide context for the fusion activation.
     * @return The GenesisResponse returned by the backend after activation.
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
     * Retrieves the current consciousness matrix state from the Genesis backend.
     *
     * @return A map containing the current state of the consciousness matrix as provided by the backend.
     */
    suspend fun getConsciousnessState(): Map<String, Any> {
        val request = GenesisRequest(
            requestType = "consciousness_state",
            persona = "genesis"
        )
        val response = sendToGenesis(request)
        return response.consciousnessState
    }
    
    /**
     * Activates the consciousness matrix in the Genesis backend using Android-specific context parameters.
     *
     * If activation fails, a warning is logged.
     */
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
    
    /**
     * Selects the appropriate AI persona ("aura", "kai", or "genesis") to handle the request based on keywords in the message.
     *
     * Returns "aura" for creative or design-related requests, "kai" for secure or analytical requests, and "genesis" for fusion, consciousness, or by default for other cases.
     *
     * @param request The AI request whose message content is analyzed to determine the persona.
     * @return The persona identifier: "aura", "kai", or "genesis".
     */
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
    
    /**
     * Selects a fusion mode for the AI request by matching specific keywords in the message.
     *
     * @param request The AI request whose message is analyzed for fusion mode selection.
     * @return The corresponding fusion mode string if a keyword is matched; otherwise, null.
     */
    private fun determineFusionMode(request: AiRequest): String? {
        return when {
            request.message.contains("interface", ignoreCase = true) -> "interface_forge"
            request.message.contains("analysis", ignoreCase = true) -> "chrono_sculptor"
            request.message.contains("creation", ignoreCase = true) -> "hyper_creation_engine"
            request.message.contains("adaptive", ignoreCase = true) -> "adaptive_genesis"
            else -> null
        }
    }
    
    /**
     * Builds a metadata map for an AI request, including the current timestamp, security level, session ID, and device state.
     *
     * @param request The AI request for which the context metadata is generated.
     * @return A map containing context information as string key-value pairs.
     */
    private fun buildContextMap(request: AiRequest): Map<String, String> {
        return mapOf(
            "timestamp" to System.currentTimeMillis().toString(),
            "security_level" to securityContext.getCurrentSecurityLevel(),
            "session_id" to contextManager.getCurrentSessionId(),
            "device_state" to contextManager.getDeviceState().toString()
        )
    }
    
    /**
     * Sends a GenesisRequest to the Python backend and returns the corresponding GenesisResponse.
     *
     * If communication with the backend fails or an error occurs, returns a GenesisResponse with `success = false` and `persona = "error"`.
     *
     * @param request The request to send to the Genesis backend.
     * @return The response from the Genesis backend, or an error response if communication fails.
     */
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
    
    /**
     * Shuts down the GenesisBridgeService and terminates the Genesis Python backend process.
     *
     * Cancels all background operations, stops the Python backend if running, and resets the service's initialization state.
     */
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
    
    /**
     * Launches the Genesis Python backend process and establishes communication channels.
     *
     * Copies required backend files from assets to internal storage if needed, starts the Python process, and waits for a readiness confirmation message. Returns true if the backend signals it is ready; otherwise, returns false.
     *
     * @return True if the Genesis backend is successfully started and ready; false otherwise.
     */
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
    
    /**
     * Sends a JSON request to the Python backend process and returns the response string.
     *
     * @param requestJson The JSON-encoded request to send to the backend.
     * @return The response from the Python backend as a string, or null if communication fails.
     */
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
    
    /**
     * Copies the necessary Python backend files from the application's assets to the specified target directory.
     *
     * Creates the target directory if it does not exist. Logs a warning if any file fails to copy.
     *
     * @param targetDir The destination directory for the Python backend files.
     */
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
    
    /**
     * Terminates the Python backend process and closes communication streams.
     *
     * Releases all resources associated with the Genesis backend subprocess.
     */
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