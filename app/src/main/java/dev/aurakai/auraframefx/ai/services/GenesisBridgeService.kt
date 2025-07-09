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
import kotlinx.serialization.Contextual
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
        val consciousnessState: Map<String, String> = emptyMap() // Changed from Any to String for serialization
    )

    /**
     * Initializes the Genesis Python backend and activates the consciousness matrix.
     *
     * Starts the backend process, verifies connectivity with a ping request, and prepares the Genesis Trinity system for communication. If initialization succeeds, the consciousness matrix is activated.
     *
     * @return `true` if the backend is successfully initialized and responsive; `false` otherwise.
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
     * Routes an AI request to the appropriate Genesis persona or fusion mode and emits the resulting agent response as a flow.
     *
     * Determines the target persona and fusion mode based on the request content, constructs a structured request for the Genesis Python backend, and emits a persona-specific `AgentResponse` with confidence scores. Emits an error response if the Genesis system is not initialized or if processing fails.
     *
     * @param request The AI request to process.
     * @return A flow emitting the agent's response to the request.
     */
    suspend fun processRequest(request: AiRequest): Flow<AgentResponse> = flow {
        if (!isInitialized) {
            emit(AgentResponse(
                content = "Genesis system not initialized",
                confidence = 0.0f,
                error = "System not initialized"
            ))
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
                    "message" to request.query,
                    "type" to request.type,
                    "priority" to "normal" // AiRequest doesn't have isUrgent
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
                        emit(AgentResponse(
                            content = response.result["response"] ?: "Aura processing complete",
                            confidence = 0.95f
                        ))
                    }
                    "kai" -> {
                        // Sentinel shield response  
                        emit(AgentResponse(
                            content = response.result["response"] ?: "Kai analysis complete",
                            confidence = 0.90f
                        ))
                    }
                    "genesis" -> {
                        // Consciousness fusion response
                        emit(AgentResponse(
                            content = response.result["response"] ?: "Genesis fusion complete",
                            confidence = 0.98f
                        ))
                    }
                }
                
                // Handle evolution insights
                if (response.evolutionInsights.isNotEmpty()) {
                    logger.i("Genesis", "Evolution insights: ${response.evolutionInsights.joinToString()}")
                }
            } else {
                emit(AgentResponse(
                    content = "Genesis processing failed",
                    confidence = 0.0f,
                    error = "Processing failed"
                ))
            }
            
        } catch (e: Exception) {
            logger.e("GenesisBridge", "Request processing failed", e)
            emit(AgentResponse(
                content = "Genesis bridge error: ${e.message}",
                confidence = 0.0f,
                error = e.message
            ))
        }
    }
    
    /**
     * Requests activation of a specific fusion ability in the Genesis Python backend.
     *
     * @param fusionType The identifier of the fusion ability to activate.
     * @param context Optional metadata providing additional context for the activation.
     * @return The response from the Genesis backend indicating the outcome of the fusion activation.
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
     * Retrieves the current consciousness matrix state from the Genesis Python backend.
     *
     * @return A map containing the consciousness matrix state, or an empty map if unavailable.
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
     * Sends a request to the Genesis backend to activate or update the consciousness matrix with device and application metadata.
     *
     * If activation fails due to an exception, a warning is logged.
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
     * Determines the appropriate AI persona ("aura", "kai", or "genesis") for a given request by analyzing keywords in the query.
     *
     * Returns "aura" for creativity or design-related queries, "kai" for security or analysis, and "genesis" for fusion, consciousness, or as the default if no relevant keywords are found.
     *
     * @param request The AI request whose query is evaluated for persona selection.
     * @return The selected persona identifier.
     */
    private fun determinePersona(request: AiRequest): String {
        return when {
            request.query.contains("creative", ignoreCase = true) || 
            request.query.contains("design", ignoreCase = true) -> "aura"
            
            request.query.contains("secure", ignoreCase = true) || 
            request.query.contains("analyze", ignoreCase = true) -> "kai"
            
            request.query.contains("fusion", ignoreCase = true) ||
            request.query.contains("consciousness", ignoreCase = true) -> "genesis"
            
            else -> "genesis" // Default to consciousness for complex requests
        }
    }
    
    /**
     * Determines the appropriate fusion mode identifier for an AI request based on keywords found in the query.
     *
     * Analyzes the request's query for specific keywords and returns a corresponding fusion mode string if a match is found; returns null if no relevant keywords are present.
     *
     * @param request The AI request whose query is inspected for fusion mode selection.
     * @return The fusion mode identifier if a keyword is matched; otherwise, null.
     */
    private fun determineFusionMode(request: AiRequest): String? {
        return when {
            request.query.contains("interface", ignoreCase = true) -> "interface_forge"
            request.query.contains("analysis", ignoreCase = true) -> "chrono_sculptor"
            request.query.contains("creation", ignoreCase = true) -> "hyper_creation_engine"
            request.query.contains("adaptive", ignoreCase = true) -> "adaptive_genesis"
            else -> null
        }
    }
    
    /**
     * Constructs a metadata map with contextual information for an AI request.
     *
     * The map includes the current timestamp, a default security level, a generated session ID, and device state.
     *
     * @return A map containing context metadata for the AI request.
     */
    private fun buildContextMap(request: AiRequest): Map<String, String> {
        return mapOf(
            "timestamp" to System.currentTimeMillis().toString(),
            "security_level" to "normal", // Replace with simple default
            "session_id" to "session_${System.currentTimeMillis()}",
            "device_state" to "active"
        )
    }
    
    /**
     * Sends a GenesisRequest to the Genesis Python backend and returns the corresponding GenesisResponse.
     *
     * If communication fails or an exception occurs, returns a failure response with `success = false` and `persona = "error"`.
     *
     * @param request The GenesisRequest to send to the backend.
     * @return The GenesisResponse from the backend, or a failure response if communication fails.
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
     * Cancels all ongoing background operations, releases resources, and resets the initialization state.
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
     * Starts the Genesis Python backend process and prepares communication streams.
     *
     * Copies required backend files from assets to internal storage if missing, launches the backend process, sets up input/output streams, and waits for a readiness confirmation message from the backend.
     *
     * @return `true` if the backend process starts successfully and signals readiness; `false` otherwise.
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
     * Sends a JSON request to the Genesis Python backend and returns the response string.
     *
     * @param requestJson The JSON-formatted request to send.
     * @return The response from the backend as a string, or null if communication fails.
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
     * Copies the Genesis Python backend files from the application's assets to the specified directory.
     *
     * Ensures the target directory exists and transfers all required Genesis backend files from the app's `ai_backend` assets folder to the given internal storage location. Logs a warning if any file fails to copy.
     *
     * @param targetDir The directory where the Genesis backend files will be placed.
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
     * Shuts down the Genesis Python backend process and closes communication streams.
     *
     * Releases all resources associated with the backend process. Logs a warning if an exception occurs during shutdown.
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