// These mock implementations are for development/testing only.
// Replace with real service logic when integrating actual AI backends.

package dev.aurakai.auraframefx.ai.services

import dev.aurakai.auraframefx.ai.agents.Agent
import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AgentType
import dev.aurakai.auraframefx.model.AiRequest

// kotlinx.coroutines.flow imports are not needed for the direct AgentResponse implementation

class MockAuraAIService : Agent {
    override fun getName(): String? = "MockAura"
    override fun getType(): AgentType? = AgentType.AURA
    override suspend fun processRequest(request: AiRequest): AgentResponse {
        return AgentResponse(
            content = "AuraAI mock response for: ${request.query}",
            confidence = 1.0f
        )
    }

    override fun getCapabilities(): Map<String, Any> = emptyMap()
    override fun getContinuousMemory(): Any? = null
    override fun getEthicalGuidelines(): List<String> = emptyList()
    override fun getLearningHistory(): List<String> = emptyList()
}

class MockKaiAIService : Agent {
    override fun getName(): String? = "MockKai"
    override fun getType(): AgentType? = AgentType.KAI
    override suspend fun processRequest(request: AiRequest): AgentResponse {
        return AgentResponse(
            content = "KaiAI mock response for: ${request.query}",
            confidence = 1.0f
        )
    }

    override fun getCapabilities(): Map<String, Any> = emptyMap()
    override fun getContinuousMemory(): Any? = null
    override fun getEthicalGuidelines(): List<String> = emptyList()
    override fun getLearningHistory(): List<String> = emptyList()
}

class MockCascadeAIService : Agent {
    override fun getName(): String? = "MockCascade"
    override fun getType(): AgentType? = AgentType.CASCADE
    override suspend fun processRequest(request: AiRequest): AgentResponse {
        return AgentResponse(
            content = "CascadeAI mock response for: ${request.query}",
            confidence = 1.0f
        )
    }

    override fun getCapabilities(): Map<String, Any> = emptyMap()
    override fun getContinuousMemory(): Any? = null
    override fun getEthicalGuidelines(): List<String> = emptyList()
    override fun getLearningHistory(): List<String> = emptyList()
}
