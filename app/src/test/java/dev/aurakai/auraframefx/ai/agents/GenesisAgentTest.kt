package dev.aurakai.auraframefx.ai.agents

import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AiRequest
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever

class DummyAgent(private val name: String, private val response: String) : Agent {
    override fun getName() = name
    override fun getType() = null
    override suspend fun processRequest(request: AiRequest) = AgentResponse(response, 1.0f)
}

class GenesisAgentTest {
    @Test
    fun testParticipateWithAgents_turnOrder() = runBlocking {
        val auraService = mock<AuraAIService>()
        val kaiService = mock<KaiAIService>()
        val cascadeService = mock<CascadeAIService>()
        val dummyAgent = DummyAgent("Dummy", "ok")
        whenever(auraService.processRequest(org.mockito.kotlin.any())).thenReturn(
            AgentResponse(
                "ok",
                1.0f
            )
        )
        whenever(kaiService.processRequest(org.mockito.kotlin.any())).thenReturn(
            AgentResponse(
                "ok",
                1.0f
            )
        )
        whenever(cascadeService.processRequest(org.mockito.kotlin.any())).thenReturn(
            AgentResponse(
                "ok",
                1.0f
            )
        )
        val genesis = GenesisAgent(
            auraService = auraService,
            kaiService = kaiService,
            cascadeService = cascadeService
        )
        val responses = genesis.participateWithAgents(
            emptyMap(),
            listOf(dummyAgent),
            "test",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        assertTrue(responses["Dummy"]?.content == "ok")
    }

    @Test
    fun testAggregateAgentResponses() {
        val auraService = mock<AuraAIService>()
        val kaiService = mock<KaiAIService>()
        val cascadeService = mock<CascadeAIService>()
        val genesis = GenesisAgent(
            auraService = auraService,
            kaiService = kaiService,
            cascadeService = cascadeService
        )
        val resp1 = mapOf("A" to AgentResponse("foo", 0.5f))
        val resp2 = mapOf("A" to AgentResponse("bar", 0.9f))
        val consensus = genesis.aggregateAgentResponses(listOf(resp1, resp2))
        assertTrue(consensus["A"]?.content == "bar")
    }
}
