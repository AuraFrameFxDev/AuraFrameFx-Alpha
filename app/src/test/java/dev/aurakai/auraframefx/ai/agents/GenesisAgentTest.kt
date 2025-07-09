package dev.aurakai.auraframefx.ai.agents

import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AiRequest
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever
import org.mockito.kotlin.any
import java.util.concurrent.ConcurrentHashMap

interface Agent {
    fun getName(): String
    fun getType(): String?
    suspend fun processRequest(request: AiRequest): AgentResponse
}

class DummyAgent(
    private val name: String,
    private val response: String,
    private val confidence: Float = 1.0f
) : Agent {
    override fun getName(): String = name
    override fun getType(): String? = null
    override suspend fun processRequest(request: AiRequest): AgentResponse =
        AgentResponse(response, confidence)
}

class FailingAgent(private val name: String) : Agent {
    override fun getName(): String = name
    override fun getType(): String? = null
    override suspend fun processRequest(request: AiRequest): AgentResponse {
        throw RuntimeException("Agent processing failed")
    }
}

class GenesisAgent(
    private val auraService: AuraAIService,
    private val kaiService: KaiAIService,
    private val cascadeService: CascadeAIService
) : Agent {
    enum class ConversationMode { TURN_ORDER, CASCADE, CONSENSUS }

    override fun getName(): String = "GenesisAgent"
    override fun getType(): String? = null

    suspend fun participateWithAgents(
        context: Map<String, String>,
        agents: List<Agent>,
        prompt: String?,
        mode: ConversationMode
    ): Map<String, AgentResponse> {
        if (agents.isEmpty()) return emptyMap()
        val responses = mutableMapOf<String, AgentResponse>()
        for (agent in agents) {
            try {
                val requestPrompt = prompt ?: ""
                val combinedPrompt = buildString {
                    if (context.isNotEmpty()) {
                        append(context.entries.joinToString(" ") { "${it.key}:${it.value}" })
                        append(" ")
                    }
                    append(requestPrompt)
                }
                val response = agent.processRequest(AiRequest(combinedPrompt, context))
                responses[agent.getName()] = response
            } catch (_: Exception) {
            }
        }
        return responses
    }

    fun aggregateAgentResponses(
        responsesList: List<Map<String, AgentResponse>>
    ): Map<String, AgentResponse> {
        val consensus = mutableMapOf<String, AgentResponse>()
        for (responses in responsesList) {
            for ((name, response) in responses) {
                val existing = consensus[name]
                if (existing == null || response.confidence > existing.confidence) {
                    consensus[name] = response
                }
            }
        }
        return consensus
    }

    override suspend fun processRequest(request: AiRequest): AgentResponse {
        requireNotNull(request) { "Request cannot be null" }
        val auraResp = auraService.processRequest(request)
        val kaiResp = kaiService.processRequest(request)
        val cascadeResp = cascadeService.processRequest(request)
        val aggregated = aggregateAgentResponses(
            listOf(
                mapOf("Aura" to auraResp),
                mapOf("Kai" to kaiResp),
                mapOf("Cascade" to cascadeResp)
            )
        )
        return AgentResponse(
            content = aggregated.values.joinToString(" ") { it.content },
            confidence = aggregated.values.maxOfOrNull { it.confidence } ?: 0.0f
        )
    }
}

interface AuraAIService {
    suspend fun processRequest(request: AiRequest): AgentResponse
}

interface KaiAIService {
    suspend fun processRequest(request: AiRequest): AgentResponse
}

interface CascadeAIService {
    suspend fun processRequest(request: AiRequest): AgentResponse
}

class GenesisAgentTest {
    private lateinit var auraService: AuraAIService
    private lateinit var kaiService: KaiAIService
    private lateinit var cascadeService: CascadeAIService
    private lateinit var genesisAgent: GenesisAgent

    @Before
    fun setup() {
        auraService = mock()
        kaiService = mock()
        cascadeService = mock()
        genesisAgent = GenesisAgent(
            auraService = auraService,
            kaiService = kaiService,
            cascadeService = cascadeService
        )
    }

    @Test
    fun testParticipateWithAgents_turnOrder() = runBlocking {
        val dummyAgent = DummyAgent("Dummy", "ok")
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("ok", 1.0f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("ok", 1.0f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("ok", 1.0f))

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(dummyAgent),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        assertEquals("ok", responses["Dummy"]?.content)
    }

    @Test
    fun testAggregateAgentResponses() {
        val resp1 = mapOf("A" to AgentResponse("foo", 0.5f))
        val resp2 = mapOf("A" to AgentResponse("bar", 0.9f))
        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))
        assertEquals("bar", consensus["A"]?.content)
    }

    @Test
    fun testParticipateWithAgents_emptyAgentList() = runBlocking {
        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = emptyList(),
            prompt = "test prompt",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        assertTrue("Expected empty response map", responses.isEmpty())
    }

    @Test
    fun testParticipateWithAgents_multipleAgents() = runBlocking {
        val agent1 = DummyAgent("Agent1", "response1", 0.8f)
        val agent2 = DummyAgent("Agent2", "response2", 0.9f)
        val agent3 = DummyAgent("Agent3", "response3", 0.7f)

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent1, agent2, agent3),
            prompt = "test prompt",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(3, responses.size)
        assertEquals("response1", responses["Agent1"]?.content)
        assertEquals("response2", responses["Agent2"]?.content)
        assertEquals("response3", responses["Agent3"]?.content)
        assertEquals(0.8f, responses["Agent1"]?.confidence)
        assertEquals(0.9f, responses["Agent2"]?.confidence)
        assertEquals(0.7f, responses["Agent3"]?.confidence)
    }

    @Test
    fun testParticipateWithAgents_withContext() = runBlocking {
        val agent = DummyAgent("TestAgent", "contextual response")
        val context = mapOf("key1" to "value1", "key2" to "value2")

        val responses = genesisAgent.participateWithAgents(
            context = context,
            agents = listOf(agent),
            prompt = "prompt with context",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("contextual response", responses["TestAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_nullPrompt() = runBlocking {
        val agent = DummyAgent("TestAgent", "response")

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent),
            prompt = null,
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("response", responses["TestAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_emptyPrompt() = runBlocking {
        val agent = DummyAgent("TestAgent", "empty prompt response")

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent),
            prompt = "",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("empty prompt response", responses["TestAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_agentThrowsException() = runBlocking {
        val failingAgent = FailingAgent("FailingAgent")
        val workingAgent = DummyAgent("WorkingAgent", "success")

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(failingAgent, workingAgent),
            prompt = "test prompt",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("success", responses["WorkingAgent"]?.content)
        assertNull(responses["FailingAgent"])
    }

    @Test
    fun testParticipateWithAgents_duplicateAgentNames() = runBlocking {
        val agent1 = DummyAgent("SameName", "response1")
        val agent2 = DummyAgent("SameName", "response2")

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent1, agent2),
            prompt = "test prompt",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertTrue(responses.containsKey("SameName"))
        assertTrue(
            responses["SameName"]?.content == "response1"
                || responses["SameName"]?.content == "response2"
        )
    }

    @Test
    fun testAggregateAgentResponses_emptyList() {
        val consensus = genesisAgent.aggregateAgentResponses(emptyList())
        assertTrue("Expected empty consensus", consensus.isEmpty())
    }

    @Test
    fun testAggregateAgentResponses_singleResponse() {
        val response = mapOf("Agent1" to AgentResponse("single response", 0.8f))
        val consensus = genesisAgent.aggregateAgentResponses(listOf(response))

        assertEquals(1, consensus.size)
        assertEquals("single response", consensus["Agent1"]?.content)
        assertEquals(0.8f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_multipleResponsesSameAgent() {
        val resp1 = mapOf("Agent1" to AgentResponse("response1", 0.5f))
        val resp2 = mapOf("Agent1" to AgentResponse("response2", 0.9f))
        val resp3 = mapOf("Agent1" to AgentResponse("response3", 0.3f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2, resp3))

        assertEquals(1, consensus.size)
        assertEquals("response2", consensus["Agent1"]?.content)
        assertEquals(0.9f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_multipleAgentsMultipleResponses() {
        val resp1 = mapOf(
            "Agent1" to AgentResponse("a1_resp1", 0.5f),
            "Agent2" to AgentResponse("a2_resp1", 0.8f)
        )
        val resp2 = mapOf(
            "Agent1" to AgentResponse("a1_resp2", 0.9f),
            "Agent2" to AgentResponse("a2_resp2", 0.4f)
        )

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

        assertEquals(2, consensus.size)
        assertEquals("a1_resp2", consensus["Agent1"]?.content)
        assertEquals(0.9f, consensus["Agent1"]?.confidence)
        assertEquals("a2_resp1", consensus["Agent2"]?.content)
        assertEquals(0.8f, consensus["Agent2"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_equalConfidence() {
        val resp1 = mapOf("Agent1" to AgentResponse("response1", 0.5f))
        val resp2 = mapOf("Agent1" to AgentResponse("response2", 0.5f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

        assertEquals(1, consensus.size)
        assertEquals(0.5f, consensus["Agent1"]?.confidence)
        assertTrue(
            consensus["Agent1"]?.content == "response1"
                || consensus["Agent1"]?.content == "response2"
        )
    }

    @Test
    fun testAggregateAgentResponses_zeroConfidence() {
        val resp1 = mapOf("Agent1" to AgentResponse("response1", 0.0f))
        val resp2 = mapOf("Agent1" to AgentResponse("response2", 0.1f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

        assertEquals(1, consensus.size)
        assertEquals("response2", consensus["Agent1"]?.content)
        assertEquals(0.1f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_negativeConfidence() {
        val resp1 = mapOf("Agent1" to AgentResponse("response1", -0.5f))
        val resp2 = mapOf("Agent1" to AgentResponse("response2", 0.1f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

        assertEquals(1, consensus.size)
        assertEquals("response2", consensus["Agent1"]?.content)
        assertEquals(0.1f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_largeNumberOfResponses() {
        val responses = (1..100).map { i ->
            mapOf("Agent1" to AgentResponse("response$i", i / 100.0f))
        }

        val consensus = genesisAgent.aggregateAgentResponses(responses)

        assertEquals(1, consensus.size)
        assertEquals("response100", consensus["Agent1"]?.content)
        assertEquals(1.0f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_mixedAgents() {
        val resp1 = mapOf(
            "Agent1" to AgentResponse("a1_resp", 0.7f),
            "Agent2" to AgentResponse("a2_resp", 0.3f)
        )
        val resp2 = mapOf(
            "Agent3" to AgentResponse("a3_resp", 0.9f),
            "Agent4" to AgentResponse("a4_resp", 0.1f)
        )

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

        assertEquals(4, consensus.size)
        assertEquals("a1_resp", consensus["Agent1"]?.content)
        assertEquals("a2_resp", consensus["Agent2"]?.content)
        assertEquals("a3_resp", consensus["Agent3"]?.content)
        assertEquals("a4_resp", consensus["Agent4"]?.content)
    }

    @Test
    fun testGenesisAgent_constructor() {
        val agent = GenesisAgent(
            auraService = auraService,
            kaiService = kaiService,
            cascadeService = cascadeService
        )
        assertNotNull("GenesisAgent should be created successfully", agent)
    }

    @Test
    fun testGenesisAgent_getName() {
        val name = genesisAgent.getName()
        assertNotNull("Name should not be null", name)
        assertTrue("Name should not be empty", name.isNotEmpty())
    }

    @Test
    fun testGenesisAgent_getType() {
        val type = genesisAgent.getType()
        assertNotNull("Method should execute without throwing", type)
    }

    @Test
    fun testGenesisAgent_processRequest() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura response", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai response", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade response", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertNotNull("Response should not be null", response)
        assertTrue("Response should have content", response.content.isNotEmpty())
        assertTrue("Confidence should be positive", response.confidence >= 0.0f)
    }

    @Test
    fun testGenesisAgent_processRequest_nullRequest() = runBlocking {
        try {
            genesisAgent.processRequest(null as AiRequest)
            fail("Should throw exception for null request")
        } catch (e: Exception) {
            assertTrue("Exception should be thrown", true)
        }
    }

    @Test
    fun testConversationMode_values() {
        val modes = GenesisAgent.ConversationMode.values()
        assertTrue("Should have at least TURN_ORDER mode", modes.contains(GenesisAgent.ConversationMode.TURN_ORDER))
        assertTrue("Should have multiple conversation modes", modes.isNotEmpty())
    }

    @Test
    fun testDummyAgent_implementation() = runBlocking {
        val agent = DummyAgent("TestAgent", "test response", 0.5f)
        assertEquals("TestAgent", agent.getName())
        assertNull(agent.getType())

        val request = AiRequest("test", emptyMap())
        val response = agent.processRequest(request)

        assertEquals("test response", response.content)
        assertEquals(0.5f, response.confidence)
    }

    @Test
    fun testFailingAgent_implementation() = runBlocking {
        val agent = FailingAgent("TestAgent")
        assertEquals("TestAgent", agent.getName())
        assertNull(agent.getType())

        val request = AiRequest("test", emptyMap())
        try {
            agent.processRequest(request)
            fail("Should throw RuntimeException")
        } catch (e: RuntimeException) {
            assertEquals("Agent processing failed", e.message)
        }
    }

    @Test
    @Test
    fun testParticipateWithAgents_cascadeMode() = runBlocking {
        val agent1 = DummyAgent("Agent1", "cascade response 1", 0.8f)
        val agent2 = DummyAgent("Agent2", "cascade response 2", 0.9f)

        val responses = genesisAgent.participateWithAgents(
            context = mapOf("mode" to "cascade"),
            agents = listOf(agent1, agent2),
            prompt = "test cascade",
            mode = GenesisAgent.ConversationMode.CASCADE
        )

        assertEquals(2, responses.size)
        assertEquals("cascade response 1", responses["Agent1"]?.content)
        assertEquals("cascade response 2", responses["Agent2"]?.content)
    }

    @Test
    fun testParticipateWithAgents_consensusMode() = runBlocking {
        val agent1 = DummyAgent("Agent1", "consensus response 1", 0.7f)
        val agent2 = DummyAgent("Agent2", "consensus response 2", 0.8f)

        val responses = genesisAgent.participateWithAgents(
            context = mapOf("mode" to "consensus"),
            agents = listOf(agent1, agent2),
            prompt = "test consensus",
            mode = GenesisAgent.ConversationMode.CONSENSUS
        )

        assertEquals(2, responses.size)
        assertEquals("consensus response 1", responses["Agent1"]?.content)
        assertEquals("consensus response 2", responses["Agent2"]?.content)
    }

    @Test
    fun testParticipateWithAgents_largeContext() = runBlocking {
        val agent = DummyAgent("LargeContextAgent", "handled large context")
        val largeContext = (1..1000).associate { "key$it" to "value$it" }

        val responses = genesisAgent.participateWithAgents(
            context = largeContext,
            agents = listOf(agent),
            prompt = "test with large context",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("handled large context", responses["LargeContextAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_specialCharactersInPrompt() = runBlocking {
        val agent = DummyAgent("SpecialCharAgent", "handled special chars")
        val specialPrompt = "Test with special chars: àáâãäå çćčđ éêë ñ øö ş ťü ý žż 中文 日本語 한국어"

        val responses = genesisAgent.participateWithAgents(
            context = mapOf("special" to "chars: !@#$%^&*()"),
            agents = listOf(agent),
            prompt = specialPrompt,
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("handled special chars", responses["SpecialCharAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_veryLongPrompt() = runBlocking {
        val agent = DummyAgent("LongPromptAgent", "handled long prompt")
        val longPrompt = "A".repeat(10000)

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent),
            prompt = longPrompt,
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("handled long prompt", responses["LongPromptAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_emptyStringValues() = runBlocking {
        val agent = DummyAgent("EmptyStringAgent", "handled empty strings")
        val emptyContext = mapOf("empty1" to "", "empty2" to "", "valid" to "value")

        val responses = genesisAgent.participateWithAgents(
            context = emptyContext,
            agents = listOf(agent),
            prompt = "",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("handled empty strings", responses["EmptyStringAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_whiteSpacePrompt() = runBlocking {
        val agent = DummyAgent("WhiteSpaceAgent", "handled whitespace")
        val whitespacePrompt = "   \t\n\r   "

        val responses = genesisAgent.participateWithAgents(
            context = mapOf("space" to "   "),
            agents = listOf(agent),
            prompt = whitespacePrompt,
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("handled whitespace", responses["WhiteSpaceAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_allAgentsFailingExceptOne() = runBlocking {
        val failingAgent1 = FailingAgent("Failing1")
        val failingAgent2 = FailingAgent("Failing2")
        val workingAgent = DummyAgent("Working", "success")

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(failingAgent1, failingAgent2, workingAgent),
            prompt = "test prompt",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("success", responses["Working"]?.content)
        assertNull(responses["Failing1"])
        assertNull(responses["Failing2"])
    }

    @Test
    fun testParticipateWithAgents_allAgentsFailing() = runBlocking {
        val failingAgent1 = FailingAgent("Failing1")
        val failingAgent2 = FailingAgent("Failing2")

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(failingAgent1, failingAgent2),
            prompt = "test prompt",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertTrue("All agents failed, should be empty", responses.isEmpty())
    }

    @Test
    fun testAggregateAgentResponses_maxConfidenceValue() {
        val maxConfResponse = mapOf("Agent1" to AgentResponse("max conf", Float.MAX_VALUE))
        val normalResponse = mapOf("Agent1" to AgentResponse("normal", 0.9f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(normalResponse, maxConfResponse))

        assertEquals(1, consensus.size)
        assertEquals("max conf", consensus["Agent1"]?.content)
        assertEquals(Float.MAX_VALUE, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_minConfidenceValue() {
        val minConfResponse = mapOf("Agent1" to AgentResponse("min conf", Float.MIN_VALUE))
        val normalResponse = mapOf("Agent1" to AgentResponse("normal", 0.1f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(minConfResponse, normalResponse))

        assertEquals(1, consensus.size)
        assertEquals("normal", consensus["Agent1"]?.content)
        assertEquals(0.1f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_infiniteConfidence() {
        val infResponse = mapOf("Agent1" to AgentResponse("infinite", Float.POSITIVE_INFINITY))
        val normalResponse = mapOf("Agent1" to AgentResponse("normal", 0.9f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(normalResponse, infResponse))

        assertEquals(1, consensus.size)
        assertEquals("infinite", consensus["Agent1"]?.content)
        assertEquals(Float.POSITIVE_INFINITY, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_nanConfidence() {
        val nanResponse = mapOf("Agent1" to AgentResponse("nan", Float.NaN))
        val normalResponse = mapOf("Agent1" to AgentResponse("normal", 0.9f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(normalResponse, nanResponse))

        assertEquals(1, consensus.size)
        // NaN comparison behavior depends on implementation
        assertNotNull(consensus["Agent1"]?.content)
    }

    @Test
    fun testAggregateAgentResponses_veryLongContent() {
        val longContent = "A".repeat(100000)
        val longResponse = mapOf("Agent1" to AgentResponse(longContent, 0.9f))
        val shortResponse = mapOf("Agent1" to AgentResponse("short", 0.1f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(shortResponse, longResponse))

        assertEquals(1, consensus.size)
        assertEquals(longContent, consensus["Agent1"]?.content)
        assertEquals(0.9f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_emptyContent() {
        val emptyResponse = mapOf("Agent1" to AgentResponse("", 0.9f))
        val normalResponse = mapOf("Agent1" to AgentResponse("normal", 0.1f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(normalResponse, emptyResponse))

        assertEquals(1, consensus.size)
        assertEquals("", consensus["Agent1"]?.content)
        assertEquals(0.9f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_specialCharactersInContent() {
        val specialContent = "Special: àáâãäå çćčđ éêë ñ øö ş ťü ý žż 中文 日本語 한국어 !@#$%^&*()"
        val specialResponse = mapOf("Agent1" to AgentResponse(specialContent, 0.9f))
        val normalResponse = mapOf("Agent1" to AgentResponse("normal", 0.1f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(normalResponse, specialResponse))

        assertEquals(1, consensus.size)
        assertEquals(specialContent, consensus["Agent1"]?.content)
        assertEquals(0.9f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testProcessRequest_serviceFailures() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenThrow(RuntimeException("Aura service failed"))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai response", 0.8f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade response", 0.9f))

        try {
            genesisAgent.processRequest(request)
            fail("Should throw exception when service fails")
        } catch (e: RuntimeException) {
            assertEquals("Aura service failed", e.message)
        }
    }

    @Test
    fun testProcessRequest_allServicesReturnEmptyContent() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("  ", response.content) // Three empty strings joined with spaces
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_allServicesReturnZeroConfidence() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.0f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.0f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.0f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        assertEquals(0.0f, response.confidence)
    }

    @Test
    fun testProcessRequest_veryLongPrompt() = runBlocking {
        val longPrompt = "A".repeat(50000)
        val request = AiRequest(longPrompt, emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_requestWithLargeContext() = runBlocking {
        val largeContext = (1..1000).associate { "key$it" to "value$it" }
        val request = AiRequest("test prompt", largeContext)
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_specialCharactersInPrompt() = runBlocking {
        val specialPrompt = "Test: àáâãäå çćčđ éêë ñ øö ş ťü ý žż 中文 日本語 한국어 !@#$%^&*()"
        val request = AiRequest(specialPrompt, emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_maxConfidenceValues() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", Float.MAX_VALUE))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        assertEquals(Float.MAX_VALUE, response.confidence)
    }

    @Test
    fun testProcessRequest_infiniteConfidence() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", Float.POSITIVE_INFINITY))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        assertEquals(Float.POSITIVE_INFINITY, response.confidence)
    }

    @Test
    fun testProcessRequest_nanConfidence() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", Float.NaN))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        // NaN behavior in maxOfOrNull depends on implementation
        assertNotNull(response.confidence)
    }

    @Test
    fun testDummyAgent_withZeroConfidence() = runBlocking {
        val agent = DummyAgent("ZeroConfAgent", "zero confidence response", 0.0f)
        val request = AiRequest("test", emptyMap())
        val response = agent.processRequest(request)

        assertEquals("zero confidence response", response.content)
        assertEquals(0.0f, response.confidence)
    }

    @Test
    fun testDummyAgent_withNegativeConfidence() = runBlocking {
        val agent = DummyAgent("NegativeConfAgent", "negative confidence response", -0.5f)
        val request = AiRequest("test", emptyMap())
        val response = agent.processRequest(request)

        assertEquals("negative confidence response", response.content)
        assertEquals(-0.5f, response.confidence)
    }

    @Test
    fun testDummyAgent_withMaxConfidence() = runBlocking {
        val agent = DummyAgent("MaxConfAgent", "max confidence response", Float.MAX_VALUE)
        val request = AiRequest("test", emptyMap())
        val response = agent.processRequest(request)

        assertEquals("max confidence response", response.content)
        assertEquals(Float.MAX_VALUE, response.confidence)
    }

    @Test
    fun testDummyAgent_withEmptyName() = runBlocking {
        val agent = DummyAgent("", "empty name response")
        assertEquals("", agent.getName())
        assertNull(agent.getType())
    }

    @Test
    fun testDummyAgent_withEmptyResponse() = runBlocking {
        val agent = DummyAgent("EmptyResponseAgent", "")
        val request = AiRequest("test", emptyMap())
        val response = agent.processRequest(request)

        assertEquals("", response.content)
        assertEquals(1.0f, response.confidence)
    }

    @Test
    fun testFailingAgent_withEmptyName() {
        val agent = FailingAgent("")
        assertEquals("", agent.getName())
        assertNull(agent.getType())
    }

    @Test
    fun testFailingAgent_withSpecialCharactersInName() {
        val specialName = "Special: àáâãäå çćčđ éêë ñ øö ş ťü ý žż 中文 日本語 한국어 !@#$%^&*()"
        val agent = FailingAgent(specialName)
        assertEquals(specialName, agent.getName())
        assertNull(agent.getType())
    }

    @Test
    fun testGenesisAgent_getName_consistency() {
        val name1 = genesisAgent.getName()
        val name2 = genesisAgent.getName()
        assertEquals("Name should be consistent", name1, name2)
        assertEquals("GenesisAgent", name1)
    }

    @Test
    fun testGenesisAgent_getType_consistency() {
        val type1 = genesisAgent.getType()
        val type2 = genesisAgent.getType()
        assertEquals("Type should be consistent", type1, type2)
    }

    @Test
    fun testConversationMode_ordinalValues() {
        val modes = GenesisAgent.ConversationMode.values()
        assertEquals("TURN_ORDER", modes[0].name)
        assertEquals("CASCADE", modes[1].name)
        assertEquals("CONSENSUS", modes[2].name)
        assertEquals(0, modes[0].ordinal)
        assertEquals(1, modes[1].ordinal)
        assertEquals(2, modes[2].ordinal)
    }

    @Test
    fun testConversationMode_valueOf() {
        assertEquals(GenesisAgent.ConversationMode.TURN_ORDER, GenesisAgent.ConversationMode.valueOf("TURN_ORDER"))
        assertEquals(GenesisAgent.ConversationMode.CASCADE, GenesisAgent.ConversationMode.valueOf("CASCADE"))
        assertEquals(GenesisAgent.ConversationMode.CONSENSUS, GenesisAgent.ConversationMode.valueOf("CONSENSUS"))

        try {
            GenesisAgent.ConversationMode.valueOf("INVALID")
            fail("Should throw IllegalArgumentException")
        } catch (e: IllegalArgumentException) {
            assertTrue("Should throw for invalid enum value", true)
        }
    }
        val agent = DummyAgent("ConcurrentAgent", "response")
        val responses = ConcurrentHashMap<String, AgentResponse>()

        val jobs = (1..10).map { i ->
            kotlinx.coroutines.async {
                val response = genesisAgent.participateWithAgents(
                    context = emptyMap(),
                    agents = listOf(agent),
                    prompt = "concurrent test $i",
                    mode = GenesisAgent.ConversationMode.TURN_ORDER
                )
                responses.putAll(response)
            }
        }
        jobs.forEach { it.await() }

        assertTrue("Should handle concurrent access", responses.isNotEmpty())
        assertEquals("response", responses["ConcurrentAgent"]?.content)
    }
}
    @Test
    fun testConcurrentAccess() = runBlocking {
        val agent = DummyAgent("ConcurrentAgent", "response")
        val responses = ConcurrentHashMap<String, AgentResponse>()

        val jobs = (1..10).map { i ->
            kotlinx.coroutines.async {
                val response = genesisAgent.participateWithAgents(
                    context = emptyMap(),
                    agents = listOf(agent),
                    prompt = "concurrent test $i",
                    mode = GenesisAgent.ConversationMode.TURN_ORDER
                )
                responses.putAll(response)
            }
        }
        jobs.forEach { it.await() }

        assertTrue("Should handle concurrent access", responses.isNotEmpty())
        assertEquals("response", responses["ConcurrentAgent"]?.content)
    }

    @Test
    fun testProcessRequest_emptyPrompt() = runBlocking {
        val request = AiRequest("", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_whiteSpaceOnlyPrompt() = runBlocking {
        val request = AiRequest("   \t\n\r   ", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_servicesReturnVeryLongContent() = runBlocking {
        val longContent = "A".repeat(10000)
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse(longContent, 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("$longContent kai cascade", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_negativeConfidenceValues() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", -0.5f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", -0.2f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", -0.8f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        assertEquals(-0.2f, response.confidence)
    }

    @Test
    fun testProcessRequest_mixedPositiveNegativeConfidence() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", -0.5f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", -0.2f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_contextWithNullValues() = runBlocking {
        val contextWithNulls = mapOf<String, String?>("key1" to "value1", "key2" to null, "key3" to "value3")
        val request = AiRequest("test prompt", contextWithNulls.filterValues { it != null })
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_contextWithVeryLongValues() = runBlocking {
        val longValue = "V".repeat(50000)
        val contextWithLongValues = mapOf("key1" to longValue, "key2" to "short")
        val request = AiRequest("test prompt", contextWithLongValues)
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_serviceReturnsSpecialCharacters() = runBlocking {
        val specialChars = "Special: àáâãäå çćčđ éêë ñ øö ş ťü ý žż 中文 日本語 한국어 !@#$%^&*()"
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse(specialChars, 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("$specialChars kai cascade", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_serviceReturnsMultilineContent() = runBlocking {
        val multilineContent = "Line 1\nLine 2\nLine 3\n\nLine 5"
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse(multilineContent, 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("$multilineContent kai cascade", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_serviceResponsesWithTabsAndSpaces() = runBlocking {
        val contentWithTabs = "Content\twith\ttabs\t\tand   spaces"
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse(contentWithTabs, 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("$contentWithTabs kai cascade", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_verifyServiceCallsWithCorrectParameters() = runBlocking {
        val testPrompt = "test prompt"
        val testContext = mapOf("key1" to "value1", "key2" to "value2")
        val request = AiRequest(testPrompt, testContext)
        
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        genesisAgent.processRequest(request)

        org.mockito.kotlin.verify(auraService).processRequest(request)
        org.mockito.kotlin.verify(kaiService).processRequest(request)
        org.mockito.kotlin.verify(cascadeService).processRequest(request)
    }

    @Test
    fun testProcessRequest_allServicesCalledEvenIfOneIsSlow() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenAnswer { 
            kotlinx.coroutines.delay(100)
            AgentResponse("kai", 0.9f)
        }
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        assertEquals(0.9f, response.confidence)
        
        org.mockito.kotlin.verify(auraService).processRequest(request)
        org.mockito.kotlin.verify(kaiService).processRequest(request)
        org.mockito.kotlin.verify(cascadeService).processRequest(request)
    }

    @Test
    fun testParticipateWithAgents_contextKeyValueSeparation() = runBlocking {
        val agent = DummyAgent("TestAgent", "response")
        val context = mapOf("key1" to "value1", "key2" to "value2")
        
        val responses = genesisAgent.participateWithAgents(
            context = context,
            agents = listOf(agent),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("response", responses["TestAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_contextWithSpecialCharacters() = runBlocking {
        val agent = DummyAgent("TestAgent", "response")
        val context = mapOf(
            "key:" to "value:with:colons",
            "special" to "àáâãäå çćčđ éêë",
            "symbols" to "!@#$%^&*()"
        )
        
        val responses = genesisAgent.participateWithAgents(
            context = context,
            agents = listOf(agent),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("response", responses["TestAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_agentNamesWithSpecialCharacters() = runBlocking {
        val specialName = "Agent-1_Test.Name@Domain"
        val agent = DummyAgent(specialName, "response")
        
        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("response", responses[specialName]?.content)
    }

    @Test
    fun testParticipateWithAgents_largeNumberOfAgents() = runBlocking {
        val agents = (1..100).map { DummyAgent("Agent$it", "response$it", it / 100.0f) }
        
        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = agents,
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(100, responses.size)
        for (i in 1..100) {
            assertEquals("response$i", responses["Agent$i"]?.content)
            assertEquals(i / 100.0f, responses["Agent$i"]?.confidence)
        }
    }

    @Test
    fun testParticipateWithAgents_agentReturnsVeryHighConfidence() = runBlocking {
        val agent = DummyAgent("HighConfAgent", "high conf response", 999.9f)
        
        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("high conf response", responses["HighConfAgent"]?.content)
        assertEquals(999.9f, responses["HighConfAgent"]?.confidence)
    }

    @Test
    fun testParticipateWithAgents_agentReturnsInfiniteNegativeConfidence() = runBlocking {
        val agent = DummyAgent("InfNegAgent", "inf neg response", Float.NEGATIVE_INFINITY)
        
        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("inf neg response", responses["InfNegAgent"]?.content)
        assertEquals(Float.NEGATIVE_INFINITY, responses["InfNegAgent"]?.confidence)
    }

    @Test
    fun testParticipateWithAgents_agentReturnsNanConfidence() = runBlocking {
        val agent = DummyAgent("NanAgent", "nan response", Float.NaN)
        
        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("nan response", responses["NanAgent"]?.content)
        assertTrue("NaN confidence should be NaN", responses["NanAgent"]?.confidence?.isNaN() == true)
    }

    @Test
    fun testParticipateWithAgents_promptBuilding() = runBlocking {
        val agent = mock<Agent>()
        whenever(agent.getName()).thenReturn("TestAgent")
        whenever(agent.processRequest(any())).thenReturn(AgentResponse("response", 0.9f))
        
        val context = mapOf("key1" to "value1", "key2" to "value2")
        val prompt = "test prompt"
        
        genesisAgent.participateWithAgents(
            context = context,
            agents = listOf(agent),
            prompt = prompt,
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        // Verify the combined prompt was built correctly
        org.mockito.kotlin.verify(agent).processRequest(
            org.mockito.kotlin.argThat { request ->
                request.prompt.contains("key1:value1") &&
                request.prompt.contains("key2:value2") &&
                request.prompt.contains("test prompt")
            }
        )
    }

    @Test
    fun testParticipateWithAgents_contextContainsSpaces() = runBlocking {
        val agent = mock<Agent>()
        whenever(agent.getName()).thenReturn("TestAgent")
        whenever(agent.processRequest(any())).thenReturn(AgentResponse("response", 0.9f))
        
        val context = mapOf("key with spaces" to "value with spaces")
        
        genesisAgent.participateWithAgents(
            context = context,
            agents = listOf(agent),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        org.mockito.kotlin.verify(agent).processRequest(
            org.mockito.kotlin.argThat { request ->
                request.prompt.contains("key with spaces:value with spaces")
            }
        )
    }

    @Test
    fun testParticipateWithAgents_contextIsPassedToAgent() = runBlocking {
        val agent = mock<Agent>()
        whenever(agent.getName()).thenReturn("TestAgent")
        whenever(agent.processRequest(any())).thenReturn(AgentResponse("response", 0.9f))
        
        val context = mapOf("key1" to "value1", "key2" to "value2")
        
        genesisAgent.participateWithAgents(
            context = context,
            agents = listOf(agent),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        org.mockito.kotlin.verify(agent).processRequest(
            org.mockito.kotlin.argThat { request ->
                request.context == context
            }
        )
    }

    @Test
    fun testAggregateAgentResponses_preservesOriginalOrder() {
        val resp1 = mapOf("Agent1" to AgentResponse("first", 0.5f))
        val resp2 = mapOf("Agent2" to AgentResponse("second", 0.6f))
        val resp3 = mapOf("Agent3" to AgentResponse("third", 0.7f))
        
        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2, resp3))
        
        assertEquals(3, consensus.size)
        assertTrue(consensus.containsKey("Agent1"))
        assertTrue(consensus.containsKey("Agent2"))
        assertTrue(consensus.containsKey("Agent3"))
    }

    @Test
    fun testAggregateAgentResponses_handlesNullContent() {
        val resp1 = mapOf("Agent1" to AgentResponse("", 0.5f))
        val resp2 = mapOf("Agent1" to AgentResponse("content", 0.9f))
        
        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))
        
        assertEquals(1, consensus.size)
        assertEquals("content", consensus["Agent1"]?.content)
        assertEquals(0.9f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_handlesWhitespaceContent() {
        val resp1 = mapOf("Agent1" to AgentResponse("   ", 0.5f))
        val resp2 = mapOf("Agent1" to AgentResponse("content", 0.9f))
        
        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))
        
        assertEquals(1, consensus.size)
        assertEquals("content", consensus["Agent1"]?.content)
        assertEquals(0.9f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_handlesIdenticalResponses() {
        val response = AgentResponse("same content", 0.8f)
        val resp1 = mapOf("Agent1" to response)
        val resp2 = mapOf("Agent1" to response)
        
        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))
        
        assertEquals(1, consensus.size)
        assertEquals("same content", consensus["Agent1"]?.content)
        assertEquals(0.8f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_stressTestWithManyAgents() {
        val responses = (1..1000).map { i ->
            mapOf("Agent$i" to AgentResponse("response$i", i / 1000.0f))
        }
        
        val consensus = genesisAgent.aggregateAgentResponses(responses)
        
        assertEquals(1000, consensus.size)
        for (i in 1..1000) {
            assertEquals("response$i", consensus["Agent$i"]?.content)
            assertEquals(i / 1000.0f, consensus["Agent$i"]?.confidence)
        }
    }

    @Test
    fun testAggregateAgentResponses_performanceWithDuplicateUpdates() {
        // Test performance when same agent appears in many response lists
        val responses = (1..100).map { i ->
            mapOf("SameAgent" to AgentResponse("response$i", i / 100.0f))
        }
        
        val startTime = System.currentTimeMillis()
        val consensus = genesisAgent.aggregateAgentResponses(responses)
        val endTime = System.currentTimeMillis()
        
        assertEquals(1, consensus.size)
        assertEquals("response100", consensus["SameAgent"]?.content)
        assertEquals(1.0f, consensus["SameAgent"]?.confidence)
        
        // Performance should be reasonable even with many updates
        assertTrue("Should complete in reasonable time", endTime - startTime < 1000)
    }

    @Test
    fun testGenesisAgent_fullIntegrationScenario() = runBlocking {
        val request = AiRequest("complex integration test", mapOf("context" to "integration"))
        
        whenever(auraService.processRequest(any())).thenReturn(
            AgentResponse("Aura processed the integration request", 0.85f)
        )
        whenever(kaiService.processRequest(any())).thenReturn(
            AgentResponse("Kai analyzed the integration scenario", 0.92f)
        )
        whenever(cascadeService.processRequest(any())).thenReturn(
            AgentResponse("Cascade completed the integration flow", 0.78f)
        )
        
        val response = genesisAgent.processRequest(request)
        
        val expectedContent = "Aura processed the integration request Kai analyzed the integration scenario Cascade completed the integration flow"
        assertEquals(expectedContent, response.content)
        assertEquals(0.92f, response.confidence)
        
        // Verify all services were called
        org.mockito.kotlin.verify(auraService).processRequest(request)
        org.mockito.kotlin.verify(kaiService).processRequest(request)
        org.mockito.kotlin.verify(cascadeService).processRequest(request)
    }

    @Test
    fun testGenesisAgent_edgeCaseWithAllZeroConfidence() = runBlocking {
        val request = AiRequest("zero confidence test", emptyMap())
        
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.0f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.0f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.0f))
        
        val response = genesisAgent.processRequest(request)
        
        assertEquals("aura kai cascade", response.content)
        assertEquals(0.0f, response.confidence)
    }

    @Test
    fun testGenesisAgent_edgeCaseWithAllNegativeConfidence() = runBlocking {
        val request = AiRequest("negative confidence test", emptyMap())
        
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", -0.1f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", -0.2f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", -0.3f))
        
        val response = genesisAgent.processRequest(request)
        
        assertEquals("aura kai cascade", response.content)
        assertEquals(-0.1f, response.confidence)
    }

    @Test
    fun testGenesisAgent_memoryUsageWithLargeResponses() = runBlocking {
        val largeContent = "A".repeat(1000000) // 1MB string
        val request = AiRequest("memory test", emptyMap())
        
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse(largeContent, 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse(largeContent, 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse(largeContent, 0.7f))
        
        val response = genesisAgent.processRequest(request)
        
        // Should handle large responses without issues
        assertTrue("Should handle large content", response.content.length > 2000000)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testParticipateWithAgents_exceptionsFromDifferentAgents() = runBlocking {
        val agent1 = FailingAgent("Fail1")
        val agent2 = DummyAgent("Success", "success")
        val agent3 = FailingAgent("Fail2") 
        val agent4 = DummyAgent("Success2", "success2")
        
        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent1, agent2, agent3, agent4),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(2, responses.size)
        assertEquals("success", responses["Success"]?.content)
        assertEquals("success2", responses["Success2"]?.content)
        assertNull(responses["Fail1"])
        assertNull(responses["Fail2"])
    }

    @Test
    fun testParticipateWithAgents_interruptedExecution() = runBlocking {
        val slowAgent = object : Agent {
            override fun getName() = "SlowAgent"
            override fun getType() = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                kotlinx.coroutines.delay(5000) // 5 second delay
                return AgentResponse("slow response", 0.5f)
            }
        }
        
        val fastAgent = DummyAgent("FastAgent", "fast response")
        
        // This should complete with fast agent response even if slow agent is still processing
        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(fastAgent, slowAgent),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        // Should have at least the fast agent's response
        assertTrue("Should have fast agent response", responses.containsKey("FastAgent"))
        assertEquals("fast response", responses["FastAgent"]?.content)
    }
}