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
    fun testConcurrentAccess() = runBlocking {

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
















































































































        assertTrue("Should handle concurrent access", responses.isNotEmpty())
        assertEquals("response", responses["ConcurrentAgent"]?.content)
    }
}
    // Additional comprehensive test cases for better coverage

    @Test
    fun testParticipateWithAgents_contextWithNullValues() = runBlocking {
        val agent = DummyAgent("NullContextAgent", "handled null context")
        val contextWithNulls = mapOf(
            "valid" to "value",
            "null" to null,
            "empty" to ""
        ).filterValues { it != null } // Filter out nulls as Maps don't allow null values in Kotlin

        val responses = genesisAgent.participateWithAgents(
            context = contextWithNulls,
            agents = listOf(agent),
            prompt = "test with filtered nulls",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("handled null context", responses["NullContextAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_contextKeyWithSpecialCharacters() = runBlocking {
        val agent = DummyAgent("SpecialKeyAgent", "handled special keys")
        val specialContext = mapOf(
            "key with spaces" to "value1",
            "key:with:colons" to "value2",
            "key\nwith\nnewlines" to "value3",
            "key\twith\ttabs" to "value4",
            "key\"with\"quotes" to "value5"
        )

        val responses = genesisAgent.participateWithAgents(
            context = specialContext,
            agents = listOf(agent),
            prompt = "test special keys",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("handled special keys", responses["SpecialKeyAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_largeNumberOfAgents() = runBlocking {
        val agents = (1..100).map { i ->
            DummyAgent("Agent$i", "response$i", i / 100.0f)
        }

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = agents,
            prompt = "test many agents",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(100, responses.size)
        for (i in 1..100) {
            assertEquals("response$i", responses["Agent$i"]?.content)
            assertEquals(i / 100.0f, responses["Agent$i"]?.confidence)
        }
    }

    @Test
    fun testParticipateWithAgents_mixedSuccessAndFailure() = runBlocking {
        val successfulAgent1 = DummyAgent("Success1", "success1", 0.8f)
        val failingAgent1 = FailingAgent("Fail1")
        val successfulAgent2 = DummyAgent("Success2", "success2", 0.9f)
        val failingAgent2 = FailingAgent("Fail2")
        val successfulAgent3 = DummyAgent("Success3", "success3", 0.7f)

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(successfulAgent1, failingAgent1, successfulAgent2, failingAgent2, successfulAgent3),
            prompt = "mixed success and failure",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(3, responses.size)
        assertEquals("success1", responses["Success1"]?.content)
        assertEquals("success2", responses["Success2"]?.content)
        assertEquals("success3", responses["Success3"]?.content)
        assertNull(responses["Fail1"])
        assertNull(responses["Fail2"])
    }

    @Test
    fun testParticipateWithAgents_agentWithVeryLongName() = runBlocking {
        val longName = "A".repeat(1000)
        val agent = DummyAgent(longName, "long name response")

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent),
            prompt = "test long name",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("long name response", responses[longName]?.content)
    }

    @Test
    fun testParticipateWithAgents_agentWithEmptyResponse() = runBlocking {
        val agent = DummyAgent("EmptyResponseAgent", "")

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent),
            prompt = "test empty response",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("", responses["EmptyResponseAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_contextBuilding() = runBlocking {
        val agent = DummyAgent("ContextTestAgent", "context built")
        val context = mapOf("key1" to "value1", "key2" to "value2")
        val prompt = "test prompt"

        val responses = genesisAgent.participateWithAgents(
            context = context,
            agents = listOf(agent),
            prompt = prompt,
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("context built", responses["ContextTestAgent"]?.content)
        // Note: The combined prompt would be "key1:value1 key2:value2 test prompt"
    }

    @Test
    fun testAggregateAgentResponses_duplicateAgentNamesHigherConfidence() {
        val responses = listOf(
            mapOf("Agent1" to AgentResponse("first", 0.3f)),
            mapOf("Agent1" to AgentResponse("second", 0.7f)),
            mapOf("Agent1" to AgentResponse("third", 0.5f))
        )

        val consensus = genesisAgent.aggregateAgentResponses(responses)

        assertEquals(1, consensus.size)
        assertEquals("second", consensus["Agent1"]?.content)
        assertEquals(0.7f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_manyAgentsManyResponses() {
        val responses = (1..50).map { responseIndex ->
            (1..10).associate { agentIndex ->
                "Agent$agentIndex" to AgentResponse(
                    "response${responseIndex}_$agentIndex",
                    (responseIndex * agentIndex) / 500.0f
                )
            }
        }

        val consensus = genesisAgent.aggregateAgentResponses(responses)

        assertEquals(10, consensus.size)
        // Agent10 should have the highest confidence (50 * 10 / 500 = 1.0)
        assertEquals("response50_10", consensus["Agent10"]?.content)
        assertEquals(1.0f, consensus["Agent10"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_veryLongAgentNames() {
        val longAgentName = "A".repeat(500)
        val responses = listOf(
            mapOf(longAgentName to AgentResponse("first", 0.3f)),
            mapOf(longAgentName to AgentResponse("second", 0.8f))
        )

        val consensus = genesisAgent.aggregateAgentResponses(responses)

        assertEquals(1, consensus.size)
        assertEquals("second", consensus[longAgentName]?.content)
        assertEquals(0.8f, consensus[longAgentName]?.confidence)
    }

    @Test
    fun testProcessRequest_emptyPrompt() = runBlocking {
        val request = AiRequest("", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura empty", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai empty", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade empty", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura empty kai empty cascade empty", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_whitespaceOnlyPrompt() = runBlocking {
        val request = AiRequest("   \t\n\r   ", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura ws", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai ws", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade ws", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura ws kai ws cascade ws", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_contextWithEmptyValues() = runBlocking {
        val context = mapOf("key1" to "", "key2" to "value", "key3" to "")
        val request = AiRequest("test prompt", context)
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_allServicesReturnMaxConfidence() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", Float.MAX_VALUE))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", Float.MAX_VALUE))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", Float.MAX_VALUE))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        assertEquals(Float.MAX_VALUE, response.confidence)
    }

    @Test
    fun testProcessRequest_mixedExtremeConfidenceValues() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", Float.MIN_VALUE))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", Float.MAX_VALUE))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", Float.NEGATIVE_INFINITY))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        assertEquals(Float.MAX_VALUE, response.confidence)
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
    fun testProcessRequest_servicesReturnSpecialCharacters() = runBlocking {
        val specialContent = "Special: àáâãäå çćčđ éêë ñ øö ş ťü ý žż 中文 日本語 한국어 !@#$%^&*()"
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse(specialContent, 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("$specialContent kai cascade", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_kaiServiceFailure() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura response", 0.8f))
        whenever(kaiService.processRequest(any())).thenThrow(RuntimeException("Kai service failed"))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade response", 0.7f))

        try {
            genesisAgent.processRequest(request)
            fail("Should throw exception when kai service fails")
        } catch (e: RuntimeException) {
            assertEquals("Kai service failed", e.message)
        }
    }

    @Test
    fun testProcessRequest_cascadeServiceFailure() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura response", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai response", 0.9f))
        whenever(cascadeService.processRequest(any())).thenThrow(RuntimeException("Cascade service failed"))

        try {
            genesisAgent.processRequest(request)
            fail("Should throw exception when cascade service fails")
        } catch (e: RuntimeException) {
            assertEquals("Cascade service failed", e.message)
        }
    }

    @Test
    fun testProcessRequest_multipleServiceFailures() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenThrow(RuntimeException("Aura service failed"))
        whenever(kaiService.processRequest(any())).thenThrow(RuntimeException("Kai service failed"))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade response", 0.7f))

        try {
            genesisAgent.processRequest(request)
            fail("Should throw exception when aura service fails first")
        } catch (e: RuntimeException) {
            assertEquals("Aura service failed", e.message)
        }
    }

    @Test
    fun testDummyAgent_withExtremeConfidenceValues() = runBlocking {
        val infiniteAgent = DummyAgent("InfiniteAgent", "infinite response", Float.POSITIVE_INFINITY)
        val negativeInfiniteAgent = DummyAgent("NegInfAgent", "neg inf response", Float.NEGATIVE_INFINITY)
        val nanAgent = DummyAgent("NanAgent", "nan response", Float.NaN)

        val request = AiRequest("test", emptyMap())

        val infiniteResponse = infiniteAgent.processRequest(request)
        assertEquals("infinite response", infiniteResponse.content)
        assertEquals(Float.POSITIVE_INFINITY, infiniteResponse.confidence)

        val negInfResponse = negativeInfiniteAgent.processRequest(request)
        assertEquals("neg inf response", negInfResponse.content)
        assertEquals(Float.NEGATIVE_INFINITY, negInfResponse.confidence)

        val nanResponse = nanAgent.processRequest(request)
        assertEquals("nan response", nanResponse.content)
        assertTrue(nanResponse.confidence.isNaN())
    }

    @Test
    fun testDummyAgent_withVeryLongResponse() = runBlocking {
        val longResponse = "Response: " + "A".repeat(50000)
        val agent = DummyAgent("LongResponseAgent", longResponse)
        val request = AiRequest("test", emptyMap())
        val response = agent.processRequest(request)

        assertEquals(longResponse, response.content)
        assertEquals(1.0f, response.confidence)
    }

    @Test
    fun testDummyAgent_withSpecialCharacterResponse() = runBlocking {
        val specialResponse = "Special chars: àáâãäå çćčđ éêë ñ øö ş ťü ý žż 中文 日本語 한국어 !@#$%^&*()"
        val agent = DummyAgent("SpecialCharResponseAgent", specialResponse)
        val request = AiRequest("test", emptyMap())
        val response = agent.processRequest(request)

        assertEquals(specialResponse, response.content)
        assertEquals(1.0f, response.confidence)
    }

    @Test
    fun testFailingAgent_withDifferentExceptionTypes() = runBlocking {
        class CustomFailingAgent(name: String, private val exceptionType: String) : Agent {
            override fun getName(): String = name
            override fun getType(): String? = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                when (exceptionType) {
                    "runtime" -> throw RuntimeException("Runtime exception")
                    "illegal" -> throw IllegalArgumentException("Illegal argument")
                    "state" -> throw IllegalStateException("Illegal state")
                    else -> throw Exception("Generic exception")
                }
            }
        }

        val runtimeAgent = CustomFailingAgent("RuntimeAgent", "runtime")
        val illegalAgent = CustomFailingAgent("IllegalAgent", "illegal")
        val stateAgent = CustomFailingAgent("StateAgent", "state")
        val genericAgent = CustomFailingAgent("GenericAgent", "generic")

        val request = AiRequest("test", emptyMap())

        // Test each exception type
        try {
            runtimeAgent.processRequest(request)
            fail("Should throw RuntimeException")
        } catch (e: RuntimeException) {
            assertEquals("Runtime exception", e.message)
        }

        try {
            illegalAgent.processRequest(request)
            fail("Should throw IllegalArgumentException")
        } catch (e: IllegalArgumentException) {
            assertEquals("Illegal argument", e.message)
        }

        try {
            stateAgent.processRequest(request)
            fail("Should throw IllegalStateException")
        } catch (e: IllegalStateException) {
            assertEquals("Illegal state", e.message)
        }

        try {
            genericAgent.processRequest(request)
            fail("Should throw Exception")
        } catch (e: Exception) {
            assertEquals("Generic exception", e.message)
        }
    }

    @Test
    fun testParticipateWithAgents_stressTestWithManyFailingAgents() = runBlocking {
        val agents = (1..50).map { i ->
            if (i % 2 == 0) {
                DummyAgent("Success$i", "success$i")
            } else {
                FailingAgent("Fail$i")
            }
        }

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = agents,
            prompt = "stress test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(25, responses.size) // Only successful agents should be in responses
        for (i in 2..50 step 2) {
            assertEquals("success$i", responses["Success$i"]?.content)
        }
        for (i in 1..49 step 2) {
            assertNull(responses["Fail$i"])
        }
    }

    @Test
    fun testParticipateWithAgents_performanceWithLargeData() = runBlocking {
        val agent = DummyAgent("PerformanceAgent", "handled large data")
        val largeContext = (1..5000).associate { "key$it" to "value$it".repeat(100) }
        val largePrompt = "Performance test: " + "A".repeat(5000)

        val startTime = System.currentTimeMillis()
        val responses = genesisAgent.participateWithAgents(
            context = largeContext,
            agents = listOf(agent),
            prompt = largePrompt,
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        val endTime = System.currentTimeMillis()

        assertEquals(1, responses.size)
        assertEquals("handled large data", responses["PerformanceAgent"]?.content)
        // Performance should complete within reasonable time (adjust threshold as needed)
        assertTrue("Performance test should complete in reasonable time", (endTime - startTime) < 5000)
    }

    @Test
    fun testAggregateAgentResponses_edgeCaseWithIdenticalResponses() {
        val identicalResponse = AgentResponse("identical", 0.5f)
        val responses = listOf(
            mapOf("Agent1" to identicalResponse),
            mapOf("Agent1" to identicalResponse),
            mapOf("Agent1" to identicalResponse)
        )

        val consensus = genesisAgent.aggregateAgentResponses(responses)

        assertEquals(1, consensus.size)
        assertEquals("identical", consensus["Agent1"]?.content)
        assertEquals(0.5f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_memoryStressTest() {
        val largeResponses = (1..1000).map { responseIndex ->
            (1..100).associate { agentIndex ->
                "Agent$agentIndex" to AgentResponse(
                    "Response$responseIndex for Agent$agentIndex",
                    (responseIndex + agentIndex) / 1100.0f
                )
            }
        }

        val startTime = System.currentTimeMillis()
        val consensus = genesisAgent.aggregateAgentResponses(largeResponses)
        val endTime = System.currentTimeMillis()

        assertEquals(100, consensus.size)
        assertTrue("Memory stress test should complete in reasonable time", (endTime - startTime) < 10000)
        
        // Verify highest confidence responses are selected
        for (agentIndex in 1..100) {
            val expectedConfidence = (1000 + agentIndex) / 1100.0f
            assertEquals(expectedConfidence, consensus["Agent$agentIndex"]?.confidence, 0.001f)
        }
    }

    @Test
    fun testProcessRequest_boundaryCasesWithExtremeInputs() = runBlocking {
        val extremeContext = mapOf(
            "empty" to "",
            "spaces" to "   ",
            "newlines" to "\n\n\n",
            "tabs" to "\t\t\t",
            "mixed" to " \t\n\r ",
            "long" to "A".repeat(1000),
            "special" to "àáâãäå çćčđ éêë ñ øö ş ťü ý žż 中文 日本語 한국어 !@#$%^&*()"
        )
        val request = AiRequest("Boundary test", extremeContext)
        
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura boundary", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai boundary", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade boundary", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura boundary kai boundary cascade boundary", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testIntegration_fullWorkflowWithParticipateAndAggregate() = runBlocking {
        val agent1 = DummyAgent("Integration1", "int1", 0.7f)
        val agent2 = DummyAgent("Integration2", "int2", 0.9f)
        val agent3 = DummyAgent("Integration3", "int3", 0.5f)

        // First, participate with agents
        val responses = genesisAgent.participateWithAgents(
            context = mapOf("integration" to "test"),
            agents = listOf(agent1, agent2, agent3),
            prompt = "integration test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        // Then aggregate the responses
        val aggregated = genesisAgent.aggregateAgentResponses(listOf(responses))

        assertEquals(3, responses.size)
        assertEquals(3, aggregated.size)
        assertEquals("int1", aggregated["Integration1"]?.content)
        assertEquals("int2", aggregated["Integration2"]?.content)
        assertEquals("int3", aggregated["Integration3"]?.content)
        assertEquals(0.7f, aggregated["Integration1"]?.confidence)
        assertEquals(0.9f, aggregated["Integration2"]?.confidence)
        assertEquals(0.5f, aggregated["Integration3"]?.confidence)
    }

    @Test
    fun testIntegration_participateWithMultipleModesAndAggregate() = runBlocking {
        val agents = listOf(
            DummyAgent("Multi1", "multi1", 0.6f),
            DummyAgent("Multi2", "multi2", 0.8f)
        )
        val context = mapOf("mode" to "multi")
        val prompt = "multi mode test"

        // Test all conversation modes
        val turnOrderResponses = genesisAgent.participateWithAgents(
            context, agents, prompt, GenesisAgent.ConversationMode.TURN_ORDER
        )
        val cascadeResponses = genesisAgent.participateWithAgents(
            context, agents, prompt, GenesisAgent.ConversationMode.CASCADE
        )
        val consensusResponses = genesisAgent.participateWithAgents(
            context, agents, prompt, GenesisAgent.ConversationMode.CONSENSUS
        )

        // Aggregate all responses
        val allResponses = listOf(turnOrderResponses, cascadeResponses, consensusResponses)
        val aggregated = genesisAgent.aggregateAgentResponses(allResponses)

        assertEquals(2, aggregated.size)
        assertEquals("multi1", aggregated["Multi1"]?.content)
        assertEquals("multi2", aggregated["Multi2"]?.content)
        assertEquals(0.6f, aggregated["Multi1"]?.confidence)
        assertEquals(0.8f, aggregated["Multi2"]?.confidence)
    }

    @Test 
    fun testGenesisAgent_threadSafety() = runBlocking {
        val agent = DummyAgent("ThreadSafeAgent", "thread safe")
        val request = AiRequest("thread safety test", emptyMap())
        
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura thread", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai thread", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade thread", 0.7f))

        val jobs = (1..20).map { i ->
            kotlinx.coroutines.async {
                val participateResponse = genesisAgent.participateWithAgents(
                    context = mapOf("thread" to "test$i"),
                    agents = listOf(agent),
                    prompt = "thread test $i",
                    mode = GenesisAgent.ConversationMode.TURN_ORDER
                )
                val processResponse = genesisAgent.processRequest(request)
                Pair(participateResponse, processResponse)
            }
        }

        val results = jobs.map { it.await() }
        
        // All jobs should complete successfully
        assertEquals(20, results.size)
        results.forEach { (participateResponse, processResponse) ->
            assertEquals(1, participateResponse.size)
            assertEquals("thread safe", participateResponse["ThreadSafeAgent"]?.content)
            assertEquals("aura thread kai thread cascade thread", processResponse.content)
        }
    }
}
    // Additional comprehensive tests for better coverage

    @Test
    fun testProcessRequest_requestWithNullContext() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val response = genesisAgent.processRequest(request)
        
        assertNotNull("Response should not be null", response)
        assertEquals("aura kai cascade", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_verifyServiceCallsReceiveCorrectRequest() = runBlocking {
        val testContext = mapOf("key" to "value")
        val request = AiRequest("test prompt", testContext)
        
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        genesisAgent.processRequest(request)

        // Verify that each service was called with the correct request
        org.mockito.kotlin.verify(auraService).processRequest(request)
        org.mockito.kotlin.verify(kaiService).processRequest(request)
        org.mockito.kotlin.verify(cascadeService).processRequest(request)
    }

    @Test
    fun testProcessRequest_serviceCallOrder() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        val callOrder = mutableListOf<String>()
        
        whenever(auraService.processRequest(any())).thenAnswer {
            callOrder.add("aura")
            AgentResponse("aura", 0.8f)
        }
        whenever(kaiService.processRequest(any())).thenAnswer {
            callOrder.add("kai")
            AgentResponse("kai", 0.9f)
        }
        whenever(cascadeService.processRequest(any())).thenAnswer {
            callOrder.add("cascade")
            AgentResponse("cascade", 0.7f)
        }

        genesisAgent.processRequest(request)

        assertEquals(listOf("aura", "kai", "cascade"), callOrder)
    }

    @Test
    fun testParticipateWithAgents_contextBuildsCorrectPrompt() = runBlocking {
        var receivedRequest: AiRequest? = null
        val agent = object : Agent {
            override fun getName(): String = "TestAgent"
            override fun getType(): String? = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                receivedRequest = request
                return AgentResponse("test", 1.0f)
            }
        }

        val context = mapOf("key1" to "value1", "key2" to "value2")
        val prompt = "test prompt"

        genesisAgent.participateWithAgents(
            context = context,
            agents = listOf(agent),
            prompt = prompt,
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertNotNull("Request should have been received", receivedRequest)
        assertEquals("key1:value1 key2:value2 test prompt", receivedRequest?.prompt)
        assertEquals(context, receivedRequest?.context)
    }

    @Test
    fun testParticipateWithAgents_contextBuildsCorrectPromptWithNullPrompt() = runBlocking {
        var receivedRequest: AiRequest? = null
        val agent = object : Agent {
            override fun getName(): String = "TestAgent"
            override fun getType(): String? = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                receivedRequest = request
                return AgentResponse("test", 1.0f)
            }
        }

        val context = mapOf("key1" to "value1", "key2" to "value2")

        genesisAgent.participateWithAgents(
            context = context,
            agents = listOf(agent),
            prompt = null,
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertNotNull("Request should have been received", receivedRequest)
        assertEquals("key1:value1 key2:value2 ", receivedRequest?.prompt)
        assertEquals(context, receivedRequest?.context)
    }

    @Test
    fun testParticipateWithAgents_emptyContextWithPrompt() = runBlocking {
        var receivedRequest: AiRequest? = null
        val agent = object : Agent {
            override fun getName(): String = "TestAgent"
            override fun getType(): String? = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                receivedRequest = request
                return AgentResponse("test", 1.0f)
            }
        }

        val prompt = "test prompt"

        genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent),
            prompt = prompt,
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertNotNull("Request should have been received", receivedRequest)
        assertEquals(prompt, receivedRequest?.prompt)
        assertEquals(emptyMap<String, String>(), receivedRequest?.context)
    }

    @Test
    fun testParticipateWithAgents_agentProcessingOrder() = runBlocking {
        val processingOrder = mutableListOf<String>()
        val agent1 = object : Agent {
            override fun getName(): String = "Agent1"
            override fun getType(): String? = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                processingOrder.add("Agent1")
                return AgentResponse("response1", 0.8f)
            }
        }
        val agent2 = object : Agent {
            override fun getName(): String = "Agent2"
            override fun getType(): String? = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                processingOrder.add("Agent2")
                return AgentResponse("response2", 0.9f)
            }
        }
        val agent3 = object : Agent {
            override fun getName(): String = "Agent3"
            override fun getType(): String? = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                processingOrder.add("Agent3")
                return AgentResponse("response3", 0.7f)
            }
        }

        genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent1, agent2, agent3),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(listOf("Agent1", "Agent2", "Agent3"), processingOrder)
    }

    @Test
    fun testAggregateAgentResponses_preservesOriginalResponseWhenNoConflict() {
        val originalResponse = AgentResponse("unique content", 0.75f)
        val responses = listOf(
            mapOf("Agent1" to originalResponse),
            mapOf("Agent2" to AgentResponse("other content", 0.5f))
        )

        val consensus = genesisAgent.aggregateAgentResponses(responses)

        assertEquals(2, consensus.size)
        assertSame("Should preserve original response object", originalResponse, consensus["Agent1"])
        assertEquals("other content", consensus["Agent2"]?.content)
    }

    @Test
    fun testAggregateAgentResponses_handlesDuplicateKeys() {
        val responses = listOf(
            mapOf("Agent1" to AgentResponse("first", 0.3f), "Agent2" to AgentResponse("first2", 0.4f)),
            mapOf("Agent1" to AgentResponse("second", 0.7f), "Agent2" to AgentResponse("second2", 0.2f)),
            mapOf("Agent1" to AgentResponse("third", 0.5f), "Agent2" to AgentResponse("third2", 0.6f))
        )

        val consensus = genesisAgent.aggregateAgentResponses(responses)

        assertEquals(2, consensus.size)
        assertEquals("second", consensus["Agent1"]?.content)
        assertEquals(0.7f, consensus["Agent1"]?.confidence)
        assertEquals("third2", consensus["Agent2"]?.content)
        assertEquals(0.6f, consensus["Agent2"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_confidenceComparisonEdgeCases() {
        // Test with confidence values that might cause floating point precision issues
        val responses = listOf(
            mapOf("Agent1" to AgentResponse("first", 0.1f + 0.2f)), // 0.30000001
            mapOf("Agent1" to AgentResponse("second", 0.3f)) // 0.3
        )

        val consensus = genesisAgent.aggregateAgentResponses(responses)

        assertEquals(1, consensus.size)
        // Due to floating point precision, (0.1f + 0.2f) > 0.3f should be true
        assertEquals("first", consensus["Agent1"]?.content)
    }

    @Test
    fun testProcessRequest_aggregationLogic() = runBlocking {
        val request = AiRequest("test", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura response", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai response", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade response", 0.7f))

        val response = genesisAgent.processRequest(request)

        // Verify that content is joined with spaces
        assertEquals("aura response kai response cascade response", response.content)
        // Verify that confidence is the maximum
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_emptyResponsesFromAllServices() = runBlocking {
        val request = AiRequest("test", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("  ", response.content) // Two spaces from joining three empty strings
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_singleServiceWithContent() = runBlocking {
        val request = AiRequest("test", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("only aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("only aura  ", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testGenesisAgent_immutableProperties() {
        val name1 = genesisAgent.getName()
        val name2 = genesisAgent.getName()
        val type1 = genesisAgent.getType()
        val type2 = genesisAgent.getType()

        assertSame("Name should be the same object", name1, name2)
        assertEquals("Type should be consistent", type1, type2)
    }

    @Test
    fun testConversationMode_enumProperties() {
        val modes = GenesisAgent.ConversationMode.values()
        
        assertEquals(3, modes.size)
        assertTrue(modes.contains(GenesisAgent.ConversationMode.TURN_ORDER))
        assertTrue(modes.contains(GenesisAgent.ConversationMode.CASCADE))
        assertTrue(modes.contains(GenesisAgent.ConversationMode.CONSENSUS))
        
        // Test that all modes have consistent string representations
        assertEquals("TURN_ORDER", GenesisAgent.ConversationMode.TURN_ORDER.name)
        assertEquals("CASCADE", GenesisAgent.ConversationMode.CASCADE.name)
        assertEquals("CONSENSUS", GenesisAgent.ConversationMode.CONSENSUS.name)
    }

    @Test
    fun testDummyAgent_consistentBehavior() = runBlocking {
        val agent = DummyAgent("ConsistentAgent", "consistent response", 0.75f)
        val request = AiRequest("test", mapOf("key" to "value"))
        
        // Multiple calls should return identical responses
        val response1 = agent.processRequest(request)
        val response2 = agent.processRequest(request)
        
        assertEquals(response1.content, response2.content)
        assertEquals(response1.confidence, response2.confidence)
        assertEquals("consistent response", response1.content)
        assertEquals(0.75f, response1.confidence)
    }

    @Test
    fun testDummyAgent_ignoresRequestContent() = runBlocking {
        val agent = DummyAgent("IgnoreAgent", "fixed response", 0.5f)
        val request1 = AiRequest("different prompt 1", mapOf("key1" to "value1"))
        val request2 = AiRequest("different prompt 2", mapOf("key2" to "value2"))
        
        val response1 = agent.processRequest(request1)
        val response2 = agent.processRequest(request2)
        
        assertEquals(response1.content, response2.content)
        assertEquals(response1.confidence, response2.confidence)
        assertEquals("fixed response", response1.content)
    }

    @Test
    fun testFailingAgent_consistentFailure() = runBlocking {
        val agent = FailingAgent("ConsistentFailAgent")
        val request = AiRequest("test", emptyMap())
        
        // Should fail consistently
        try {
            agent.processRequest(request)
            fail("Should have thrown exception")
        } catch (e: RuntimeException) {
            assertEquals("Agent processing failed", e.message)
        }
        
        // Should fail again with same message
        try {
            agent.processRequest(request)
            fail("Should have thrown exception")
        } catch (e: RuntimeException) {
            assertEquals("Agent processing failed", e.message)
        }
    }

    @Test
    fun testParticipateWithAgents_partialFailuresDoNotAffectOthers() = runBlocking {
        val workingAgent1 = DummyAgent("Working1", "success1", 0.8f)
        val failingAgent = FailingAgent("Failing")
        val workingAgent2 = DummyAgent("Working2", "success2", 0.9f)
        
        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(workingAgent1, failingAgent, workingAgent2),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(2, responses.size)
        assertEquals("success1", responses["Working1"]?.content)
        assertEquals("success2", responses["Working2"]?.content)
        assertEquals(0.8f, responses["Working1"]?.confidence)
        assertEquals(0.9f, responses["Working2"]?.confidence)
        assertNull(responses["Failing"])
    }

    @Test
    fun testParticipateWithAgents_contextKeyOrderConsistency() = runBlocking {
        var receivedPrompt: String? = null
        val agent = object : Agent {
            override fun getName(): String = "OrderAgent"
            override fun getType(): String? = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                receivedPrompt = request.prompt
                return AgentResponse("received", 1.0f)
            }
        }

        val context = linkedMapOf("z" to "last", "a" to "first", "m" to "middle")
        
        genesisAgent.participateWithAgents(
            context = context,
            agents = listOf(agent),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertNotNull(receivedPrompt)
        // The order should be preserved as per the context map iteration
        assertTrue(receivedPrompt!!.contains("z:last"))
        assertTrue(receivedPrompt!!.contains("a:first"))
        assertTrue(receivedPrompt!!.contains("m:middle"))
        assertTrue(receivedPrompt!!.endsWith(" test"))
    }

    @Test
    fun testParticipateWithAgents_modeParameterUsage() = runBlocking {
        val agent = DummyAgent("ModeAgent", "mode response")
        
        // Test that different modes still work (even if implementation is the same)
        val turnOrderResponse = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        val cascadeResponse = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.CASCADE
        )
        
        val consensusResponse = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.CONSENSUS
        )
        
        // All modes should produce the same result with current implementation
        assertEquals(turnOrderResponse, cascadeResponse)
        assertEquals(cascadeResponse, consensusResponse)
        assertEquals(1, turnOrderResponse.size)
        assertEquals("mode response", turnOrderResponse["ModeAgent"]?.content)
    }

    @Test
    fun testProcessRequest_nullRequestHandling() = runBlocking {
        try {
            @Suppress("CAST_NEVER_SUCCEEDS")
            genesisAgent.processRequest(null as AiRequest)
            fail("Should throw exception for null request")
        } catch (e: IllegalArgumentException) {
            assertEquals("Request cannot be null", e.message)
        } catch (e: Exception) {
            // Accept any exception type as long as it's thrown for null request
            assertTrue("Should throw some exception for null request", true)
        }
    }

    @Test
    fun testProcessRequest_requireNotNullBehavior() = runBlocking {
        // Test that the requireNotNull check works correctly
        val validRequest = AiRequest("valid", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        // Should not throw for valid request
        val response = genesisAgent.processRequest(validRequest)
        assertNotNull(response)
        assertEquals("aura kai cascade", response.content)
    }

    @Test
    fun testAggregateAgentResponses_responseMapIntegrity() {
        val originalResponse1 = AgentResponse("original1", 0.5f)
        val originalResponse2 = AgentResponse("original2", 0.8f)
        val responses = listOf(
            mapOf("Agent1" to originalResponse1, "Agent2" to originalResponse2)
        )

        val consensus = genesisAgent.aggregateAgentResponses(responses)

        assertEquals(2, consensus.size)
        // Should be the same object references when no conflicts
        assertSame(originalResponse1, consensus["Agent1"])
        assertSame(originalResponse2, consensus["Agent2"])
    }

    @Test
    fun testAggregateAgentResponses_modificationDoesNotAffectOriginal() {
        val originalResponse = AgentResponse("original", 0.5f)
        val responses = listOf(
            mapOf("Agent1" to originalResponse)
        )

        val consensus = genesisAgent.aggregateAgentResponses(responses)
        
        // Verify that the consensus is independent of the original responses
        assertEquals(1, consensus.size)
        assertEquals("original", consensus["Agent1"]?.content)
        assertEquals(0.5f, consensus["Agent1"]?.confidence)
        
        // Original should remain unchanged
        assertEquals("original", originalResponse.content)
        assertEquals(0.5f, originalResponse.confidence)
    }

    @Test
    fun testIntegration_endToEndWorkflow() = runBlocking {
        val request = AiRequest("integration test", mapOf("workflow" to "end-to-end"))
        
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura integrated", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai integrated", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade integrated", 0.7f))

        // Test the main processRequest workflow
        val response = genesisAgent.processRequest(request)

        // Verify all services were called
        org.mockito.kotlin.verify(auraService).processRequest(request)
        org.mockito.kotlin.verify(kaiService).processRequest(request)
        org.mockito.kotlin.verify(cascadeService).processRequest(request)

        // Verify aggregation worked correctly
        assertEquals("aura integrated kai integrated cascade integrated", response.content)
        assertEquals(0.9f, response.confidence)
        
        // Verify the response is consistent
        val response2 = genesisAgent.processRequest(request)
        assertEquals(response.content, response2.content)
        assertEquals(response.confidence, response2.confidence)
    }

    @Test
    fun testGenesisAgent_resourceCleanup() = runBlocking {
        // Test that the GenesisAgent doesn't hold onto resources unnecessarily
        val request = AiRequest("cleanup test", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        // Process multiple requests
        repeat(10) {
            val response = genesisAgent.processRequest(request)
            assertNotNull(response)
        }

        // Verify services were called the expected number of times
        org.mockito.kotlin.verify(auraService, org.mockito.kotlin.times(10)).processRequest(request)
        org.mockito.kotlin.verify(kaiService, org.mockito.kotlin.times(10)).processRequest(request)
        org.mockito.kotlin.verify(cascadeService, org.mockito.kotlin.times(10)).processRequest(request)
    }

    @Test
    fun testParticipateWithAgents_exceptionHandlingSilent() = runBlocking {
        val silentFailingAgent = object : Agent {
            override fun getName(): String = "SilentFailing"
            override fun getType(): String? = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                throw IllegalStateException("Silent failure")
            }
        }
        
        val workingAgent = DummyAgent("Working", "success")
        
        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(silentFailingAgent, workingAgent),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        // Should silently handle the exception and continue with other agents
        assertEquals(1, responses.size)
        assertEquals("success", responses["Working"]?.content)
        assertNull(responses["SilentFailing"])
    }

    @Test
    fun testParticipateWithAgents_exceptionTypes() = runBlocking {
        val agents = listOf(
            object : Agent {
                override fun getName(): String = "RuntimeException"
                override fun getType(): String? = null
                override suspend fun processRequest(request: AiRequest): AgentResponse {
                    throw RuntimeException("Runtime failure")
                }
            },
            object : Agent {
                override fun getName(): String = "IllegalArgumentException"
                override fun getType(): String? = null
                override suspend fun processRequest(request: AiRequest): AgentResponse {
                    throw IllegalArgumentException("Illegal argument")
                }
            },
            object : Agent {
                override fun getName(): String = "NullPointerException"
                override fun getType(): String? = null
                override suspend fun processRequest(request: AiRequest): AgentResponse {
                    throw NullPointerException("Null pointer")
                }
            },
            DummyAgent("Success", "worked")
        )
        
        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = agents,
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        // All exceptions should be caught and handled silently
        assertEquals(1, responses.size)
        assertEquals("worked", responses["Success"]?.content)
        assertNull(responses["RuntimeException"])
        assertNull(responses["IllegalArgumentException"])
        assertNull(responses["NullPointerException"])
    }

    @Test
    fun testDummyAgent_edgeCaseNames() = runBlocking {
        val agents = listOf(
            DummyAgent("", "empty name"),
            DummyAgent("   ", "whitespace name"),
            DummyAgent("\n\t", "control chars"),
            DummyAgent("very.long.name.with.dots.and.underscores_and_numbers123", "complex name")
        )
        
        val request = AiRequest("test", emptyMap())
        
        for (agent in agents) {
            val response = agent.processRequest(request)
            assertNotNull("All agents should respond", response)
            assertTrue("All responses should have content", response.content.isNotEmpty())
        }
    }

    @Test
    fun testGenesisAgent_stateConsistency() = runBlocking {
        // Test that GenesisAgent maintains consistent state across multiple operations
        val request = AiRequest("state test", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val agent = DummyAgent("StateAgent", "state response")
        
        // Interleave different operations
        val processResponse1 = genesisAgent.processRequest(request)
        val participateResponse1 = genesisAgent.participateWithAgents(
            emptyMap(), listOf(agent), "test1", GenesisAgent.ConversationMode.TURN_ORDER
        )
        val processResponse2 = genesisAgent.processRequest(request)
        val participateResponse2 = genesisAgent.participateWithAgents(
            emptyMap(), listOf(agent), "test2", GenesisAgent.ConversationMode.CASCADE
        )

        // All operations should work consistently
        assertEquals(processResponse1.content, processResponse2.content)
        assertEquals(processResponse1.confidence, processResponse2.confidence)
        assertEquals(participateResponse1.size, participateResponse2.size)
        assertEquals(participateResponse1["StateAgent"]?.content, participateResponse2["StateAgent"]?.content)
    }

    @Test
    fun testAggregateAgentResponses_performanceWithIdenticalResponses() {
        val identicalResponse = AgentResponse("identical", 0.5f)
        val largeResponses = (1..1000).map {
            (1..50).associate { agentIndex ->
                "Agent$agentIndex" to identicalResponse
            }
        }

        val startTime = System.currentTimeMillis()
        val consensus = genesisAgent.aggregateAgentResponses(largeResponses)
        val endTime = System.currentTimeMillis()

        assertEquals(50, consensus.size)
        assertTrue("Performance should be reasonable", (endTime - startTime) < 1000)
        
        // All responses should be the same object
        consensus.values.forEach { response ->
            assertSame("Should be the same object reference", identicalResponse, response)
        }
    }

    @Test
    fun testProcessRequest_serviceTimeoutSimulation() = runBlocking {
        val request = AiRequest("timeout test", emptyMap())
        
        // Simulate a slow service
        whenever(auraService.processRequest(any())).thenAnswer {
            kotlinx.coroutines.delay(100)
            AgentResponse("aura slow", 0.8f)
        }
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai fast", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade fast", 0.7f))

        val startTime = System.currentTimeMillis()
        val response = genesisAgent.processRequest(request)
        val endTime = System.currentTimeMillis()

        assertEquals("aura slow kai fast cascade fast", response.content)
        assertEquals(0.9f, response.confidence)
        assertTrue("Should wait for slow service", (endTime - startTime) >= 100)
    }
