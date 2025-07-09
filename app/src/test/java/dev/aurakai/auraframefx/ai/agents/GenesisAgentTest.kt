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
import kotlinx.coroutines.async
import java.util.*

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
    fun testParticipateWithAgents_contextKeyOrdering() = runBlocking {
        val agent = DummyAgent("OrderingAgent", "ordered response")
        val orderedContext = linkedMapOf(
            "z_key" to "z_value",
            "a_key" to "a_value",
            "m_key" to "m_value"
        )

        val responses = genesisAgent.participateWithAgents(
            context = orderedContext,
            agents = listOf(agent),
            prompt = "test ordering",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("ordered response", responses["OrderingAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_contextWithNullValues() = runBlocking {
        val agent = DummyAgent("NullValueAgent", "handled nulls")
        val contextWithBlanks = mapOf(
            "normal" to "value",
            "blank" to "",
            "spaces" to "   "
        )

        val responses = genesisAgent.participateWithAgents(
            context = contextWithBlanks,
            agents = listOf(agent),
            prompt = "test with blanks",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("handled nulls", responses["NullValueAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_unicodeInContext() = runBlocking {
        val agent = DummyAgent("UnicodeAgent", "unicode handled")
        val unicodeContext = mapOf(
            "emoji" to "😀🎉🚀",
            "chinese" to "你好世界",
            "arabic" to "مرحبا بالعالم",
            "russian" to "Привет мир"
        )

        val responses = genesisAgent.participateWithAgents(
            context = unicodeContext,
            agents = listOf(agent),
            prompt = "unicode test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("unicode handled", responses["UnicodeAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_promptBuilding() = runBlocking {
        val agent = DummyAgent("PromptAgent", "prompt built")
        val context = mapOf("key1" to "value1", "key2" to "value2")

        val responses = genesisAgent.participateWithAgents(
            context = context,
            agents = listOf(agent),
            prompt = "base prompt",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("prompt built", responses["PromptAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_promptBuildingWithNoPrompt() = runBlocking {
        val agent = DummyAgent("NoPromptAgent", "no prompt response")
        val context = mapOf("context" to "only")

        val responses = genesisAgent.participateWithAgents(
            context = context,
            agents = listOf(agent),
            prompt = null,
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("no prompt response", responses["NoPromptAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_mixedAgentTypes() = runBlocking {
        val dummyAgent = DummyAgent("Dummy", "dummy response")
        val failingAgent = FailingAgent("Failing")
        val anotherDummy = DummyAgent("Another", "another response")

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(dummyAgent, failingAgent, anotherDummy),
            prompt = "mixed test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(2, responses.size)
        assertEquals("dummy response", responses["Dummy"]?.content)
        assertEquals("another response", responses["Another"]?.content)
        assertNull(responses["Failing"])
    }

    @Test
    fun testParticipateWithAgents_agentProcessingOrder() = runBlocking {
        val responses = mutableListOf<String>()
        val agent1 = object : Agent {
            override fun getName() = "Agent1"
            override fun getType() = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                responses.add("Agent1")
                return AgentResponse("response1", 1.0f)
            }
        }
        val agent2 = object : Agent {
            override fun getName() = "Agent2"
            override fun getType() = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                responses.add("Agent2")
                return AgentResponse("response2", 1.0f)
            }
        }

        val agentResponses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent1, agent2),
            prompt = "order test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(2, agentResponses.size)
        assertEquals(listOf("Agent1", "Agent2"), responses)
    }

    @Test
    fun testAggregateAgentResponses_multipleResponsesPerAgent() {
        val responses = listOf(
            mapOf("Agent1" to AgentResponse("resp1", 0.1f)),
            mapOf("Agent1" to AgentResponse("resp2", 0.2f)),
            mapOf("Agent1" to AgentResponse("resp3", 0.3f)),
            mapOf("Agent1" to AgentResponse("resp4", 0.4f)),
            mapOf("Agent1" to AgentResponse("resp5", 0.5f))
        )

        val consensus = genesisAgent.aggregateAgentResponses(responses)

        assertEquals(1, consensus.size)
        assertEquals("resp5", consensus["Agent1"]?.content)
        assertEquals(0.5f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_manyAgents() {
        val responses = listOf(
            (1..100).associate { i -> "Agent$i" to AgentResponse("response$i", i / 100.0f) }
        )

        val consensus = genesisAgent.aggregateAgentResponses(responses)

        assertEquals(100, consensus.size)
        assertEquals("response100", consensus["Agent100"]?.content)
        assertEquals(1.0f, consensus["Agent100"]?.confidence)
        assertEquals("response1", consensus["Agent1"]?.content)
        assertEquals(0.01f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_identicalConfidencesSameAgent() {
        val responses = listOf(
            mapOf("Agent1" to AgentResponse("first", 0.5f)),
            mapOf("Agent1" to AgentResponse("second", 0.5f)),
            mapOf("Agent1" to AgentResponse("third", 0.5f))
        )

        val consensus = genesisAgent.aggregateAgentResponses(responses)

        assertEquals(1, consensus.size)
        assertEquals(0.5f, consensus["Agent1"]?.confidence)
        // Content should be one of the responses, but we can't predict which due to map ordering
        assertTrue(
            consensus["Agent1"]?.content in listOf("first", "second", "third")
        )
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
    fun testProcessRequest_whitespacePrompt() = runBlocking {
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
        val context = mapOf("key1" to "", "key2" to "value2", "key3" to "")
        val request = AiRequest("test prompt", context)
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura ctx", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai ctx", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade ctx", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura ctx kai ctx cascade ctx", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_duplicateServiceResponses() = runBlocking {
        val request = AiRequest("test", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("duplicate", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("duplicate", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("duplicate", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertEquals("duplicate duplicate duplicate", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_veryLongServiceResponses() = runBlocking {
        val longResponse = "A".repeat(10000)
        val request = AiRequest("test", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse(longResponse, 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("short", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("normal", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertTrue("Response should contain long content", response.content.contains(longResponse))
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_negativeConfidenceValues() = runBlocking {
        val request = AiRequest("test", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", -0.5f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", -0.3f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", -0.8f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        assertEquals(-0.3f, response.confidence)
    }

    @Test
    fun testProcessRequest_mixedConfidenceValues() = runBlocking {
        val request = AiRequest("test", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", Float.NEGATIVE_INFINITY))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", Float.POSITIVE_INFINITY))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.5f))

        val response = genesisAgent.processRequest(request)

        assertEquals("aura kai cascade", response.content)
        assertEquals(Float.POSITIVE_INFINITY, response.confidence)
    }

    @Test
    fun testConversationMode_enumProperties() {
        val modes = GenesisAgent.ConversationMode.values()
        assertEquals(3, modes.size)
        
        // Test that each mode is unique
        val uniqueModes = modes.toSet()
        assertEquals(modes.size, uniqueModes.size)
        
        // Test string representation
        assertEquals("TURN_ORDER", GenesisAgent.ConversationMode.TURN_ORDER.toString())
        assertEquals("CASCADE", GenesisAgent.ConversationMode.CASCADE.toString())
        assertEquals("CONSENSUS", GenesisAgent.ConversationMode.CONSENSUS.toString())
    }

    @Test
    fun testAgentInterface_defaultMethods() {
        // Test that Agent interface methods work as expected
        val testAgent = object : Agent {
            override fun getName() = "TestInterface"
            override fun getType() = "TestType"
            override suspend fun processRequest(request: AiRequest) = AgentResponse("test", 1.0f)
        }

        assertEquals("TestInterface", testAgent.getName())
        assertEquals("TestType", testAgent.getType())
    }

    @Test
    fun testDummyAgent_constructorDefaults() {
        val agentWithDefaults = DummyAgent("DefaultAgent", "default response")
        assertEquals("DefaultAgent", agentWithDefaults.getName())
        assertEquals("default response", agentWithDefaults.processRequest(AiRequest("test", emptyMap())).content)
        assertEquals(1.0f, agentWithDefaults.processRequest(AiRequest("test", emptyMap())).confidence)
    }

    @Test
    fun testDummyAgent_allConstructorParams() = runBlocking {
        val agent = DummyAgent("FullAgent", "full response", 0.75f)
        val response = agent.processRequest(AiRequest("test", emptyMap()))
        
        assertEquals("FullAgent", agent.getName())
        assertEquals("full response", response.content)
        assertEquals(0.75f, response.confidence)
    }

    @Test 
    fun testFailingAgent_exceptionMessage() = runBlocking {
        val agent = FailingAgent("FailAgent")
        val request = AiRequest("test", emptyMap())
        
        try {
            agent.processRequest(request)
            fail("Should have thrown RuntimeException")
        } catch (e: RuntimeException) {
            assertEquals("Agent processing failed", e.message)
            assertTrue(e.javaClass == RuntimeException::class.java)
        }
    }

    @Test
    fun testParticipateWithAgents_exceptionHandling() = runBlocking {
        val throwingAgent = object : Agent {
            override fun getName() = "ThrowingAgent"
            override fun getType() = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                throw IllegalStateException("Custom exception")
            }
        }
        val workingAgent = DummyAgent("WorkingAgent", "success")

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(throwingAgent, workingAgent),
            prompt = "exception test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("success", responses["WorkingAgent"]?.content)
        assertNull(responses["ThrowingAgent"])
    }

    @Test
    fun testParticipateWithAgents_nullResponseHandling() = runBlocking {
        val nullAgent = object : Agent {
            override fun getName() = "NullAgent"
            override fun getType() = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                return AgentResponse("", 0.0f)
            }
        }

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(nullAgent),
            prompt = "null test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("", responses["NullAgent"]?.content)
        assertEquals(0.0f, responses["NullAgent"]?.confidence)
    }

    @Test
    fun testGenesisAgent_threadSafety() = runBlocking {
        val request = AiRequest("thread test", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val jobs = (1..50).map {
            kotlinx.coroutines.async {
                genesisAgent.processRequest(request)
            }
        }

        val results = jobs.map { it.await() }
        assertEquals(50, results.size)
        results.forEach { response ->
            assertEquals("aura kai cascade", response.content)
            assertEquals(0.9f, response.confidence)
        }
    }

    @Test
    fun testAggregateAgentResponses_edgeCaseConfidenceComparison() {
        val responses = listOf(
            mapOf("Agent1" to AgentResponse("neg_inf", Float.NEGATIVE_INFINITY)),
            mapOf("Agent1" to AgentResponse("pos_inf", Float.POSITIVE_INFINITY)),
            mapOf("Agent1" to AgentResponse("nan", Float.NaN)),
            mapOf("Agent1" to AgentResponse("min", Float.MIN_VALUE)),
            mapOf("Agent1" to AgentResponse("max", Float.MAX_VALUE))
        )

        val consensus = genesisAgent.aggregateAgentResponses(responses)
        
        assertEquals(1, consensus.size)
        assertNotNull(consensus["Agent1"])
        // The specific result depends on Float comparison implementation
        assertTrue(consensus["Agent1"]?.content in listOf("neg_inf", "pos_inf", "nan", "min", "max"))
    }

    @Test
    fun testProcessRequest_aggregationBehavior() = runBlocking {
        val request = AiRequest("aggregation test", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.5f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.8f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.3f))

        val response = genesisAgent.processRequest(request)

        // The aggregation should pick the highest confidence (0.8f from kai)
        assertEquals("aura kai cascade", response.content)
        assertEquals(0.8f, response.confidence)
    }

    @Test
    fun testProcessRequest_serviceResponseOrder() = runBlocking {
        val request = AiRequest("order test", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("FIRST", 0.5f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("SECOND", 0.6f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("THIRD", 0.4f))

        val response = genesisAgent.processRequest(request)

        // Content should maintain the order: Aura, Kai, Cascade
        assertEquals("FIRST SECOND THIRD", response.content)
        assertEquals(0.6f, response.confidence)
    }
}
}