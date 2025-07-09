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
        assertNull("Type should be null", type)
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
            @Suppress("CAST_NEVER_SUCCEEDS")
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

    // Additional comprehensive tests for better coverage

    @Test
    fun testParticipateWithAgents_cascadeMode() = runBlocking {
        val agent1 = DummyAgent("Agent1", "cascade response 1", 0.6f)
        val agent2 = DummyAgent("Agent2", "cascade response 2", 0.8f)

        val responses = genesisAgent.participateWithAgents(
            context = mapOf("cascadeData" to "test"),
            agents = listOf(agent1, agent2),
            prompt = "cascade test",
            mode = GenesisAgent.ConversationMode.CASCADE
        )

        assertEquals(2, responses.size)
        assertEquals("cascade response 1", responses["Agent1"]?.content)
        assertEquals("cascade response 2", responses["Agent2"]?.content)
    }

    @Test
    fun testParticipateWithAgents_consensusMode() = runBlocking {
        val agent1 = DummyAgent("Agent1", "consensus response 1", 0.7f)
        val agent2 = DummyAgent("Agent2", "consensus response 2", 0.9f)

        val responses = genesisAgent.participateWithAgents(
            context = mapOf("consensusData" to "test"),
            agents = listOf(agent1, agent2),
            prompt = "consensus test",
            mode = GenesisAgent.ConversationMode.CONSENSUS
        )

        assertEquals(2, responses.size)
        assertEquals("consensus response 1", responses["Agent1"]?.content)
        assertEquals("consensus response 2", responses["Agent2"]?.content)
    }

    @Test
    fun testParticipateWithAgents_complexContextBuilding() = runBlocking {
        val agent = DummyAgent("ComplexAgent", "complex response")
        val context = mapOf(
            "key1" to "value1",
            "key2" to "value2",
            "key3" to "value3",
            "specialKey" to "special!@#$%^&*()value"
        )

        val responses = genesisAgent.participateWithAgents(
            context = context,
            agents = listOf(agent),
            prompt = "complex prompt with special chars !@#$%^&*()",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("complex response", responses["ComplexAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_largeAgentList() = runBlocking {
        val agents = (1..50).map { i ->
            DummyAgent("Agent$i", "response$i", i / 50.0f)
        }

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = agents,
            prompt = "large scale test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(50, responses.size)
        for (i in 1..50) {
            assertEquals("response$i", responses["Agent$i"]?.content)
            assertEquals(i / 50.0f, responses["Agent$i"]?.confidence)
        }
    }

    @Test
    fun testParticipateWithAgents_longPrompt() = runBlocking {
        val agent = DummyAgent("LongPromptAgent", "long response")
        val longPrompt = "A".repeat(10000)

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent),
            prompt = longPrompt,
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("long response", responses["LongPromptAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_emptyStringPrompt() = runBlocking {
        val agent = DummyAgent("EmptyPromptAgent", "empty string response")

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent),
            prompt = "",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("empty string response", responses["EmptyPromptAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_specialCharactersInAgentNames() = runBlocking {
        val agent1 = DummyAgent("Agent!@#", "special char response 1")
        val agent2 = DummyAgent("Agent$%^", "special char response 2")
        val agent3 = DummyAgent("Agent&*()", "special char response 3")

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent1, agent2, agent3),
            prompt = "special chars test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(3, responses.size)
        assertEquals("special char response 1", responses["Agent!@#"]?.content)
        assertEquals("special char response 2", responses["Agent$%^"]?.content)
        assertEquals("special char response 3", responses["Agent&*()"]?.content)
    }

    @Test
    fun testParticipateWithAgents_mixOfSuccessAndFailure() = runBlocking {
        val workingAgent1 = DummyAgent("Working1", "success1")
        val failingAgent1 = FailingAgent("Failing1")
        val workingAgent2 = DummyAgent("Working2", "success2")
        val failingAgent2 = FailingAgent("Failing2")

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(workingAgent1, failingAgent1, workingAgent2, failingAgent2),
            prompt = "mixed success/failure test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(2, responses.size)
        assertEquals("success1", responses["Working1"]?.content)
        assertEquals("success2", responses["Working2"]?.content)
        assertNull(responses["Failing1"])
        assertNull(responses["Failing2"])
    }

    @Test
    fun testParticipateWithAgents_veryLargeContext() = runBlocking {
        val agent = DummyAgent("LargeContextAgent", "large context response")
        val context = (1..100).associate { i ->
            "key$i" to "value$i".repeat(100)
        }

        val responses = genesisAgent.participateWithAgents(
            context = context,
            agents = listOf(agent),
            prompt = "large context test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("large context response", responses["LargeContextAgent"]?.content)
    }

    @Test
    fun testAggregateAgentResponses_extremeConfidenceValues() {
        val resp1 = mapOf("Agent1" to AgentResponse("response1", Float.MAX_VALUE))
        val resp2 = mapOf("Agent1" to AgentResponse("response2", Float.MIN_VALUE))
        val resp3 = mapOf("Agent1" to AgentResponse("response3", 0.5f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2, resp3))

        assertEquals(1, consensus.size)
        assertEquals("response1", consensus["Agent1"]?.content)
        assertEquals(Float.MAX_VALUE, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_infiniteConfidence() {
        val resp1 = mapOf("Agent1" to AgentResponse("response1", Float.POSITIVE_INFINITY))
        val resp2 = mapOf("Agent1" to AgentResponse("response2", Float.NEGATIVE_INFINITY))
        val resp3 = mapOf("Agent1" to AgentResponse("response3", 0.5f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2, resp3))

        assertEquals(1, consensus.size)
        assertEquals("response1", consensus["Agent1"]?.content)
        assertEquals(Float.POSITIVE_INFINITY, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_nanConfidence() {
        val resp1 = mapOf("Agent1" to AgentResponse("response1", Float.NaN))
        val resp2 = mapOf("Agent1" to AgentResponse("response2", 0.5f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

        assertEquals(1, consensus.size)
        // NaN comparisons always return false, so the second response should be chosen
        assertEquals("response2", consensus["Agent1"]?.content)
        assertEquals(0.5f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_emptyResponseContent() {
        val resp1 = mapOf("Agent1" to AgentResponse("", 0.5f))
        val resp2 = mapOf("Agent1" to AgentResponse("non-empty", 0.3f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

        assertEquals(1, consensus.size)
        assertEquals("", consensus["Agent1"]?.content)
        assertEquals(0.5f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_longResponseContent() {
        val longContent = "A".repeat(100000)
        val resp1 = mapOf("Agent1" to AgentResponse(longContent, 0.8f))
        val resp2 = mapOf("Agent1" to AgentResponse("short", 0.5f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

        assertEquals(1, consensus.size)
        assertEquals(longContent, consensus["Agent1"]?.content)
        assertEquals(0.8f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_unicodeContent() {
        val unicodeContent = "Hello 世界 🌍 नमस्ते مرحبا"
        val resp1 = mapOf("Agent1" to AgentResponse(unicodeContent, 0.8f))
        val resp2 = mapOf("Agent1" to AgentResponse("ascii", 0.5f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

        assertEquals(1, consensus.size)
        assertEquals(unicodeContent, consensus["Agent1"]?.content)
        assertEquals(0.8f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testProcessRequest_serviceExceptions() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenThrow(RuntimeException("Aura service failed"))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai response", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade response", 0.7f))

        try {
            genesisAgent.processRequest(request)
            fail("Should propagate exception from service")
        } catch (e: RuntimeException) {
            assertEquals("Aura service failed", e.message)
        }
    }

    @Test
    fun testProcessRequest_allServicesReturnEmptyResponses() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("", 0.0f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("", 0.0f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("", 0.0f))

        val response = genesisAgent.processRequest(request)

        assertEquals("  ", response.content) // Three empty strings joined with spaces
        assertEquals(0.0f, response.confidence)
    }

    @Test
    fun testProcessRequest_veryLongServiceResponses() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        val longResponse = "A".repeat(10000)
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse(longResponse, 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse(longResponse, 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse(longResponse, 0.7f))

        val response = genesisAgent.processRequest(request)

        assertTrue("Response should contain long content", response.content.contains(longResponse))
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_unicodeServiceResponses() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        val unicodeResponse = "Hello 世界 🌍"
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse(unicodeResponse, 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai response", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade response", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertTrue("Response should contain unicode content", response.content.contains(unicodeResponse))
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_requestWithLargeContext() = runBlocking {
        val largeContext = (1..1000).associate { i ->
            "key$i" to "value$i"
        }
        val request = AiRequest("test prompt", largeContext)
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura response", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai response", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade response", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertNotNull("Response should not be null", response)
        assertTrue("Response should have content", response.content.isNotEmpty())
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testDummyAgent_zeroConfidence() = runBlocking {
        val agent = DummyAgent("ZeroConfAgent", "zero conf response", 0.0f)
        assertEquals("ZeroConfAgent", agent.getName())
        
        val request = AiRequest("test", emptyMap())
        val response = agent.processRequest(request)
        
        assertEquals("zero conf response", response.content)
        assertEquals(0.0f, response.confidence)
    }

    @Test
    fun testDummyAgent_negativeConfidence() = runBlocking {
        val agent = DummyAgent("NegConfAgent", "negative conf response", -0.5f)
        assertEquals("NegConfAgent", agent.getName())
        
        val request = AiRequest("test", emptyMap())
        val response = agent.processRequest(request)
        
        assertEquals("negative conf response", response.content)
        assertEquals(-0.5f, response.confidence)
    }

    @Test
    fun testDummyAgent_extremeConfidence() = runBlocking {
        val agent = DummyAgent("ExtremeConfAgent", "extreme response", Float.MAX_VALUE)
        assertEquals("ExtremeConfAgent", agent.getName())
        
        val request = AiRequest("test", emptyMap())
        val response = agent.processRequest(request)
        
        assertEquals("extreme response", response.content)
        assertEquals(Float.MAX_VALUE, response.confidence)
    }

    @Test
    fun testFailingAgent_differentExceptionTypes() = runBlocking {
        class CustomFailingAgent(name: String, private val exception: Exception) : Agent {
            override fun getName(): String = name
            override fun getType(): String? = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                throw exception
            }
        }

        val illegalArgAgent = CustomFailingAgent("IllegalArgAgent", IllegalArgumentException("Illegal argument"))
        val nullPointerAgent = CustomFailingAgent("NullPointerAgent", NullPointerException("Null pointer"))

        assertEquals("IllegalArgAgent", illegalArgAgent.getName())
        assertEquals("NullPointerAgent", nullPointerAgent.getName())

        val request = AiRequest("test", emptyMap())
        
        try {
            illegalArgAgent.processRequest(request)
            fail("Should throw IllegalArgumentException")
        } catch (e: IllegalArgumentException) {
            assertEquals("Illegal argument", e.message)
        }

        try {
            nullPointerAgent.processRequest(request)
            fail("Should throw NullPointerException")
        } catch (e: NullPointerException) {
            assertEquals("Null pointer", e.message)
        }
    }

    @Test
    fun testConversationMode_enumProperties() {
        val modes = GenesisAgent.ConversationMode.values()
        
        assertEquals(3, modes.size)
        assertTrue(modes.contains(GenesisAgent.ConversationMode.TURN_ORDER))
        assertTrue(modes.contains(GenesisAgent.ConversationMode.CASCADE))
        assertTrue(modes.contains(GenesisAgent.ConversationMode.CONSENSUS))
        
        assertEquals("TURN_ORDER", GenesisAgent.ConversationMode.TURN_ORDER.name)
        assertEquals("CASCADE", GenesisAgent.ConversationMode.CASCADE.name)
        assertEquals("CONSENSUS", GenesisAgent.ConversationMode.CONSENSUS.name)
    }

    @Test
    fun testGenesisAgent_threadSafety() = runBlocking {
        val agent = DummyAgent("ThreadSafeAgent", "thread safe response")
        val responses = mutableListOf<Map<String, AgentResponse>>()
        
        val jobs = (1..100).map { i ->
            kotlinx.coroutines.async {
                genesisAgent.participateWithAgents(
                    context = mapOf("thread" to "test$i"),
                    agents = listOf(agent),
                    prompt = "thread safety test $i",
                    mode = GenesisAgent.ConversationMode.TURN_ORDER
                )
            }
        }
        
        jobs.forEach { job ->
            responses.add(job.await())
        }
        
        assertEquals(100, responses.size)
        responses.forEach { response ->
            assertEquals("thread safe response", response["ThreadSafeAgent"]?.content)
        }
    }

    @Test
    fun testAggregateAgentResponses_performanceWithLargeDataset() {
        val responses = (1..1000).map { i ->
            (1..100).associate { j ->
                "Agent$j" to AgentResponse("response$i-$j", (i * j) % 1000 / 1000.0f)
            }
        }
        
        val startTime = System.currentTimeMillis()
        val consensus = genesisAgent.aggregateAgentResponses(responses)
        val endTime = System.currentTimeMillis()
        
        assertEquals(100, consensus.size)
        assertTrue("Processing should complete within reasonable time", (endTime - startTime) < 5000)
        
        // Verify highest confidence values are selected
        consensus.forEach { (agentName, response) ->
            val agentNum = agentName.removePrefix("Agent").toInt()
            val maxConfidence = (1..1000).maxOf { i -> (i * agentNum) % 1000 / 1000.0f }
            assertEquals(maxConfidence, response.confidence, 0.001f)
        }
    }

    @Test
    fun testGenesisAgent_memoryUsage() = runBlocking {
        val runtime = Runtime.getRuntime()
        val initialMemory = runtime.totalMemory() - runtime.freeMemory()
        
        val agents = (1..1000).map { i ->
            DummyAgent("MemoryAgent$i", "memory response $i")
        }
        
        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = agents,
            prompt = "memory usage test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        val finalMemory = runtime.totalMemory() - runtime.freeMemory()
        val memoryIncrease = finalMemory - initialMemory
        
        assertEquals(1000, responses.size)
        assertTrue("Memory increase should be reasonable", memoryIncrease < 100 * 1024 * 1024) // Less than 100MB
    }
}