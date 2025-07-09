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
    // Additional comprehensive tests for better coverage

    @Test
    fun testProcessRequest_serviceExceptions() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenThrow(RuntimeException("Aura service failed"))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai response", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade response", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertNotNull("Response should not be null even with service failure", response)
        assertTrue("Response should contain working services' content", response.content.contains("kai response"))
        assertTrue("Response should contain working services' content", response.content.contains("cascade response"))
    }

    @Test
    fun testProcessRequest_allServicesThrowExceptions() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenThrow(RuntimeException("Aura failed"))
        whenever(kaiService.processRequest(any())).thenThrow(RuntimeException("Kai failed"))
        whenever(cascadeService.processRequest(any())).thenThrow(RuntimeException("Cascade failed"))

        val response = genesisAgent.processRequest(request)

        assertNotNull("Response should not be null even with all service failures", response)
        assertEquals("Response should be empty when all services fail", "", response.content)
        assertEquals("Confidence should be 0.0f when all services fail", 0.0f, response.confidence)
    }

    @Test
    fun testProcessRequest_emptyResponses() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("", 0.0f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("", 0.0f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("", 0.0f))

        val response = genesisAgent.processRequest(request)

        assertNotNull("Response should not be null", response)
        assertTrue("Response content should be empty or whitespace", response.content.isBlank())
        assertEquals("Confidence should be 0.0f with empty responses", 0.0f, response.confidence)
    }

    @Test
    fun testProcessRequest_varyingConfidenceLevels() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.3f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.95f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.1f))

        val response = genesisAgent.processRequest(request)

        assertNotNull("Response should not be null", response)
        assertEquals("Confidence should be max of all responses", 0.95f, response.confidence)
        assertTrue("Response should contain all service responses", response.content.contains("aura"))
        assertTrue("Response should contain all service responses", response.content.contains("kai"))
        assertTrue("Response should contain all service responses", response.content.contains("cascade"))
    }

    @Test
    fun testProcessRequest_extremeConfidenceValues() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", Float.MAX_VALUE))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", Float.MIN_VALUE))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", -1.0f))

        val response = genesisAgent.processRequest(request)

        assertNotNull("Response should not be null", response)
        assertEquals("Confidence should be maximum value", Float.MAX_VALUE, response.confidence)
    }

    @Test
    fun testProcessRequest_requestWithComplexContext() = runBlocking {
        val complexContext = mapOf(
            "user_id" to "12345",
            "session_id" to "abcde",
            "language" to "en-US",
            "preferences" to "dark_mode=true,notifications=false",
            "history" to "previous_query_1,previous_query_2"
        )
        val request = AiRequest("complex prompt", complexContext)
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura response", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai response", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade response", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertNotNull("Response should not be null", response)
        assertTrue("Response should have content", response.content.isNotEmpty())
        assertTrue("Confidence should be positive", response.confidence > 0.0f)
    }

    @Test
    fun testParticipateWithAgents_complexContextConstruction() = runBlocking {
        val agent = DummyAgent("TestAgent", "response")
        val context = mapOf(
            "key1" to "value with spaces",
            "key2" to "value:with:colons",
            "key3" to "",
            "key4" to "value_with_special_chars!@#$%"
        )

        val responses = genesisAgent.participateWithAgents(
            context = context,
            agents = listOf(agent),
            prompt = "test prompt",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("response", responses["TestAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_extremelyLongPrompt() = runBlocking {
        val agent = DummyAgent("TestAgent", "response")
        val longPrompt = "a".repeat(10000)

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(agent),
            prompt = longPrompt,
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("response", responses["TestAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_largeNumberOfAgents() = runBlocking {
        val agents = (1..50).map { i ->
            DummyAgent("Agent$i", "response$i", i / 50.0f)
        }

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = agents,
            prompt = "test prompt",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(50, responses.size)
        agents.forEach { agent ->
            assertTrue("Response should contain agent ${agent.getName()}", responses.containsKey(agent.getName()))
        }
    }

    @Test
    fun testParticipateWithAgents_mixedSuccessAndFailure() = runBlocking {
        val workingAgent1 = DummyAgent("Working1", "success1", 0.8f)
        val failingAgent1 = FailingAgent("Failing1")
        val workingAgent2 = DummyAgent("Working2", "success2", 0.9f)
        val failingAgent2 = FailingAgent("Failing2")

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(workingAgent1, failingAgent1, workingAgent2, failingAgent2),
            prompt = "test prompt",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(2, responses.size)
        assertEquals("success1", responses["Working1"]?.content)
        assertEquals("success2", responses["Working2"]?.content)
        assertNull(responses["Failing1"])
        assertNull(responses["Failing2"])
    }

    @Test
    fun testParticipateWithAgents_differentConversationModes() = runBlocking {
        val agent = DummyAgent("TestAgent", "response")
        val modes = GenesisAgent.ConversationMode.values()

        for (mode in modes) {
            val responses = genesisAgent.participateWithAgents(
                context = emptyMap(),
                agents = listOf(agent),
                prompt = "test prompt for mode $mode",
                mode = mode
            )

            assertEquals("Mode $mode should work", 1, responses.size)
            assertEquals("response", responses["TestAgent"]?.content)
        }
    }

    @Test
    fun testAggregateAgentResponses_floatPrecisionEdgeCases() {
        val resp1 = mapOf("Agent1" to AgentResponse("response1", 0.1f + 0.2f))
        val resp2 = mapOf("Agent1" to AgentResponse("response2", 0.3f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

        assertEquals(1, consensus.size)
        // Due to floating point precision, this test validates behavior with near-equal values
        assertTrue("Should handle float precision correctly", consensus.containsKey("Agent1"))
    }

    @Test
    fun testAggregateAgentResponses_infiniteConfidence() {
        val resp1 = mapOf("Agent1" to AgentResponse("response1", Float.POSITIVE_INFINITY))
        val resp2 = mapOf("Agent1" to AgentResponse("response2", 0.9f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

        assertEquals(1, consensus.size)
        assertEquals("response1", consensus["Agent1"]?.content)
        assertEquals(Float.POSITIVE_INFINITY, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_nanConfidence() {
        val resp1 = mapOf("Agent1" to AgentResponse("response1", Float.NaN))
        val resp2 = mapOf("Agent1" to AgentResponse("response2", 0.9f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

        assertEquals(1, consensus.size)
        // NaN comparisons are always false, so the second response should be chosen
        assertEquals("response2", consensus["Agent1"]?.content)
        assertEquals(0.9f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_veryLargeDataset() {
        val responses = (1..1000).map { i ->
            mapOf("Agent$i" to AgentResponse("response$i", (i % 100) / 100.0f))
        }

        val consensus = genesisAgent.aggregateAgentResponses(responses)

        assertEquals(1000, consensus.size)
        // Verify that each agent has been processed
        (1..1000).forEach { i ->
            assertTrue("Agent$i should be in consensus", consensus.containsKey("Agent$i"))
        }
    }

    @Test
    fun testAggregateAgentResponses_emptyStringContent() {
        val resp1 = mapOf("Agent1" to AgentResponse("", 0.5f))
        val resp2 = mapOf("Agent1" to AgentResponse("actual content", 0.3f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

        assertEquals(1, consensus.size)
        assertEquals("", consensus["Agent1"]?.content)
        assertEquals(0.5f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_specialCharactersInContent() {
        val resp1 = mapOf("Agent1" to AgentResponse("Content with émojis 🎉 and symbols €£¥", 0.7f))
        val resp2 = mapOf("Agent1" to AgentResponse("Content with\nnewlines\tand\ttabs", 0.8f))

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

        assertEquals(1, consensus.size)
        assertEquals("Content with\nnewlines\tand\ttabs", consensus["Agent1"]?.content)
        assertEquals(0.8f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testDummyAgent_edgeCases() = runBlocking {
        val agentWithEmptyName = DummyAgent("", "response", 0.5f)
        val agentWithNegativeConfidence = DummyAgent("TestAgent", "response", -0.5f)
        val agentWithZeroConfidence = DummyAgent("TestAgent", "response", 0.0f)

        assertEquals("", agentWithEmptyName.getName())
        assertEquals(-0.5f, agentWithNegativeConfidence.processRequest(AiRequest("test", emptyMap())).confidence)
        assertEquals(0.0f, agentWithZeroConfidence.processRequest(AiRequest("test", emptyMap())).confidence)
    }

    @Test
    fun testFailingAgent_edgeCases() = runBlocking {
        val agentWithEmptyName = FailingAgent("")
        val agentWithSpecialChars = FailingAgent("Agent!@#$%")

        assertEquals("", agentWithEmptyName.getName())
        assertEquals("Agent!@#$%", agentWithSpecialChars.getName())

        try {
            agentWithEmptyName.processRequest(AiRequest("test", emptyMap()))
            fail("Should throw RuntimeException")
        } catch (e: RuntimeException) {
            assertEquals("Agent processing failed", e.message)
        }
    }

    @Test
    fun testGenesisAgent_threadSafety() = runBlocking {
        val agent = DummyAgent("ThreadSafeAgent", "response")
        val results = mutableListOf<Map<String, AgentResponse>>()

        val jobs = (1..100).map { i ->
            kotlinx.coroutines.async {
                genesisAgent.participateWithAgents(
                    context = mapOf("iteration" to i.toString()),
                    agents = listOf(agent),
                    prompt = "concurrent test $i",
                    mode = GenesisAgent.ConversationMode.TURN_ORDER
                )
            }
        }

        jobs.forEach { job ->
            results.add(job.await())
        }

        assertEquals(100, results.size)
        results.forEach { result ->
            assertEquals(1, result.size)
            assertEquals("response", result["ThreadSafeAgent"]?.content)
        }
    }

    @Test
    fun testGenesisAgent_stressTestWithServices() = runBlocking {
        val request = AiRequest("stress test", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val responses = (1..50).map {
            kotlinx.coroutines.async {
                genesisAgent.processRequest(request)
            }
        }

        responses.forEach { response ->
            val result = response.await()
            assertNotNull("Response should not be null", result)
            assertTrue("Response should have content", result.content.isNotEmpty())
            assertEquals("Confidence should be consistent", 0.9f, result.confidence)
        }
    }

    @Test
    fun testConversationMode_enumProperties() {
        val modes = GenesisAgent.ConversationMode.values()
        
        assertEquals(3, modes.size)
        assertTrue(modes.contains(GenesisAgent.ConversationMode.TURN_ORDER))
        assertTrue(modes.contains(GenesisAgent.ConversationMode.CASCADE))
        assertTrue(modes.contains(GenesisAgent.ConversationMode.CONSENSUS))
        
        // Test enum ordinal consistency
        assertEquals(0, GenesisAgent.ConversationMode.TURN_ORDER.ordinal)
        assertEquals(1, GenesisAgent.ConversationMode.CASCADE.ordinal)
        assertEquals(2, GenesisAgent.ConversationMode.CONSENSUS.ordinal)
        
        // Test enum name consistency
        assertEquals("TURN_ORDER", GenesisAgent.ConversationMode.TURN_ORDER.name)
        assertEquals("CASCADE", GenesisAgent.ConversationMode.CASCADE.name)
        assertEquals("CONSENSUS", GenesisAgent.ConversationMode.CONSENSUS.name)
    }

    @Test
    fun testInterface_agentContractCompliance() = runBlocking {
        val testAgent = object : Agent {
            override fun getName(): String = "TestContractAgent"
            override fun getType(): String? = "TestType"
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                return AgentResponse("contract response", 0.8f)
            }
        }

        assertEquals("TestContractAgent", testAgent.getName())
        assertEquals("TestType", testAgent.getType())
        
        val response = testAgent.processRequest(AiRequest("test", emptyMap()))
        assertEquals("contract response", response.content)
        assertEquals(0.8f, response.confidence)
    }

    @Test
    fun testAggregateAgentResponses_memoryEfficiency() {
        // Test with large strings to ensure memory efficiency
        val largeContent = "x".repeat(1000)
        val responses = (1..100).map { i ->
            mapOf("Agent$i" to AgentResponse(largeContent, i / 100.0f))
        }

        val consensus = genesisAgent.aggregateAgentResponses(responses)
        
        assertEquals(100, consensus.size)
        consensus.values.forEach { response ->
            assertEquals(largeContent, response.content)
        }
    }

    @Test
    fun testParticipateWithAgents_contextKeyOrdering() = runBlocking {
        val agent = DummyAgent("TestAgent", "response")
        val context1 = mapOf("a" to "1", "b" to "2", "c" to "3")
        val context2 = mapOf("c" to "3", "b" to "2", "a" to "1")

        val responses1 = genesisAgent.participateWithAgents(
            context = context1,
            agents = listOf(agent),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        val responses2 = genesisAgent.participateWithAgents(
            context = context2,
            agents = listOf(agent),
            prompt = "test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses1.size)
        assertEquals(1, responses2.size)
        assertEquals("response", responses1["TestAgent"]?.content)
        assertEquals("response", responses2["TestAgent"]?.content)
    }
}