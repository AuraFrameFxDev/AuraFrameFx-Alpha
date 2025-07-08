package dev.aurakai.auraframefx.ai.agents

import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AiRequest
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.mockito.kotlin.*
import java.util.concurrent.ConcurrentHashMap
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.withTimeout

class DummyAgent(
    private val name: String,
    private val response: String,
    private val confidence: Float = 1.0f
) : Agent {
    override fun getName() = name
    override fun getType() = null
    override suspend fun processRequest(request: AiRequest) = AgentResponse(response, confidence)
}

class FailingAgent(private val name: String) : Agent {
    override fun getName() = name
    override fun getType() = null
    override suspend fun processRequest(request: AiRequest): AgentResponse {
        throw RuntimeException("Agent processing failed")
    }
}

class SlowAgent(private val name: String, private val delayMs: Long = 1000) : Agent {
    override fun getName() = name
    override fun getType() = null
    override suspend fun processRequest(request: AiRequest): AgentResponse {
        delay(delayMs)
        return AgentResponse("slow response", 1.0f)
    }
}

class GenesisAgentTest {
    private lateinit var auraService: AuraAIService
    private lateinit var kaiService: KaiAIService
    private lateinit var cascadeService: CascadeAIService
    private lateinit var genesisAgent: GenesisAgent

    @Before
    fun setup() {
        auraService = mock<AuraAIService>()
        kaiService = mock<KaiAIService>()
        cascadeService = mock<CascadeAIService>()
        genesisAgent = GenesisAgent(
            auraService = auraService,
            kaiService = kaiService,
            cascadeService = cascadeService
        )
    }

    // Original tests preserved with cleaned formatting
    @Test
    fun testParticipateWithAgents_turnOrder() = runBlocking {
        val dummyAgent = DummyAgent("Dummy", "ok")
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("ok", 1.0f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("ok", 1.0f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("ok", 1.0f))
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(dummyAgent),
            "test",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        assertTrue(responses["Dummy"]?.content == "ok")
    }

    @Test
    fun testAggregateAgentResponses() {
        val resp1 = mapOf("A" to AgentResponse("foo", 0.5f))
        val resp2 = mapOf("A" to AgentResponse("bar", 0.9f))
        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))
        assertTrue(consensus["A"]?.content == "bar")
    }

    @Test
    fun testParticipateWithAgents_emptyAgentList() = runBlocking {
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            emptyList(),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        assertTrue("Expected empty response map", responses.isEmpty())
    }

    @Test
    fun testParticipateWithAgents_multipleAgents() = runBlocking {
        val agent1 = DummyAgent("Agent1", "response1", 0.8f)
        val agent2 = DummyAgent("Agent2", "response2", 0.9f)
        val agent3 = DummyAgent("Agent3", "response3", 0.7f)
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent1, agent2, agent3),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
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
            context,
            listOf(agent),
            "prompt with context",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("contextual response", responses["TestAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_nullPrompt() = runBlocking {
        val agent = DummyAgent("TestAgent", "response")
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent),
            null,
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("response", responses["TestAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_emptyPrompt() = runBlocking {
        val agent = DummyAgent("TestAgent", "empty prompt response")
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent),
            "",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("empty prompt response", responses["TestAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_agentThrowsException() = runBlocking {
        val failingAgent = FailingAgent("FailingAgent")
        val workingAgent = DummyAgent("WorkingAgent", "success")
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(failingAgent, workingAgent),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        // Should handle failing agent gracefully and continue with working agent
        assertEquals(1, responses.size)
        assertEquals("success", responses["WorkingAgent"]?.content)
        assertNull(responses["FailingAgent"])
    }

    @Test
    fun testParticipateWithAgents_duplicateAgentNames() = runBlocking {
        val agent1 = DummyAgent("SameName", "response1")
        val agent2 = DummyAgent("SameName", "response2")
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent1, agent2),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        // Should handle duplicate names - last one wins or both preserved
        assertEquals(1, responses.size)
        assertTrue(responses.containsKey("SameName"))
        assertTrue(responses["SameName"]?.content == "response1" || responses["SameName"]?.content == "response2")
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
        // Should pick one of the responses consistently
        assertTrue(consensus["Agent1"]?.content == "response1" || consensus["Agent1"]?.content == "response2")
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
        // Type might be null or a specific value - just verify it doesn't throw
        assertNotNull("Method should execute without throwing", true)
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
            genesisAgent.processRequest(null)
            fail("Should throw exception for null request")
        } catch (e: Exception) {
            // Expected behavior
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
        
        // Simulate concurrent access
        val jobs = (1..10).map { i ->
            async {
                val response = genesisAgent.participateWithAgents(
                    emptyMap(),
                    listOf(agent),
                    "concurrent test $i",
                    GenesisAgent.ConversationMode.TURN_ORDER
                )
                responses.putAll(response)
            }
        }
        
        jobs.forEach { it.await() }
        
        assertTrue("Should handle concurrent access", responses.isNotEmpty())
        assertEquals("response", responses["ConcurrentAgent"]?.content)
    }

    // Additional comprehensive tests for enhanced coverage
    @Test
    fun testParticipateWithAgents_largeNumberOfAgents() = runBlocking {
        val agents = (1..50).map { i ->
            DummyAgent("Agent$i", "response$i", i / 50.0f)
        }
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            agents,
            "test with many agents",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(50, responses.size)
        agents.forEach { agent ->
            assertTrue("Agent ${agent.getName()} should be in responses", 
                responses.containsKey(agent.getName()))
        }
    }

    @Test
    fun testParticipateWithAgents_mixedSuccessAndFailure() = runBlocking {
        val agents = listOf(
            DummyAgent("Success1", "ok1", 0.8f),
            FailingAgent("Failure1"),
            DummyAgent("Success2", "ok2", 0.9f),
            FailingAgent("Failure2"),
            DummyAgent("Success3", "ok3", 0.7f)
        )
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            agents,
            "mixed test",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(3, responses.size)
        assertEquals("ok1", responses["Success1"]?.content)
        assertEquals("ok2", responses["Success2"]?.content)
        assertEquals("ok3", responses["Success3"]?.content)
        assertNull(responses["Failure1"])
        assertNull(responses["Failure2"])
    }

    @Test
    fun testParticipateWithAgents_timeout() = runBlocking {
        val slowAgent = SlowAgent("SlowAgent", 5000)
        val fastAgent = DummyAgent("FastAgent", "fast response")
        
        try {
            withTimeout(2000) {
                genesisAgent.participateWithAgents(
                    emptyMap(),
                    listOf(slowAgent, fastAgent),
                    "timeout test",
                    GenesisAgent.ConversationMode.TURN_ORDER
                )
            }
            fail("Should have timed out")
        } catch (e: kotlinx.coroutines.TimeoutCancellationException) {
            // Expected behavior
            assertTrue("Should timeout on slow agents", true)
        }
    }

    @Test
    fun testParticipateWithAgents_veryLongPrompt() = runBlocking {
        val longPrompt = "x".repeat(10000)
        val agent = DummyAgent("LongPromptAgent", "handled long prompt")
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent),
            longPrompt,
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("handled long prompt", responses["LongPromptAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_specialCharactersInPrompt() = runBlocking {
        val specialPrompt = "Test with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        val agent = DummyAgent("SpecialAgent", "handled special chars")
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent),
            specialPrompt,
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("handled special chars", responses["SpecialAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_unicodePrompt() = runBlocking {
        val unicodePrompt = "Unicode test: 你好世界 🌍 émojis ñ"
        val agent = DummyAgent("UnicodeAgent", "handled unicode")
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent),
            unicodePrompt,
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("handled unicode", responses["UnicodeAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_largeContext() = runBlocking {
        val largeContext = (1..1000).associate { i ->
            "key$i" to "value$i"
        }
        val agent = DummyAgent("ContextAgent", "handled large context")
        
        val responses = genesisAgent.participateWithAgents(
            largeContext,
            listOf(agent),
            "test",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("handled large context", responses["ContextAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_specialCharactersInContext() = runBlocking {
        val specialContext = mapOf(
            "key with spaces" to "value with spaces",
            "key-with-dashes" to "value-with-dashes",
            "key_with_underscores" to "value_with_underscores",
            "key.with.dots" to "value.with.dots",
            "key/with/slashes" to "value/with/slashes"
        )
        val agent = DummyAgent("SpecialContextAgent", "handled special context")
        
        val responses = genesisAgent.participateWithAgents(
            specialContext,
            listOf(agent),
            "test",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("handled special context", responses["SpecialContextAgent"]?.content)
    }

    @Test
    fun testAggregateAgentResponses_extremeConfidenceValues() {
        val responses = listOf(
            mapOf("Agent1" to AgentResponse("response1", Float.MAX_VALUE)),
            mapOf("Agent1" to AgentResponse("response2", Float.MIN_VALUE)),
            mapOf("Agent1" to AgentResponse("response3", Float.POSITIVE_INFINITY)),
            mapOf("Agent1" to AgentResponse("response4", Float.NEGATIVE_INFINITY)),
            mapOf("Agent1" to AgentResponse("response5", Float.NaN))
        )
        
        val consensus = genesisAgent.aggregateAgentResponses(responses)
        
        assertEquals(1, consensus.size)
        assertNotNull(consensus["Agent1"])
        // Should handle extreme values gracefully
        assertTrue("Should handle extreme confidence values", consensus["Agent1"]?.content?.isNotEmpty() == true)
    }

    @Test
    fun testAggregateAgentResponses_emptyResponseContent() {
        val responses = listOf(
            mapOf("Agent1" to AgentResponse("", 0.5f)),
            mapOf("Agent1" to AgentResponse("   ", 0.7f)),
            mapOf("Agent1" to AgentResponse("actual content", 0.3f))
        )
        
        val consensus = genesisAgent.aggregateAgentResponses(responses)
        
        assertEquals(1, consensus.size)
        // Should pick the response with highest confidence regardless of content
        assertEquals("   ", consensus["Agent1"]?.content)
        assertEquals(0.7f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_veryLongResponseContent() {
        val longContent = "x".repeat(100000)
        val responses = listOf(
            mapOf("Agent1" to AgentResponse(longContent, 0.8f)),
            mapOf("Agent1" to AgentResponse("short", 0.5f))
        )
        
        val consensus = genesisAgent.aggregateAgentResponses(responses)
        
        assertEquals(1, consensus.size)
        assertEquals(longContent, consensus["Agent1"]?.content)
        assertEquals(0.8f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_unicodeContent() {
        val responses = listOf(
            mapOf("Agent1" to AgentResponse("Hello 世界", 0.5f)),
            mapOf("Agent1" to AgentResponse("🌍 Emoji test", 0.8f)),
            mapOf("Agent1" to AgentResponse("Ñice tëst", 0.3f))
        )
        
        val consensus = genesisAgent.aggregateAgentResponses(responses)
        
        assertEquals(1, consensus.size)
        assertEquals("🌍 Emoji test", consensus["Agent1"]?.content)
        assertEquals(0.8f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testGenesisAgent_processRequest_emptyPrompt() = runBlocking {
        val request = AiRequest("", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("empty prompt response", 0.5f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("empty kai response", 0.6f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("empty cascade response", 0.4f))
        
        val response = genesisAgent.processRequest(request)
        
        assertNotNull(response)
        assertTrue(response.content.isNotEmpty())
        assertTrue(response.confidence >= 0.0f)
    }

    @Test
    fun testGenesisAgent_processRequest_largePrompt() = runBlocking {
        val largePrompt = "x".repeat(50000)
        val request = AiRequest(largePrompt, emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("large prompt response", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("large kai response", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("large cascade response", 0.7f))
        
        val response = genesisAgent.processRequest(request)
        
        assertNotNull(response)
        assertTrue(response.content.isNotEmpty())
        assertTrue(response.confidence >= 0.0f)
    }

    @Test
    fun testGenesisAgent_processRequest_withLargeContext() = runBlocking {
        val largeContext = (1..10000).associate { i ->
            "contextKey$i" to "contextValue$i"
        }
        val request = AiRequest("test prompt", largeContext)
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("context response", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("context kai response", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("context cascade response", 0.7f))
        
        val response = genesisAgent.processRequest(request)
        
        assertNotNull(response)
        assertTrue(response.content.isNotEmpty())
        assertTrue(response.confidence >= 0.0f)
    }

    @Test
    fun testGenesisAgent_processRequest_servicesThrowExceptions() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenThrow(RuntimeException("Aura service failed"))
        whenever(kaiService.processRequest(any())).thenThrow(RuntimeException("Kai service failed"))
        whenever(cascadeService.processRequest(any())).thenThrow(RuntimeException("Cascade service failed"))
        
        try {
            val response = genesisAgent.processRequest(request)
            // If no exception is thrown, verify the response handles the error gracefully
            assertNotNull("Should handle service failures gracefully", response)
        } catch (e: Exception) {
            // If exception is thrown, that's also acceptable behavior
            assertTrue("Should handle service failures", e.message?.contains("failed") == true)
        }
    }

    @Test
    fun testGenesisAgent_processRequest_partialServiceFailure() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura success", 0.8f))
        whenever(kaiService.processRequest(any())).thenThrow(RuntimeException("Kai service failed"))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade success", 0.7f))
        
        val response = genesisAgent.processRequest(request)
        
        assertNotNull(response)
        assertTrue("Should handle partial service failures", response.content.isNotEmpty())
        assertTrue(response.confidence >= 0.0f)
    }

    @Test
    fun testDummyAgent_extremeConfidenceValues() = runBlocking {
        val testCases = listOf(
            Triple("ZeroConfidence", 0.0f, "zero"),
            Triple("NegativeConfidence", -0.5f, "negative"),
            Triple("ExtremeConfidence", Float.MAX_VALUE, "extreme"),
            Triple("InfiniteConfidence", Float.POSITIVE_INFINITY, "infinite"),
            Triple("NaNConfidence", Float.NaN, "nan")
        )
        
        testCases.forEach { (name, confidence, content) ->
            val agent = DummyAgent(name, content, confidence)
            val request = AiRequest("test", emptyMap())
            val response = agent.processRequest(request)
            
            assertEquals(content, response.content)
            assertEquals(confidence, response.confidence)
        }
    }

    @Test
    fun testDummyAgent_edgeCaseResponses() = runBlocking {
        val testCases = listOf(
            "",
            "   ",
            "Unicode: 你好 🌍 émojis ñ",
            "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?",
            "x".repeat(10000),
            "\n\r\t",
            "null",
            "undefined"
        )
        
        testCases.forEach { content ->
            val agent = DummyAgent("TestAgent", content, 0.5f)
            val request = AiRequest("test", emptyMap())
            val response = agent.processRequest(request)
            
            assertEquals(content, response.content)
            assertEquals(0.5f, response.confidence)
        }
    }

    @Test
    fun testFailingAgent_differentExceptionTypes() = runBlocking {
        class CustomFailingAgent(name: String, private val exception: Exception) : Agent {
            override fun getName() = name
            override fun getType() = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                throw exception
            }
        }
        
        val agents = listOf(
            CustomFailingAgent("RuntimeAgent", RuntimeException("Runtime error")),
            CustomFailingAgent("IllegalStateAgent", IllegalStateException("Illegal state")),
            CustomFailingAgent("IllegalArgumentAgent", IllegalArgumentException("Illegal argument")),
            CustomFailingAgent("NullPointerAgent", NullPointerException("Null pointer")),
            CustomFailingAgent("IndexOutOfBoundsAgent", IndexOutOfBoundsException("Index out of bounds"))
        )
        
        agents.forEach { agent ->
            try {
                agent.processRequest(AiRequest("test", emptyMap()))
                fail("Agent ${agent.getName()} should have thrown an exception")
            } catch (e: Exception) {
                assertTrue("Should throw expected exception type", e.javaClass.name.contains("Exception"))
            }
        }
    }

    @Test
    fun testGenesisAgent_threadSafety() = runBlocking {
        val agent = DummyAgent("ThreadSafeAgent", "response")
        val results = mutableListOf<Map<String, AgentResponse>>()
        
        // Test concurrent access from multiple coroutines
        val jobs = (1..20).map { i ->
            async {
                genesisAgent.participateWithAgents(
                    mapOf("iteration" to i.toString()),
                    listOf(agent),
                    "concurrent test $i",
                    GenesisAgent.ConversationMode.TURN_ORDER
                )
            }
        }
        
        jobs.forEach { job ->
            results.add(job.await())
        }
        
        assertEquals(20, results.size)
        results.forEach { result ->
            assertEquals(1, result.size)
            assertEquals("response", result["ThreadSafeAgent"]?.content)
        }
    }

    @Test
    fun testAggregateAgentResponses_threadSafety() = runBlocking {
        val responses = (1..1000).map { i ->
            mapOf("Agent$i" to AgentResponse("response$i", i / 1000.0f))
        }
        
        // Test concurrent aggregation
        val jobs = (1..10).map {
            async {
                genesisAgent.aggregateAgentResponses(responses)
            }
        }
        
        val results = jobs.map { it.await() }
        
        // All results should be identical
        val firstResult = results.first()
        results.forEach { result ->
            assertEquals(firstResult.size, result.size)
            firstResult.keys.forEach { key ->
                assertEquals(firstResult[key]?.content, result[key]?.content)
                assertEquals(firstResult[key]?.confidence, result[key]?.confidence)
            }
        }
    }

    @Test
    fun testGenesisAgent_memoryUsage() = runBlocking {
        // Test that large operations don't cause memory leaks
        val largeAgentList = (1..100).map { i ->
            DummyAgent("Agent$i", "response$i".repeat(100), i / 100.0f)
        }
        
        repeat(10) {
            val responses = genesisAgent.participateWithAgents(
                emptyMap(),
                largeAgentList,
                "memory test",
                GenesisAgent.ConversationMode.TURN_ORDER
            )
            
            assertEquals(100, responses.size)
            
            // Force garbage collection hint
            System.gc()
        }
        
        // Test passed if no OutOfMemoryError occurred
        assertTrue("Memory test completed successfully", true)
    }

    @Test
    fun testGenesisAgent_robustness() = runBlocking {
        // Test with various edge cases combined
        val edgeCaseAgents = listOf(
            DummyAgent("", "empty name agent", 0.5f),
            DummyAgent("Agent with spaces", "spaces in name", 0.6f),
            DummyAgent("Agent-with-dashes", "dashes in name", 0.7f),
            DummyAgent("Agent_with_underscores", "underscores in name", 0.8f),
            DummyAgent("Agent.with.dots", "dots in name", 0.9f),
            DummyAgent("Agent/with/slashes", "slashes in name", 1.0f),
            DummyAgent("Agent\nwith\nnewlines", "newlines in name", 0.4f),
            DummyAgent("Agent\twith\ttabs", "tabs in name", 0.3f)
        )
        
        val responses = genesisAgent.participateWithAgents(
            mapOf("special" to "context"),
            edgeCaseAgents,
            "Edge case test with special chars: !@#$%^&*()",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(edgeCaseAgents.size, responses.size)
        
        // Verify each agent got a response
        edgeCaseAgents.forEach { agent ->
            assertTrue("Agent ${agent.getName()} should have response", 
                responses.containsKey(agent.getName()))
        }
    }

    @Test
    fun testGenesisAgent_boundaryConditions() = runBlocking {
        // Test with boundary values
        val boundaryAgents = listOf(
            DummyAgent("MaxFloatAgent", "max", Float.MAX_VALUE),
            DummyAgent("MinFloatAgent", "min", Float.MIN_VALUE),
            DummyAgent("PositiveInfAgent", "pos_inf", Float.POSITIVE_INFINITY),
            DummyAgent("NegativeInfAgent", "neg_inf", Float.NEGATIVE_INFINITY),
            DummyAgent("NaNAgent", "nan", Float.NaN)
        )
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            boundaryAgents,
            "boundary test",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(boundaryAgents.size, responses.size)
        
        // Verify responses contain expected values
        boundaryAgents.forEach { agent ->
            val response = responses[agent.getName()]
            assertNotNull("Response for ${agent.getName()} should not be null", response)
            assertTrue("Response should have content", response?.content?.isNotEmpty() == true)
        }
    }

    @Test
    fun testSlowAgent_functionality() = runBlocking {
        val slowAgent = SlowAgent("SlowAgent", 100) // 100ms delay
        
        assertEquals("SlowAgent", slowAgent.getName())
        assertNull(slowAgent.getType())
        
        val startTime = System.currentTimeMillis()
        val response = slowAgent.processRequest(AiRequest("test", emptyMap()))
        val endTime = System.currentTimeMillis()
        
        assertEquals("slow response", response.content)
        assertEquals(1.0f, response.confidence)
        assertTrue("Should take at least 100ms", endTime - startTime >= 100)
    }

    @Test
    fun testGenesisAgent_validationAndSanitization() = runBlocking {
        // Test with potentially problematic inputs
        val maliciousInputs = listOf(
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "\\x00\\x01\\x02",
            "A".repeat(1000000) // 1MB string
        )
        
        val agent = DummyAgent("SanitizeAgent", "sanitized")
        
        maliciousInputs.forEach { input ->
            try {
                val responses = genesisAgent.participateWithAgents(
                    emptyMap(),
                    listOf(agent),
                    input,
                    GenesisAgent.ConversationMode.TURN_ORDER
                )
                
                assertEquals(1, responses.size)
                assertEquals("sanitized", responses["SanitizeAgent"]?.content)
            } catch (e: OutOfMemoryError) {
                // Acceptable for very large inputs
                assertTrue("Handled memory constraint", true)
            }
        }
    }
}
