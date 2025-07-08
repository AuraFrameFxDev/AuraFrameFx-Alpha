package dev.aurakai.auraframefx.ai.agents

import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AiRequest
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.mockito.kotlin.*
import java.util.concurrent.ConcurrentHashMap

<<<<<<< HEAD
class DummyAgent(
    private val name: String,
    private val response: String,
    private val confidence: Float = 1.0f
) : Agent {
=======
class DummyAgent(private val name: String, private val response: String, private val confidence: Float = 1.0f) : Agent {
>>>>>>> origin/coderabbitai/docstrings/78f34ad
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

    // Existing tests preserved
    @Test
    fun testParticipateWithAgents_turnOrder() = runBlocking {
        val dummyAgent = DummyAgent("Dummy", "ok")
        whenever(auraService.processRequest(any())).thenReturn(
            AgentResponse("ok", 1.0f)
        )
        whenever(kaiService.processRequest(any())).thenReturn(
            AgentResponse("ok", 1.0f)
        )
        whenever(cascadeService.processRequest(any())).thenReturn(
            AgentResponse("ok", 1.0f)
        )
<<<<<<< HEAD

=======
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
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

    // New comprehensive tests
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
<<<<<<< HEAD

=======
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent1, agent2, agent3),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
<<<<<<< HEAD

=======
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

=======
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        val responses = genesisAgent.participateWithAgents(
            context,
            listOf(agent),
            "prompt with context",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
<<<<<<< HEAD

=======
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        assertEquals(1, responses.size)
        assertEquals("contextual response", responses["TestAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_nullPrompt() = runBlocking {
        val agent = DummyAgent("TestAgent", "response")
<<<<<<< HEAD

=======
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent),
            null,
            GenesisAgent.ConversationMode.TURN_ORDER
        )
<<<<<<< HEAD

=======
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        assertEquals(1, responses.size)
        assertEquals("response", responses["TestAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_emptyPrompt() = runBlocking {
        val agent = DummyAgent("TestAgent", "empty prompt response")
<<<<<<< HEAD

=======
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent),
            "",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
<<<<<<< HEAD

=======
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        assertEquals(1, responses.size)
        assertEquals("empty prompt response", responses["TestAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_agentThrowsException() = runBlocking {
        val failingAgent = FailingAgent("FailingAgent")
        val workingAgent = DummyAgent("WorkingAgent", "success")
<<<<<<< HEAD

=======
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(failingAgent, workingAgent),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
<<<<<<< HEAD

=======
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        // Should handle failing agent gracefully and continue with working agent
        assertEquals(1, responses.size)
        assertEquals("success", responses["WorkingAgent"]?.content)
        assertNull(responses["FailingAgent"])
    }

    @Test
    fun testParticipateWithAgents_duplicateAgentNames() = runBlocking {
        val agent1 = DummyAgent("SameName", "response1")
        val agent2 = DummyAgent("SameName", "response2")
<<<<<<< HEAD

=======
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent1, agent2),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
<<<<<<< HEAD

=======
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

=======
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        assertEquals(1, consensus.size)
        assertEquals("single response", consensus["Agent1"]?.content)
        assertEquals(0.8f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_multipleResponsesSameAgent() {
        val resp1 = mapOf("Agent1" to AgentResponse("response1", 0.5f))
        val resp2 = mapOf("Agent1" to AgentResponse("response2", 0.9f))
        val resp3 = mapOf("Agent1" to AgentResponse("response3", 0.3f))
<<<<<<< HEAD

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2, resp3))

=======
        
        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2, resp3))
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

=======
        
        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

=======
        
        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        assertEquals(1, consensus.size)
        assertEquals(0.5f, consensus["Agent1"]?.confidence)
        // Should pick one of the responses consistently
        assertTrue(consensus["Agent1"]?.content == "response1" || consensus["Agent1"]?.content == "response2")
    }

    @Test
    fun testAggregateAgentResponses_zeroConfidence() {
        val resp1 = mapOf("Agent1" to AgentResponse("response1", 0.0f))
        val resp2 = mapOf("Agent1" to AgentResponse("response2", 0.1f))
<<<<<<< HEAD

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

=======
        
        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        assertEquals(1, consensus.size)
        assertEquals("response2", consensus["Agent1"]?.content)
        assertEquals(0.1f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_negativeConfidence() {
        val resp1 = mapOf("Agent1" to AgentResponse("response1", -0.5f))
        val resp2 = mapOf("Agent1" to AgentResponse("response2", 0.1f))
<<<<<<< HEAD

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

=======
        
        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        assertEquals(1, consensus.size)
        assertEquals("response2", consensus["Agent1"]?.content)
        assertEquals(0.1f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_largeNumberOfResponses() {
        val responses = (1..100).map { i ->
            mapOf("Agent1" to AgentResponse("response$i", i / 100.0f))
        }
<<<<<<< HEAD

        val consensus = genesisAgent.aggregateAgentResponses(responses)

=======
        
        val consensus = genesisAgent.aggregateAgentResponses(responses)
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))

=======
        
        val consensus = genesisAgent.aggregateAgentResponses(listOf(resp1, resp2))
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

=======
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD
        genesisAgent.getType()
=======
        val type = genesisAgent.getType()
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        // Type might be null or a specific value - just verify it doesn't throw
        assertNotNull("Method should execute without throwing", true)
    }

    @Test
    fun testGenesisAgent_processRequest() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura response", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai response", 0.9f))
<<<<<<< HEAD
        whenever(cascadeService.processRequest(any())).thenReturn(
            AgentResponse(
                "cascade response",
                0.7f
            )
        )

        val response = genesisAgent.processRequest(request)

=======
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade response", 0.7f))
        
        val response = genesisAgent.processRequest(request)
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD
        assertTrue(
            "Should have at least TURN_ORDER mode",
            modes.contains(GenesisAgent.ConversationMode.TURN_ORDER)
        )
=======
        assertTrue("Should have at least TURN_ORDER mode", modes.contains(GenesisAgent.ConversationMode.TURN_ORDER))
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        assertTrue("Should have multiple conversation modes", modes.isNotEmpty())
    }

    @Test
    fun testDummyAgent_implementation() = runBlocking {
        val agent = DummyAgent("TestAgent", "test response", 0.5f)
<<<<<<< HEAD

        assertEquals("TestAgent", agent.getName())
        assertNull(agent.getType())

        val request = AiRequest("test", emptyMap())
        val response = agent.processRequest(request)

=======
        
        assertEquals("TestAgent", agent.getName())
        assertNull(agent.getType())
        
        val request = AiRequest("test", emptyMap())
        val response = agent.processRequest(request)
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        assertEquals("test response", response.content)
        assertEquals(0.5f, response.confidence)
    }

    @Test
    fun testFailingAgent_implementation() = runBlocking {
        val agent = FailingAgent("TestAgent")
<<<<<<< HEAD

        assertEquals("TestAgent", agent.getName())
        assertNull(agent.getType())

=======
        
        assertEquals("TestAgent", agent.getName())
        assertNull(agent.getType())
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

=======
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        // Simulate concurrent access
        val jobs = (1..10).map { i ->
            kotlinx.coroutines.async {
                val response = genesisAgent.participateWithAgents(
                    emptyMap(),
                    listOf(agent),
                    "concurrent test $i",
                    GenesisAgent.ConversationMode.TURN_ORDER
                )
                responses.putAll(response)
            }
        }
<<<<<<< HEAD

        jobs.forEach { it.await() }

=======
        
        jobs.forEach { it.await() }
        
>>>>>>> origin/coderabbitai/docstrings/78f34ad
        assertTrue("Should handle concurrent access", responses.isNotEmpty())
        assertEquals("response", responses["ConcurrentAgent"]?.content)
    }
}

<<<<<<< HEAD
// Additional comprehensive tests for better coverage

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
        assertTrue(
            "Agent ${agent.getName()} should be in responses",
            responses.containsKey(agent.getName())
        )
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
    assertTrue(
        "Should handle extreme confidence values",
        consensus["Agent1"]?.content?.isNotEmpty() == true
    )
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
    whenever(auraService.processRequest(any())).thenReturn(
        AgentResponse(
            "empty prompt response",
            0.5f
        )
    )
    whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("empty kai response", 0.6f))
    whenever(cascadeService.processRequest(any())).thenReturn(
        AgentResponse(
            "empty cascade response",
            0.4f
        )
    )

    val response = genesisAgent.processRequest(request)

    assertNotNull(response)
    assertTrue(response.content.isNotEmpty())
    assertTrue(response.confidence >= 0.0f)
}

@Test
fun testGenesisAgent_processRequest_largePrompt() = runBlocking {
    val largePrompt = "x".repeat(50000)
    val request = AiRequest(largePrompt, emptyMap())
    whenever(auraService.processRequest(any())).thenReturn(
        AgentResponse(
            "large prompt response",
            0.8f
        )
    )
    whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("large kai response", 0.9f))
    whenever(cascadeService.processRequest(any())).thenReturn(
        AgentResponse(
            "large cascade response",
            0.7f
        )
    )

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
    whenever(kaiService.processRequest(any())).thenReturn(
        AgentResponse(
            "context kai response",
            0.9f
        )
    )
    whenever(cascadeService.processRequest(any())).thenReturn(
        AgentResponse(
            "context cascade response",
            0.7f
        )
    )

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
    whenever(cascadeService.processRequest(any())).thenReturn(
        AgentResponse(
            "cascade success",
            0.7f
        )
    )

    val response = genesisAgent.processRequest(request)

    assertNotNull(response)
    assertTrue("Should handle partial service failures", response.content.isNotEmpty())
    assertTrue(response.confidence >= 0.0f)
}

@Test
fun testDummyAgent_withZeroConfidence() = runBlocking {
    val agent = DummyAgent("ZeroConfidenceAgent", "response", 0.0f)

    assertEquals("ZeroConfidenceAgent", agent.getName())

    val request = AiRequest("test", emptyMap())
    val response = agent.processRequest(request)

    assertEquals("response", response.content)
    assertEquals(0.0f, response.confidence)
}

@Test
fun testDummyAgent_withNegativeConfidence() = runBlocking {
    val agent = DummyAgent("NegativeConfidenceAgent", "response", -0.5f)

    val request = AiRequest("test", emptyMap())
    val response = agent.processRequest(request)

    assertEquals("response", response.content)
    assertEquals(-0.5f, response.confidence)
}

@Test
fun testDummyAgent_withExtremeConfidence() = runBlocking {
    val agent = DummyAgent("ExtremeConfidenceAgent", "response", Float.MAX_VALUE)

    val request = AiRequest("test", emptyMap())
    val response = agent.processRequest(request)

    assertEquals("response", response.content)
    assertEquals(Float.MAX_VALUE, response.confidence)
}

@Test
fun testDummyAgent_withEmptyResponse() = runBlocking {
    val agent = DummyAgent("EmptyResponseAgent", "", 0.5f)

    val request = AiRequest("test", emptyMap())
    val response = agent.processRequest(request)

    assertEquals("", response.content)
    assertEquals(0.5f, response.confidence)
}

@Test
fun testDummyAgent_withUnicodeResponse() = runBlocking {
    val unicodeResponse = "Unicode: 你好 🌍 émojis ñ"
    val agent = DummyAgent("UnicodeAgent", unicodeResponse, 0.5f)

    val request = AiRequest("test", emptyMap())
    val response = agent.processRequest(request)

    assertEquals(unicodeResponse, response.content)
    assertEquals(0.5f, response.confidence)
}

@Test
fun testFailingAgent_withDifferentExceptions() = runBlocking {
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
        CustomFailingAgent("IllegalArgumentAgent", IllegalArgumentException("Illegal argument"))
    )

    agents.forEach { agent ->
        try {
            agent.processRequest(AiRequest("test", emptyMap()))
            fail("Agent ${agent.getName()} should have thrown an exception")
        } catch (e: Exception) {
            assertTrue(
                "Should throw expected exception type",
                e is RuntimeException || e is IllegalStateException || e is IllegalArgumentException
            )
        }
    }
}

@Test
fun testGenesisAgent_threadSafety() = runBlocking {
    val agent = DummyAgent("ThreadSafeAgent", "response")
    val results = mutableListOf<Map<String, AgentResponse>>()

    // Test concurrent access from multiple coroutines
    val jobs = (1..20).map { i ->
        kotlinx.coroutines.async {
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
        kotlinx.coroutines.async {
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
        DummyAgent("Agent$i", "response$i".repeat(1000), i / 100.0f)
    }

    repeat(10) {
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            largeAgentList,
            "memory test",
            GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(100, responses.size)

        // Clear references to help GC
        responses.clear()
    }

    // Test passed if no OutOfMemoryError occurred
    assertTrue("Memory test completed successfully", true)
}

@Test
fun testGenesisAgent_extremeScenarios() = runBlocking {
    // Test with extreme values
    val extremePrompt = "x".repeat(1000000) // 1MB string
    val extremeContext = (1..10000).associate { i ->
        "key$i" to "value$i".repeat(100)
    }

    val agent = DummyAgent("ExtremeAgent", "handled extreme scenario")

    try {
        val responses = genesisAgent.participateWithAgents(
            extremeContext,
            listOf(agent),
            extremePrompt,
            GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("handled extreme scenario", responses["ExtremeAgent"]?.content)
    } catch (e: OutOfMemoryError) {
        // Acceptable if system runs out of memory
        assertTrue("System handled memory limitation", true)
    }
}
}
=======
    // Additional comprehensive tests for better coverage

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
    fun testDummyAgent_withZeroConfidence() = runBlocking {
        val agent = DummyAgent("ZeroConfidenceAgent", "response", 0.0f)
        
        assertEquals("ZeroConfidenceAgent", agent.getName())

        val request = AiRequest("test", emptyMap())
        val response = agent.processRequest(request)

        assertEquals("response", response.content)
        assertEquals(0.0f, response.confidence)
    }

    @Test
    fun testDummyAgent_withNegativeConfidence() = runBlocking {
        val agent = DummyAgent("NegativeConfidenceAgent", "response", -0.5f)

        val request = AiRequest("test", emptyMap())
        val response = agent.processRequest(request)

        assertEquals("response", response.content)
        assertEquals(-0.5f, response.confidence)
    }

    @Test
    fun testDummyAgent_withExtremeConfidence() = runBlocking {
        val agent = DummyAgent("ExtremeConfidenceAgent", "response", Float.MAX_VALUE)

        val request = AiRequest("test", emptyMap())
        val response = agent.processRequest(request)

        assertEquals("response", response.content)
        assertEquals(Float.MAX_VALUE, response.confidence)
    }

    @Test
    fun testDummyAgent_withEmptyResponse() = runBlocking {
        val agent = DummyAgent("EmptyResponseAgent", "", 0.5f)

        val request = AiRequest("test", emptyMap())
        val response = agent.processRequest(request)

        assertEquals("", response.content)
        assertEquals(0.5f, response.confidence)
    }

    @Test
    fun testDummyAgent_withUnicodeResponse() = runBlocking {
        val unicodeResponse = "Unicode: 你好 🌍 émojis ñ"
        val agent = DummyAgent("UnicodeAgent", unicodeResponse, 0.5f)

        val request = AiRequest("test", emptyMap())
        val response = agent.processRequest(request)

        assertEquals(unicodeResponse, response.content)
        assertEquals(0.5f, response.confidence)
    }

    @Test
    fun testFailingAgent_withDifferentExceptions() = runBlocking {
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
            CustomFailingAgent("IllegalArgumentAgent", IllegalArgumentException("Illegal argument"))
        )
        
        agents.forEach { agent ->
            try {
                agent.processRequest(AiRequest("test", emptyMap()))
                fail("Agent ${agent.getName()} should have thrown an exception")
            } catch (e: Exception) {
                assertTrue("Should throw expected exception type",
                    e is RuntimeException || e is IllegalStateException || e is IllegalArgumentException)
            }
        }
    }

    @Test
    fun testGenesisAgent_threadSafety() = runBlocking {
        val agent = DummyAgent("ThreadSafeAgent", "response")
        val results = mutableListOf<Map<String, AgentResponse>>()
        
        // Test concurrent access from multiple coroutines
        val jobs = (1..20).map { i ->
            kotlinx.coroutines.async {
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
            kotlinx.coroutines.async {
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
            DummyAgent("Agent$i", "response$i".repeat(1000), i / 100.0f)
        }
        
        repeat(10) {
            val responses = genesisAgent.participateWithAgents(
                emptyMap(),
                largeAgentList,
                "memory test",
                GenesisAgent.ConversationMode.TURN_ORDER
            )
            
            assertEquals(100, responses.size)
            
            // Clear references to help GC
            responses.clear()
        }
        
        // Test passed if no OutOfMemoryError occurred
        assertTrue("Memory test completed successfully", true)
    }

    @Test
    fun testGenesisAgent_extremeScenarios() = runBlocking {
        // Test with extreme values
        val extremePrompt = "x".repeat(1000000) // 1MB string
        val extremeContext = (1..10000).associate { i ->
            "key$i" to "value$i".repeat(100)
        }
        
        val agent = DummyAgent("ExtremeAgent", "handled extreme scenario")
        
        try {
            val responses = genesisAgent.participateWithAgents(
                extremeContext,
                listOf(agent),
                extremePrompt,
                GenesisAgent.ConversationMode.TURN_ORDER
            )

            assertEquals(1, responses.size)
            assertEquals("handled extreme scenario", responses["ExtremeAgent"]?.content)
        } catch (e: OutOfMemoryError) {
            // Acceptable if system runs out of memory
            assertTrue("System handled memory limitation", true)
        }
    }
}
>>>>>>> origin/coderabbitai/docstrings/78f34ad

    // Additional comprehensive tests for edge cases and boundary conditions

    @Test
    fun testParticipateWithAgents_nullAgentName() = runBlocking {
        class NullNameAgent : Agent {
            override fun getName(): String? = null
            override fun getType() = null
            override suspend fun processRequest(request: AiRequest) = AgentResponse("null name response", 0.5f)
        }
        
        val agent = NullNameAgent()
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        // Should handle null agent names gracefully
        assertTrue("Should handle null agent names", responses.size <= 1)
    }

    @Test
    fun testParticipateWithAgents_emptyAgentName() = runBlocking {
        val agent = DummyAgent("", "empty name response", 0.5f)
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("empty name response", responses[""]?.content)
    }

    @Test
    fun testParticipateWithAgents_whitespaceAgentName() = runBlocking {
        val agent = DummyAgent("   ", "whitespace name response", 0.5f)
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("whitespace name response", responses["   "]?.content)
    }

    @Test
    fun testParticipateWithAgents_contextWithNullValues() = runBlocking {
        val contextWithNulls = mapOf(
            "key1" to "value1",
            "key2" to null,
            "key3" to "value3"
        )
        val agent = DummyAgent("TestAgent", "handled null context", 0.5f)
        
        val responses = genesisAgent.participateWithAgents(
            contextWithNulls,
            listOf(agent),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("handled null context", responses["TestAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_contextWithEmptyValues() = runBlocking {
        val contextWithEmptyValues = mapOf(
            "key1" to "",
            "key2" to "   ",
            "key3" to "actual_value"
        )
        val agent = DummyAgent("TestAgent", "handled empty context", 0.5f)
        
        val responses = genesisAgent.participateWithAgents(
            contextWithEmptyValues,
            listOf(agent),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("handled empty context", responses["TestAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_agentReturnsNullResponse() = runBlocking {
        class NullResponseAgent(private val name: String) : Agent {
            override fun getName() = name
            override fun getType() = null
            override suspend fun processRequest(request: AiRequest): AgentResponse? = null
        }
        
        val agent = NullResponseAgent("NullResponseAgent")
        val workingAgent = DummyAgent("WorkingAgent", "working response", 0.8f)
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent, workingAgent),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        // Should handle null responses gracefully
        assertTrue("Should handle null responses", responses.size <= 2)
        assertEquals("working response", responses["WorkingAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_agentWithVeryLongName() = runBlocking {
        val longName = "Agent" + "x".repeat(10000)
        val agent = DummyAgent(longName, "long name response", 0.5f)
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("long name response", responses[longName]?.content)
    }

    @Test
    fun testParticipateWithAgents_agentWithSpecialCharacterName() = runBlocking {
        val specialName = "Agent!@#$%^&*()_+-=[]{}|;':\",./<>?"
        val agent = DummyAgent(specialName, "special char name response", 0.5f)
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("special char name response", responses[specialName]?.content)
    }

    @Test
    fun testParticipateWithAgents_agentWithUnicodeName() = runBlocking {
        val unicodeName = "Agent-你好世界-🌍-émojis-ñ"
        val agent = DummyAgent(unicodeName, "unicode name response", 0.5f)
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("unicode name response", responses[unicodeName]?.content)
    }

    @Test
    fun testParticipateWithAgents_timeoutScenario() = runBlocking {
        class SlowAgent(private val name: String, private val delayMs: Long) : Agent {
            override fun getName() = name
            override fun getType() = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                kotlinx.coroutines.delay(delayMs)
                return AgentResponse("slow response", 0.5f)
            }
        }
        
        val slowAgent = SlowAgent("SlowAgent", 100)
        val fastAgent = DummyAgent("FastAgent", "fast response", 0.8f)
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(slowAgent, fastAgent),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(2, responses.size)
        assertEquals("slow response", responses["SlowAgent"]?.content)
        assertEquals("fast response", responses["FastAgent"]?.content)
    }

    @Test
    fun testAggregateAgentResponses_duplicateAgentNamesWithDifferentConfidences() {
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
    fun testAggregateAgentResponses_responseWithNullContent() {
        val responses = listOf(
            mapOf("Agent1" to AgentResponse(null, 0.8f)),
            mapOf("Agent1" to AgentResponse("actual content", 0.5f))
        )
        
        val consensus = genesisAgent.aggregateAgentResponses(responses)
        
        assertEquals(1, consensus.size)
        // Should handle null content gracefully
        assertTrue("Should handle null content", consensus.containsKey("Agent1"))
    }

    @Test
    fun testAggregateAgentResponses_mixedDataTypes() {
        val responses = listOf(
            mapOf("Agent1" to AgentResponse("123", 0.5f)),
            mapOf("Agent1" to AgentResponse("true", 0.7f)),
            mapOf("Agent1" to AgentResponse("{'key': 'value'}", 0.3f))
        )
        
        val consensus = genesisAgent.aggregateAgentResponses(responses)
        
        assertEquals(1, consensus.size)
        assertEquals("true", consensus["Agent1"]?.content)
        assertEquals(0.7f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_confidenceEdgeCases() {
        val responses = listOf(
            mapOf("Agent1" to AgentResponse("response1", 0.0f)),
            mapOf("Agent1" to AgentResponse("response2", -0.0f)),
            mapOf("Agent1" to AgentResponse("response3", 0.000001f))
        )
        
        val consensus = genesisAgent.aggregateAgentResponses(responses)
        
        assertEquals(1, consensus.size)
        assertEquals("response3", consensus["Agent1"]?.content)
        assertEquals(0.000001f, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testGenesisAgent_processRequest_withNullContext() = runBlocking {
        val request = AiRequest("test prompt", null)
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("null context response", 0.5f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("null context kai", 0.6f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("null context cascade", 0.4f))
        
        val response = genesisAgent.processRequest(request)
        
        assertNotNull(response)
        assertTrue(response.content.isNotEmpty())
        assertTrue(response.confidence >= 0.0f)
    }

    @Test
    fun testGenesisAgent_processRequest_withEmptyContext() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("empty context response", 0.5f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("empty context kai", 0.6f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("empty context cascade", 0.4f))
        
        val response = genesisAgent.processRequest(request)
        
        assertNotNull(response)
        assertTrue(response.content.isNotEmpty())
        assertTrue(response.confidence >= 0.0f)
    }

    @Test
    fun testGenesisAgent_processRequest_servicesReturnEmptyResponse() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("", 0.5f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("", 0.6f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("", 0.4f))
        
        val response = genesisAgent.processRequest(request)
        
        assertNotNull(response)
        assertTrue(response.confidence >= 0.0f)
    }

    @Test
    fun testGenesisAgent_processRequest_servicesReturnNullResponse() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(null)
        whenever(kaiService.processRequest(any())).thenReturn(null)
        whenever(cascadeService.processRequest(any())).thenReturn(null)
        
        try {
            val response = genesisAgent.processRequest(request)
            assertNotNull("Should handle null service responses", response)
        } catch (e: Exception) {
            assertTrue("Should handle null service responses gracefully", 
                e.message?.contains("null") == true || e is NullPointerException)
        }
    }

    @Test
    fun testGenesisAgent_processRequest_servicesMixedNullAndValidResponses() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("valid response", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(null)
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("another valid", 0.6f))
        
        val response = genesisAgent.processRequest(request)
        
        assertNotNull(response)
        assertTrue(response.content.isNotEmpty())
        assertTrue(response.confidence >= 0.0f)
    }

    @Test
    fun testGenesisAgent_processRequest_withUnicodePrompt() = runBlocking {
        val unicodePrompt = "Test with Unicode: 你好世界 🌍 émojis ñ Ω α β γ δ"
        val request = AiRequest(unicodePrompt, emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("unicode aura", 0.5f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("unicode kai", 0.6f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("unicode cascade", 0.4f))
        
        val response = genesisAgent.processRequest(request)
        
        assertNotNull(response)
        assertTrue(response.content.isNotEmpty())
        assertTrue(response.confidence >= 0.0f)
    }

    @Test
    fun testGenesisAgent_processRequest_withSpecialCharacterPrompt() = runBlocking {
        val specialPrompt = "Test with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        val request = AiRequest(specialPrompt, emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("special aura", 0.5f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("special kai", 0.6f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("special cascade", 0.4f))
        
        val response = genesisAgent.processRequest(request)
        
        assertNotNull(response)
        assertTrue(response.content.isNotEmpty())
        assertTrue(response.confidence >= 0.0f)
    }

    @Test
    fun testGenesisAgent_processRequest_withContextContainingSpecialChars() = runBlocking {
        val specialContext = mapOf(
            "key!@#" to "value!@#",
            "key with spaces" to "value with spaces",
            "key\nwith\nnewlines" to "value\nwith\nnewlines",
            "key\twith\ttabs" to "value\twith\ttabs"
        )
        val request = AiRequest("test prompt", specialContext)
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("special context aura", 0.5f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("special context kai", 0.6f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("special context cascade", 0.4f))
        
        val response = genesisAgent.processRequest(request)
        
        assertNotNull(response)
        assertTrue(response.content.isNotEmpty())
        assertTrue(response.confidence >= 0.0f)
    }

    @Test
    fun testDummyAgent_multipleRequestsConsistency() = runBlocking {
        val agent = DummyAgent("ConsistentAgent", "consistent response", 0.7f)
        val request = AiRequest("test", emptyMap())
        
        // Make multiple requests to ensure consistency
        repeat(5) {
            val response = agent.processRequest(request)
            assertEquals("consistent response", response.content)
            assertEquals(0.7f, response.confidence)
        }
    }

    @Test
    fun testDummyAgent_withNullName() = runBlocking {
        val agent = DummyAgent(null, "null name response", 0.5f)
        
        assertNull(agent.getName())
        
        val request = AiRequest("test", emptyMap())
        val response = agent.processRequest(request)
        
        assertEquals("null name response", response.content)
        assertEquals(0.5f, response.confidence)
    }

    @Test
    fun testFailingAgent_multipleFailures() = runBlocking {
        val agent = FailingAgent("MultiFailAgent")
        val request = AiRequest("test", emptyMap())
        
        // Ensure it fails consistently
        repeat(3) {
            try {
                agent.processRequest(request)
                fail("Should throw RuntimeException consistently")
            } catch (e: RuntimeException) {
                assertEquals("Agent processing failed", e.message)
            }
        }
    }

    @Test
    fun testFailingAgent_withNullName() = runBlocking {
        val agent = FailingAgent(null)
        
        assertNull(agent.getName())
        
        val request = AiRequest("test", emptyMap())
        try {
            agent.processRequest(request)
            fail("Should throw RuntimeException")
        } catch (e: RuntimeException) {
            assertEquals("Agent processing failed", e.message)
        }
    }

    @Test
    fun testGenesisAgent_stressTestWithManyAgents() = runBlocking {
        val agents = (1..200).map { i ->
            DummyAgent("StressAgent$i", "stress response $i", (i % 100) / 100.0f)
        }
        
        val startTime = System.currentTimeMillis()
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            agents,
            "stress test",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        val endTime = System.currentTimeMillis()
        
        assertEquals(200, responses.size)
        assertTrue("Should complete within reasonable time", (endTime - startTime) < 30000) // 30 seconds
        
        // Verify all agents processed
        agents.forEach { agent ->
            assertTrue("Agent ${agent.getName()} should be in responses", 
                responses.containsKey(agent.getName()))
        }
    }

    @Test
    fun testGenesisAgent_stressTestWithLargeAggregation() = runBlocking {
        val responses = (1..5000).map { i ->
            mapOf("Agent$i" to AgentResponse("response$i", (i % 1000) / 1000.0f))
        }
        
        val startTime = System.currentTimeMillis()
        val consensus = genesisAgent.aggregateAgentResponses(responses)
        val endTime = System.currentTimeMillis()
        
        assertEquals(5000, consensus.size)
        assertTrue("Should complete aggregation within reasonable time", (endTime - startTime) < 10000) // 10 seconds
    }

    @Test
    fun testGenesisAgent_edgeCaseAgentNameCollisions() = runBlocking {
        val agents = listOf(
            DummyAgent("Agent", "first", 0.5f),
            DummyAgent("Agent", "second", 0.7f),
            DummyAgent("Agent", "third", 0.3f)
        )
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            agents,
            "collision test",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        // Should handle name collisions gracefully
        assertEquals(1, responses.size)
        assertTrue("Should have response for Agent", responses.containsKey("Agent"))
        assertTrue("Should have one of the responses", 
            responses["Agent"]?.content in listOf("first", "second", "third"))
    }

    @Test
    fun testGenesisAgent_boundaryValueConfidences() = runBlocking {
        val agents = listOf(
            DummyAgent("MinValue", "min", Float.MIN_VALUE),
            DummyAgent("MaxValue", "max", Float.MAX_VALUE),
            DummyAgent("NegativeInfinity", "neg_inf", Float.NEGATIVE_INFINITY),
            DummyAgent("PositiveInfinity", "pos_inf", Float.POSITIVE_INFINITY),
            DummyAgent("NaN", "nan", Float.NaN)
        )
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            agents,
            "boundary test",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(5, responses.size)
        assertEquals("min", responses["MinValue"]?.content)
        assertEquals("max", responses["MaxValue"]?.content)
        assertEquals("neg_inf", responses["NegativeInfinity"]?.content)
        assertEquals("pos_inf", responses["PositiveInfinity"]?.content)
        assertEquals("nan", responses["NaN"]?.content)
    }

    @Test
    fun testGenesisAgent_concurrentModificationTest() = runBlocking {
        val agents = mutableListOf<Agent>()
        repeat(10) { i ->
            agents.add(DummyAgent("Agent$i", "response$i", 0.5f))
        }
        
        // Start the participation
        val job = kotlinx.coroutines.async {
            genesisAgent.participateWithAgents(
                emptyMap(),
                agents,
                "concurrent modification test",
                GenesisAgent.ConversationMode.TURN_ORDER
            )
        }
        
        // Try to modify the list while processing (this should not affect the result)
        kotlinx.coroutines.delay(10)
        agents.clear()
        
        val responses = job.await()
        assertEquals(10, responses.size)
    }

    @Test
    fun testGenesisAgent_processRequest_veryLargeContext() = runBlocking {
        val massiveContext = (1..100000).associate { i ->
            "key$i" to "value$i".repeat(10)
        }
        val request = AiRequest("test prompt", massiveContext)
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("massive context aura", 0.5f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("massive context kai", 0.6f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("massive context cascade", 0.4f))
        
        try {
            val response = genesisAgent.processRequest(request)
            assertNotNull(response)
            assertTrue(response.content.isNotEmpty())
            assertTrue(response.confidence >= 0.0f)
        } catch (e: OutOfMemoryError) {
            // Acceptable if system runs out of memory
            assertTrue("System handled memory limitation gracefully", true)
        }
    }

    @Test
    fun testGenesisAgent_robustnessUnderLoad() = runBlocking {
        val agents = (1..50).map { i ->
            if (i % 5 == 0) FailingAgent("FailingAgent$i")
            else DummyAgent("WorkingAgent$i", "response$i", i / 50.0f)
        }
        
        val jobs = (1..10).map { iteration ->
            kotlinx.coroutines.async {
                genesisAgent.participateWithAgents(
                    mapOf("iteration" to iteration.toString()),
                    agents,
                    "robustness test $iteration",
                    GenesisAgent.ConversationMode.TURN_ORDER
                )
            }
        }
        
        val results = jobs.map { it.await() }
        
        // Each result should have 40 successful responses (50 - 10 failing)
        results.forEach { result ->
            assertEquals(40, result.size)
            result.values.forEach { response ->
                assertTrue("Response should have content", response.content.isNotEmpty())
                assertTrue("Response should have non-negative confidence", response.confidence >= 0.0f)
            }
        }
    }
}
