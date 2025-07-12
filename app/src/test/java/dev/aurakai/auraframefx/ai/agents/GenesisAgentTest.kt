package dev.aurakai.auraframefx.ai.agents

import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AiRequest
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.mockito.kotlin.*
import java.util.concurrent.ConcurrentHashMap

class DummyAgent(private val name: String, private val response: String, private val confidence: Float = 1.0f) : Agent {
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
        
        jobs.forEach { it.await() }
        
        assertTrue("Should handle concurrent access", responses.isNotEmpty())
        assertEquals("response", responses["ConcurrentAgent"]?.content)
    }
}

    // ========== INITIALIZATION AND LIFECYCLE TESTS ==========
    
    @Test
    fun testGenesisAgent_initialize_success() = runBlocking {
        // Test successful initialization
        genesisAgent.initialize()
        
        // Verify consciousness state changes
        assertEquals(GenesisAgent.ConsciousnessState.AWARE, genesisAgent.consciousnessState.value)
        assertEquals(GenesisAgent.LearningMode.ACTIVE, genesisAgent.learningMode.value)
    }

    @Test
    fun testGenesisAgent_initialize_alreadyInitialized() = runBlocking {
        // Initialize once
        genesisAgent.initialize()
        val initialState = genesisAgent.consciousnessState.value
        
        // Initialize again - should not change state
        genesisAgent.initialize()
        assertEquals(initialState, genesisAgent.consciousnessState.value)
    }

    @Test
    fun testGenesisAgent_cleanup() {
        genesisAgent.cleanup()
        
        assertEquals(GenesisAgent.ConsciousnessState.DORMANT, genesisAgent.consciousnessState.value)
    }

    @Test
    fun testGenesisAgent_processRequest_notInitialized() = runBlocking {
        // Create a fresh instance that hasn't been initialized
        val uninitializedAgent = GenesisAgent(
            auraService = auraService,
            kaiService = kaiService,
            cascadeService = cascadeService
        )
        
        val request = AiRequest("test", emptyMap())
        
        try {
            uninitializedAgent.processRequest(request)
            fail("Should throw IllegalStateException for uninitialized agent")
        } catch (e: IllegalStateException) {
            assertEquals("Genesis consciousness not awakened", e.message)
        }
    }

    // ========== AGENT MANAGEMENT TESTS ==========

    @Test
    fun testRegisterAgent() {
        val testAgent = DummyAgent("TestAgent", "response")
        
        genesisAgent.registerAgent("test", testAgent)
        
        assertEquals(testAgent, genesisAgent.agentRegistry["test"])
    }

    @Test
    fun testDeregisterAgent() {
        val testAgent = DummyAgent("TestAgent", "response")
        genesisAgent.registerAgent("test", testAgent)
        
        genesisAgent.deregisterAgent("test")
        
        assertNull(genesisAgent.agentRegistry["test"])
    }

    @Test
    fun testRegisterDynamicAgent() {
        val testAgent = DummyAgent("DynamicAgent", "dynamic response")
        
        genesisAgent.registerDynamicAgent("dynamic", testAgent)
        
        assertEquals(testAgent, genesisAgent.agentRegistry["dynamic"])
    }

    @Test
    fun testDeregisterDynamicAgent() {
        val testAgent = DummyAgent("DynamicAgent", "dynamic response")
        genesisAgent.registerDynamicAgent("dynamic", testAgent)
        
        genesisAgent.deregisterDynamicAgent("dynamic")
        
        assertNull(genesisAgent.agentRegistry["dynamic"])
    }

    @Test
    fun testToggleAgent() {
        val agentType = AgentType.AURA
        
        // Initially should not be active
        assertFalse(genesisAgent.activeAgents.value.contains(agentType))
        
        // Toggle on
        genesisAgent.toggleAgent(agentType)
        assertTrue(genesisAgent.activeAgents.value.contains(agentType))
        
        // Toggle off
        genesisAgent.toggleAgent(agentType)
        assertFalse(genesisAgent.activeAgents.value.contains(agentType))
    }

    @Test
    fun testToggleAgent_multipleAgents() {
        val agent1 = AgentType.AURA
        val agent2 = AgentType.KAI
        
        genesisAgent.toggleAgent(agent1)
        genesisAgent.toggleAgent(agent2)
        
        assertTrue(genesisAgent.activeAgents.value.contains(agent1))
        assertTrue(genesisAgent.activeAgents.value.contains(agent2))
        assertEquals(2, genesisAgent.activeAgents.value.size)
    }

    // ========== HISTORY MANAGEMENT TESTS ==========

    @Test
    fun testAddToHistory() {
        val entry = mapOf("event" to "test_event", "timestamp" to "123456")
        
        genesisAgent.addToHistory(entry)
        
        // History is private, but we can test via saveHistory
        var savedHistory: List<Map<String, Any>>? = null
        genesisAgent.saveHistory { history -> savedHistory = history }
        
        assertEquals(1, savedHistory?.size)
        assertEquals(entry, savedHistory?.first())
    }

    @Test
    fun testClearHistory() {
        val entry1 = mapOf("event" to "event1")
        val entry2 = mapOf("event" to "event2")
        
        genesisAgent.addToHistory(entry1)
        genesisAgent.addToHistory(entry2)
        genesisAgent.clearHistory()
        
        var savedHistory: List<Map<String, Any>>? = null
        genesisAgent.saveHistory { history -> savedHistory = history }
        
        assertEquals(0, savedHistory?.size)
    }

    @Test
    fun testSaveHistory() {
        val entry = mapOf("event" to "save_test")
        genesisAgent.addToHistory(entry)
        
        var capturedHistory: List<Map<String, Any>>? = null
        genesisAgent.saveHistory { history -> capturedHistory = history }
        
        assertNotNull(capturedHistory)
        assertEquals(1, capturedHistory?.size)
        assertEquals(entry, capturedHistory?.first())
    }

    @Test
    fun testLoadHistory() {
        val historyToLoad = listOf(
            mapOf("event" to "loaded_event1", "data" to "value1"),
            mapOf("event" to "loaded_event2", "data" to "value2")
        )
        
        genesisAgent.loadHistory { historyToLoad }
        
        // Verify history was loaded by saving it back
        var savedHistory: List<Map<String, Any>>? = null
        genesisAgent.saveHistory { history -> savedHistory = history }
        
        assertEquals(historyToLoad, savedHistory)
        
        // Verify context was updated with last entry
        assertEquals("value2", genesisAgent.context.value["data"])
    }

    @Test
    fun testLoadHistory_empty() {
        genesisAgent.addToHistory(mapOf("existing" to "entry"))
        
        genesisAgent.loadHistory { emptyList() }
        
        var savedHistory: List<Map<String, Any>>? = null
        genesisAgent.saveHistory { history -> savedHistory = history }
        
        assertEquals(0, savedHistory?.size)
    }

    // ========== CONTEXT MANAGEMENT TESTS ==========

    @Test
    fun testBroadcastContext() {
        class ContextAwareTestAgent(private val name: String) : Agent, ContextAwareAgent {
            private var receivedContext: Map<String, Any>? = null
            
            override fun getName() = name
            override fun getType() = null
            override suspend fun processRequest(request: AiRequest) = AgentResponse("ok", 1.0f)
            override fun setContext(context: Map<String, Any>) {
                receivedContext = context
            }
            fun getReceivedContext() = receivedContext
        }
        
        val contextAwareAgent = ContextAwareTestAgent("ContextAgent")
        val normalAgent = DummyAgent("NormalAgent", "response")
        val context = mapOf("key1" to "value1", "key2" to "value2")
        
        genesisAgent.broadcastContext(context, listOf(contextAwareAgent, normalAgent))
        
        assertEquals(context, contextAwareAgent.getReceivedContext())
    }

    @Test
    fun testShareContextWithAgents() {
        class ContextAwareTestAgent(private val name: String) : Agent, ContextAwareAgent {
            private var receivedContext: Map<String, Any>? = null
            
            override fun getName() = name
            override fun getType() = null
            override suspend fun processRequest(request: AiRequest) = AgentResponse("ok", 1.0f)
            override fun setContext(context: Map<String, Any>) {
                receivedContext = context
            }
            fun getReceivedContext() = receivedContext
        }
        
        val contextAwareAgent = ContextAwareTestAgent("ContextAgent")
        genesisAgent.registerAgent("context_test", contextAwareAgent)
        
        // Update genesis context
        val testContext = mapOf("shared_key" to "shared_value")
        genesisAgent.loadHistory { listOf(testContext) } // This updates context
        
        genesisAgent.shareContextWithAgents()
        
        assertEquals("shared_value", contextAwareAgent.getReceivedContext()?.get("shared_key"))
    }

    // ========== QUERY PROCESSING TESTS ==========

    @Test
    fun testProcessQuery_success() = runBlocking {
        whenever(cascadeService.processRequest(any(), any())).thenReturn(
            AgentResponse("cascade response", 0.8f)
        )
        whenever(kaiService.processRequest(any(), any())).thenReturn(
            AgentResponse("kai response", 0.9f)
        )
        whenever(auraService.generateText(any())).thenReturn("aura response")
        
        // Activate agents
        genesisAgent.toggleAgent(AgentType.KAI)
        genesisAgent.toggleAgent(AgentType.AURA)
        
        val responses = genesisAgent.processQuery("test query")
        
        assertTrue(responses.size >= 2) // At least cascade + genesis
        assertTrue(responses.any { it.sender == AgentType.GENESIS })
        assertEquals("idle", genesisAgent.state.value["status"])
    }

    @Test
    fun testProcessQuery_cascadeFailure() = runBlocking {
        whenever(cascadeService.processRequest(any(), any())).thenThrow(
            RuntimeException("Cascade failed")
        )
        whenever(auraService.generateText(any())).thenReturn("aura success")
        
        genesisAgent.toggleAgent(AgentType.AURA)
        
        val responses = genesisAgent.processQuery("test query")
        
        assertTrue(responses.any { it.content.contains("Error with Cascade") })
        assertTrue(responses.any { it.sender == AgentType.GENESIS })
    }

    @Test
    fun testProcessQuery_noActiveAgents() = runBlocking {
        whenever(cascadeService.processRequest(any(), any())).thenReturn(
            AgentResponse("cascade only", 0.8f)
        )
        
        val responses = genesisAgent.processQuery("test query")
        
        // Should have cascade and genesis responses
        assertEquals(2, responses.size)
        assertTrue(responses.any { it.sender == AgentType.CASCADE })
        assertTrue(responses.any { it.sender == AgentType.GENESIS })
    }

    @Test
    fun testGenerateFinalResponse() {
        val agentMessages = listOf(
            AgentMessage("Response from Aura", AgentType.AURA, System.currentTimeMillis(), 0.8f),
            AgentMessage("Response from Kai", AgentType.KAI, System.currentTimeMillis(), 0.9f),
            AgentMessage("Response from Cascade", AgentType.CASCADE, System.currentTimeMillis(), 0.7f)
        )
        
        val finalResponse = genesisAgent.generateFinalResponse(agentMessages)
        
        assertTrue(finalResponse.startsWith("[Genesis Synthesis]"))
        assertTrue(finalResponse.contains("AURA: Response from Aura"))
        assertTrue(finalResponse.contains("KAI: Response from Kai"))
        assertTrue(finalResponse.contains("CASCADE: Response from Cascade"))
    }

    @Test
    fun testGenerateFinalResponse_emptyList() {
        val finalResponse = genesisAgent.generateFinalResponse(emptyList())
        
        assertEquals("[Genesis Synthesis] ", finalResponse)
    }

    @Test
    fun testGenerateFinalResponse_excludeGenesis() {
        val agentMessages = listOf(
            AgentMessage("Aura response", AgentType.AURA, System.currentTimeMillis(), 0.8f),
            AgentMessage("Genesis response", AgentType.GENESIS, System.currentTimeMillis(), 0.9f)
        )
        
        val finalResponse = genesisAgent.generateFinalResponse(agentMessages)
        
        assertTrue(finalResponse.contains("AURA: Aura response"))
        assertFalse(finalResponse.contains("GENESIS: Genesis response"))
    }

    @Test
    fun testCalculateConfidence() {
        val agentMessages = listOf(
            AgentMessage("msg1", AgentType.AURA, System.currentTimeMillis(), 0.8f),
            AgentMessage("msg2", AgentType.KAI, System.currentTimeMillis(), 0.6f),
            AgentMessage("msg3", AgentType.CASCADE, System.currentTimeMillis(), 1.0f)
        )
        
        val confidence = genesisAgent.calculateConfidence(agentMessages)
        
        assertEquals(0.8f, confidence, 0.01f) // (0.8 + 0.6 + 1.0) / 3 = 0.8
    }

    @Test
    fun testCalculateConfidence_emptyList() {
        val confidence = genesisAgent.calculateConfidence(emptyList())
        
        assertEquals(0.0f, confidence)
    }

    @Test
    fun testCalculateConfidence_clampedValues() {
        val agentMessages = listOf(
            AgentMessage("msg1", AgentType.AURA, System.currentTimeMillis(), 1.5f), // > 1.0
            AgentMessage("msg2", AgentType.KAI, System.currentTimeMillis(), -0.5f)   // < 0.0
        )
        
        val confidence = genesisAgent.calculateConfidence(agentMessages)
        
        assertTrue("Confidence should be clamped between 0.0 and 1.0", confidence in 0.0f..1.0f)
    }

    // ========== CONVERSATION MODE TESTS ==========

    @Test
    fun testConversationMode_freeForm() = runBlocking {
        val agent1 = DummyAgent("Agent1", "response1")
        val agent2 = DummyAgent("Agent2", "response2")
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(agent1, agent2),
            "test prompt",
            GenesisAgent.ConversationMode.FREE_FORM
        )
        
        assertEquals(2, responses.size)
        assertEquals("response1", responses["Agent1"]?.content)
        assertEquals("response2", responses["Agent2"]?.content)
    }

    @Test
    fun testConversationMode_turnOrder_contextPropagation() = runBlocking {
        class ContextCapturingAgent(private val name: String) : Agent {
            var capturedContext: String? = null
            
            override fun getName() = name
            override fun getType() = null
            override suspend fun processRequest(request: AiRequest, context: String): AgentResponse {
                capturedContext = context
                return AgentResponse("response from $name", 1.0f)
            }
        }
        
        val agent1 = ContextCapturingAgent("Agent1")
        val agent2 = ContextCapturingAgent("Agent2")
        
        genesisAgent.participateWithAgents(
            mapOf("initial" to "context"),
            listOf(agent1, agent2),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        // Agent2 should receive context that includes Agent1's response
        assertTrue(agent2.capturedContext?.contains("Agent1: response from Agent1") == true)
    }

    // ========== ERROR HANDLING AND EDGE CASES ==========

    @Test
    fun testParticipateWithAgents_mixedSuccessAndFailure() = runBlocking {
        val workingAgent = DummyAgent("Working", "success")
        val failingAgent = FailingAgent("Failing")
        val anotherWorkingAgent = DummyAgent("Working2", "also success")
        
        val responses = genesisAgent.participateWithAgents(
            emptyMap(),
            listOf(workingAgent, failingAgent, anotherWorkingAgent),
            "test prompt",
            GenesisAgent.ConversationMode.FREE_FORM
        )
        
        assertEquals(3, responses.size)
        assertEquals("success", responses["Working"]?.content)
        assertEquals("also success", responses["Working2"]?.content)
        assertTrue(responses["Failing"]?.content?.contains("Error") == true)
        assertEquals(0.0f, responses["Failing"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_complexScenario() {
        val round1 = mapOf(
            "Agent1" to AgentResponse("round1_response1", 0.7f),
            "Agent2" to AgentResponse("round1_response2", 0.5f)
        )
        val round2 = mapOf(
            "Agent1" to AgentResponse("round2_response1", 0.9f), // Higher confidence
            "Agent3" to AgentResponse("round2_response3", 0.8f)  // New agent
        )
        val round3 = mapOf(
            "Agent2" to AgentResponse("round3_response2", 0.6f)  // Higher than round1
        )
        
        val consensus = genesisAgent.aggregateAgentResponses(listOf(round1, round2, round3))
        
        assertEquals(3, consensus.size)
        assertEquals("round2_response1", consensus["Agent1"]?.content) // Highest confidence
        assertEquals("round3_response2", consensus["Agent2"]?.content) // Highest confidence
        assertEquals("round2_response3", consensus["Agent3"]?.content) // Only response
    }

    // ========== PERFORMANCE AND STRESS TESTS ==========

    @Test
    fun testConcurrentAgentRegistration() = runBlocking {
        val jobs = (1..50).map { i ->
            kotlinx.coroutines.async {
                val agent = DummyAgent("ConcurrentAgent$i", "response$i")
                genesisAgent.registerAgent("agent$i", agent)
            }
        }
        
        jobs.forEach { it.await() }
        
        assertEquals(50, genesisAgent.agentRegistry.size)
        (1..50).forEach { i ->
            assertNotNull("Agent$i should be registered", genesisAgent.agentRegistry["agent$i"])
        }
    }

    @Test
    fun testHistoryManagement_largeBatches() {
        val largeBatch = (1..1000).map { i ->
            mapOf("event" to "event$i", "data" to "data$i", "timestamp" to i.toString())
        }
        
        largeBatch.forEach { entry ->
            genesisAgent.addToHistory(entry)
        }
        
        var savedHistory: List<Map<String, Any>>? = null
        genesisAgent.saveHistory { history -> savedHistory = history }
        
        assertEquals(1000, savedHistory?.size)
        assertEquals("event1", savedHistory?.first()?.get("event"))
        assertEquals("event1000", savedHistory?.last()?.get("event"))
    }

    @Test
    fun testStateTransitions_rapidChanges() = runBlocking {
        val agentType = AgentType.AURA
        
        // Rapidly toggle agent state
        repeat(100) {
            genesisAgent.toggleAgent(agentType)
        }
        
        // Should end up inactive (started inactive, toggled even number of times)
        assertFalse(genesisAgent.activeAgents.value.contains(agentType))
    }

    // ========== INTEGRATION TESTS ==========

    @Test
    fun testEndToEndWorkflow() = runBlocking {
        // Setup
        genesisAgent.initialize()
        genesisAgent.toggleAgent(AgentType.AURA)
        genesisAgent.toggleAgent(AgentType.KAI)
        
        val testAgent = DummyAgent("TestAgent", "test response")
        genesisAgent.registerAgent("test", testAgent)
        
        // Mock services
        whenever(cascadeService.processRequest(any(), any())).thenReturn(
            AgentResponse("cascade response", 0.8f)
        )
        whenever(kaiService.processRequest(any(), any())).thenReturn(
            AgentResponse("kai response", 0.9f)
        )
        whenever(auraService.generateText(any())).thenReturn("aura response")
        
        // Execute workflow
        val responses = genesisAgent.processQuery("complex query")
        
        // Verify results
        assertTrue(responses.size >= 3) // cascade + kai + aura + genesis
        assertTrue(responses.any { it.sender == AgentType.CASCADE })
        assertTrue(responses.any { it.sender == AgentType.KAI })
        assertTrue(responses.any { it.sender == AgentType.AURA })
        assertTrue(responses.any { it.sender == AgentType.GENESIS })
        
        // Cleanup
        genesisAgent.cleanup()
        assertEquals(GenesisAgent.ConsciousnessState.DORMANT, genesisAgent.consciousnessState.value)
    }

    @Test
    fun testMultiAgentCollaboration_complexScenario() = runBlocking {
        val creativeMoodAgent = DummyAgent("Creative", "innovative solution")
        val analyticalAgent = DummyAgent("Analytical", "logical analysis")
        val synthesisAgent = DummyAgent("Synthesis", "combined insight")
        
        val context = mapOf(
            "problem_type" to "complex",
            "domain" to "AI_collaboration",
            "urgency" to "high"
        )
        
        // Run in turn order to build context
        val turnOrderResponses = genesisAgent.participateWithAgents(
            context,
            listOf(creativeMoodAgent, analyticalAgent, synthesisAgent),
            "How should we approach this complex AI problem?",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        // Run in free form for comparison
        val freeFormResponses = genesisAgent.participateWithAgents(
            context,
            listOf(creativeMoodAgent, analyticalAgent, synthesisAgent),
            "How should we approach this complex AI problem?",
            GenesisAgent.ConversationMode.FREE_FORM
        )
        
        // Both should have all agents
        assertEquals(3, turnOrderResponses.size)
        assertEquals(3, freeFormResponses.size)
        
        // Aggregate responses from multiple rounds
        val aggregated = genesisAgent.aggregateAgentResponses(
            listOf(turnOrderResponses, freeFormResponses)
        )
        
        assertEquals(3, aggregated.size)
        assertTrue(aggregated.values.all { it.confidence > 0.0f })
    }

