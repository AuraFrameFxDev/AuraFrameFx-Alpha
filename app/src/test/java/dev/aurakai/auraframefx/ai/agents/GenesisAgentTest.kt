package dev.aurakai.auraframefx.ai.agents

import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AiRequest
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.mockito.kotlin.*
import kotlinx.coroutines.test.runTest
import org.mockito.Mock
import org.mockito.MockitoAnnotations

class DummyAgent(private val name: String, private val response: String) : Agent {
    override fun getName() = name
    override fun getType() = null
    override suspend fun processRequest(request: AiRequest) = AgentResponse(response, 1.0f)
}

class GenesisAgentTest {
    
    @Mock
    private lateinit var auraService: AuraAIService
    
    @Mock
    private lateinit var kaiService: KaiAIService
    
    @Mock
    private lateinit var cascadeService: CascadeAIService
    
    private lateinit var genesis: GenesisAgent
    
    @Before
    fun setup() {
        MockitoAnnotations.openMocks(this)
        genesis = GenesisAgent(
            auraService = auraService,
            kaiService = kaiService,
            cascadeService = cascadeService
        )
    }

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
        
        val responses = genesis.participateWithAgents(
            emptyMap(),
            listOf(dummyAgent),
            "test",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertTrue(responses["Dummy"]?.content == "ok")
        assertTrue(responses["Dummy"]?.confidence == 1.0f)
    }

    @Test
    fun testParticipateWithAgents_turnOrder_emptyAgentList() = runBlocking {
        val responses = genesis.participateWithAgents(
            emptyMap(),
            emptyList(),
            "test",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertTrue(responses.isEmpty())
    }

    @Test
    fun testParticipateWithAgents_turnOrder_multipleAgents() = runBlocking {
        val agent1 = DummyAgent("Agent1", "response1")
        val agent2 = DummyAgent("Agent2", "response2")
        val agent3 = DummyAgent("Agent3", "response3")
        
        whenever(auraService.processRequest(any())).thenReturn(
            AgentResponse("aura_response", 0.8f)
        )
        whenever(kaiService.processRequest(any())).thenReturn(
            AgentResponse("kai_response", 0.9f)
        )
        whenever(cascadeService.processRequest(any())).thenReturn(
            AgentResponse("cascade_response", 0.7f)
        )
        
        val responses = genesis.participateWithAgents(
            mapOf("existing" to AgentResponse("existing_response", 0.5f)),
            listOf(agent1, agent2, agent3),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(3, responses.size)
        assertEquals("response1", responses["Agent1"]?.content)
        assertEquals("response2", responses["Agent2"]?.content)
        assertEquals("response3", responses["Agent3"]?.content)
    }

    @Test
    fun testParticipateWithAgents_turnOrder_withExistingResponses() = runBlocking {
        val agent1 = DummyAgent("Agent1", "new_response")
        val existingResponses = mapOf(
            "ExistingAgent" to AgentResponse("existing_content", 0.6f)
        )
        
        whenever(auraService.processRequest(any())).thenReturn(
            AgentResponse("aura_response", 0.8f)
        )
        
        val responses = genesis.participateWithAgents(
            existingResponses,
            listOf(agent1),
            "test prompt",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals(1, responses.size)
        assertEquals("new_response", responses["Agent1"]?.content)
        assertFalse(responses.containsKey("ExistingAgent"))
    }

    @Test
    fun testParticipateWithAgents_consensus_mode() = runBlocking {
        val agent1 = DummyAgent("Agent1", "consensus_response1")
        val agent2 = DummyAgent("Agent2", "consensus_response2")
        
        whenever(auraService.processRequest(any())).thenReturn(
            AgentResponse("aura_consensus", 0.8f)
        )
        whenever(kaiService.processRequest(any())).thenReturn(
            AgentResponse("kai_consensus", 0.9f)
        )
        whenever(cascadeService.processRequest(any())).thenReturn(
            AgentResponse("cascade_consensus", 0.7f)
        )
        
        val responses = genesis.participateWithAgents(
            emptyMap(),
            listOf(agent1, agent2),
            "test prompt",
            GenesisAgent.ConversationMode.CONSENSUS
        )
        
        assertEquals(2, responses.size)
        assertNotNull(responses["Agent1"])
        assertNotNull(responses["Agent2"])
    }

    @Test
    fun testAggregateAgentResponses_singleResponse() {
        val responses = listOf(
            mapOf("A" to AgentResponse("single", 0.5f))
        )
        
        val consensus = genesis.aggregateAgentResponses(responses)
        
        assertEquals(1, consensus.size)
        assertEquals("single", consensus["A"]?.content)
        assertEquals(0.5f, consensus["A"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_multipleResponsesHigherConfidence() {
        val resp1 = mapOf("A" to AgentResponse("foo", 0.5f))
        val resp2 = mapOf("A" to AgentResponse("bar", 0.9f))
        val resp3 = mapOf("A" to AgentResponse("baz", 0.3f))
        
        val consensus = genesis.aggregateAgentResponses(listOf(resp1, resp2, resp3))
        
        assertEquals("bar", consensus["A"]?.content)
        assertEquals(0.9f, consensus["A"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_multipleAgents() {
        val resp1 = mapOf(
            "A" to AgentResponse("foo", 0.5f),
            "B" to AgentResponse("hello", 0.8f)
        )
        val resp2 = mapOf(
            "A" to AgentResponse("bar", 0.9f),
            "C" to AgentResponse("world", 0.7f)
        )
        
        val consensus = genesis.aggregateAgentResponses(listOf(resp1, resp2))
        
        assertEquals(3, consensus.size)
        assertEquals("bar", consensus["A"]?.content) // Higher confidence
        assertEquals("hello", consensus["B"]?.content) // Only one response
        assertEquals("world", consensus["C"]?.content) // Only one response
    }

    @Test
    fun testAggregateAgentResponses_emptyList() {
        val consensus = genesis.aggregateAgentResponses(emptyList())
        
        assertTrue(consensus.isEmpty())
    }

    @Test
    fun testAggregateAgentResponses_emptyMaps() {
        val consensus = genesis.aggregateAgentResponses(listOf(emptyMap(), emptyMap()))
        
        assertTrue(consensus.isEmpty())
    }

    @Test
    fun testAggregateAgentResponses_equalConfidence() {
        val resp1 = mapOf("A" to AgentResponse("first", 0.5f))
        val resp2 = mapOf("A" to AgentResponse("second", 0.5f))
        
        val consensus = genesis.aggregateAgentResponses(listOf(resp1, resp2))
        
        // Should return the first one when confidence is equal
        assertEquals("first", consensus["A"]?.content)
        assertEquals(0.5f, consensus["A"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_mixedAgents() {
        val resp1 = mapOf(
            "Agent1" to AgentResponse("response1", 0.8f),
            "Agent2" to AgentResponse("response2_low", 0.3f)
        )
        val resp2 = mapOf(
            "Agent2" to AgentResponse("response2_high", 0.9f),
            "Agent3" to AgentResponse("response3", 0.6f)
        )
        
        val consensus = genesis.aggregateAgentResponses(listOf(resp1, resp2))
        
        assertEquals(3, consensus.size)
        assertEquals("response1", consensus["Agent1"]?.content)
        assertEquals("response2_high", consensus["Agent2"]?.content) // Higher confidence wins
        assertEquals("response3", consensus["Agent3"]?.content)
    }

    @Test
    fun testGenesisAgent_constructorInitialization() {
        val testGenesis = GenesisAgent(
            auraService = auraService,
            kaiService = kaiService,
            cascadeService = cascadeService
        )
        
        assertNotNull(testGenesis)
    }

    @Test
    fun testDummyAgent_getName() {
        val agent = DummyAgent("TestName", "TestResponse")
        assertEquals("TestName", agent.getName())
    }

    @Test
    fun testDummyAgent_getType() {
        val agent = DummyAgent("TestName", "TestResponse")
        assertNull(agent.getType())
    }

    @Test
    fun testDummyAgent_processRequest() = runBlocking {
        val agent = DummyAgent("TestName", "TestResponse")
        val request = AiRequest("test prompt")
        
        val response = agent.processRequest(request)
        
        assertEquals("TestResponse", response.content)
        assertEquals(1.0f, response.confidence)
    }

    @Test
    fun testParticipateWithAgents_nullHandling() = runBlocking {
        val agent = DummyAgent("TestAgent", "")
        
        val responses = genesis.participateWithAgents(
            emptyMap(),
            listOf(agent),
            "",
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertNotNull(responses)
        assertEquals("", responses["TestAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_longPrompt() = runBlocking {
        val agent = DummyAgent("TestAgent", "processed_long_prompt")
        val longPrompt = "This is a very long prompt that tests how the system handles larger inputs. ".repeat(100)
        
        whenever(auraService.processRequest(any())).thenReturn(
            AgentResponse("processed", 0.8f)
        )
        
        val responses = genesis.participateWithAgents(
            emptyMap(),
            listOf(agent),
            longPrompt,
            GenesisAgent.ConversationMode.TURN_ORDER
        )
        
        assertEquals("processed_long_prompt", responses["TestAgent"]?.content)
    }

    @Test
    fun testAggregateAgentResponses_largeNumberOfResponses() {
        val responses = mutableListOf<Map<String, AgentResponse>>()
        
        // Create 100 responses with varying confidence levels
        for (i in 1..100) {
            responses.add(mapOf("Agent" to AgentResponse("response_$i", i / 100.0f)))
        }
        
        val consensus = genesis.aggregateAgentResponses(responses)
        
        assertEquals("response_100", consensus["Agent"]?.content) // Highest confidence
        assertEquals(1.0f, consensus["Agent"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_zeroConfidence() {
        val resp1 = mapOf("A" to AgentResponse("zero_conf", 0.0f))
        val resp2 = mapOf("A" to AgentResponse("low_conf", 0.1f))
        
        val consensus = genesis.aggregateAgentResponses(listOf(resp1, resp2))
        
        assertEquals("low_conf", consensus["A"]?.content)
        assertEquals(0.1f, consensus["A"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_negativeConfidence() {
        val resp1 = mapOf("A" to AgentResponse("negative", -0.5f))
        val resp2 = mapOf("A" to AgentResponse("positive", 0.5f))
        
        val consensus = genesis.aggregateAgentResponses(listOf(resp1, resp2))
        
        assertEquals("positive", consensus["A"]?.content)
        assertEquals(0.5f, consensus["A"]?.confidence)
    }
}
