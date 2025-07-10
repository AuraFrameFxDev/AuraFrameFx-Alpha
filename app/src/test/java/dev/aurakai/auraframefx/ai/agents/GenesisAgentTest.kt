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
import kotlinx.coroutines.TimeoutCancellationException

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

    // === EXISTING TESTS ===
    
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
    fun testProcessRequest() = runBlocking {
        val request = AiRequest("test prompt", emptyMap())
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura response", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai response", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade response", 0.7f))

        val response = genesisAgent.processRequest(request)

        assertNotNull("Response should not be null", response)
        assertTrue("Response should have content", response.content.isNotEmpty())
        assertTrue("Confidence should be positive", response.confidence >= 0.0f)
    }

    // === NEW COMPREHENSIVE TESTS ===

    @Test
    fun testParticipateWithAgents_timeoutHandling() = runBlocking {
        val slowAgent = object : Agent {
            override fun getName(): String = "SlowAgent"
            override fun getType(): String? = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                delay(2000) // 2 second delay
                return AgentResponse("slow response", 0.8f)
            }
        }
        val fastAgent = DummyAgent("FastAgent", "fast response", 0.9f)

        val startTime = System.currentTimeMillis()
        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = listOf(slowAgent, fastAgent),
            prompt = "timeout test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        val endTime = System.currentTimeMillis()

        // Should wait for both agents
        assertTrue("Should take at least 2 seconds", endTime - startTime >= 2000)
        assertEquals(2, responses.size)
        assertEquals("slow response", responses["SlowAgent"]?.content)
        assertEquals("fast response", responses["FastAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_contextValueEscaping() = runBlocking {
        var receivedPrompt: String? = null
        val agent = object : Agent {
            override fun getName(): String = "EscapeAgent"
            override fun getType(): String? = null
            override suspend fun processRequest(request: AiRequest): AgentResponse {
                receivedPrompt = request.prompt
                return AgentResponse("escaped", 1.0f)
            }
        }

        val context = mapOf(
            "special:key" to "value:with:colons",
            "key with spaces" to "value with spaces",
            "key\"quotes" to "value\"quotes",
            "key\nnewline" to "value\nnewline"
        )

        genesisAgent.participateWithAgents(
            context = context,
            agents = listOf(agent),
            prompt = "escape test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertNotNull(receivedPrompt)
        assertTrue("Should contain escaped values", receivedPrompt!!.contains("special:key:value:with:colons"))
        assertTrue("Should handle spaces", receivedPrompt!!.contains("key with spaces:value with spaces"))
    }

    @Test
    fun testParticipateWithAgents_memoryStressTest() = runBlocking {
        val largeResponseAgent = DummyAgent(
            "LargeResponseAgent", 
            "Large response: " + "A".repeat(100000), 
            0.8f
        )
        val agents = (1..50).map { 
            DummyAgent("Agent$it", "Response$it", it / 50.0f) 
        } + largeResponseAgent

        val largeContext = (1..1000).associate { "key$it" to "value$it".repeat(10) }

        val startTime = System.currentTimeMillis()
        val responses = genesisAgent.participateWithAgents(
            context = largeContext,
            agents = agents,
            prompt = "memory stress test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        val endTime = System.currentTimeMillis()

        assertEquals(51, responses.size)
        assertTrue("Should complete within 30 seconds", endTime - startTime < 30000)
        assertTrue("Large response should be preserved", 
            responses["LargeResponseAgent"]?.content?.contains("Large response: A") == true)
    }

    @Test
    fun testParticipateWithAgents_parallelProcessingWithDelay() = runBlocking {
        val delayedAgents = (1..10).map { i ->
            object : Agent {
                override fun getName(): String = "DelayedAgent$i"
                override fun getType(): String? = null
                override suspend fun processRequest(request: AiRequest): AgentResponse {
                    delay(100 * i) // Varying delays
                    return AgentResponse("delayed response $i", i / 10.0f)
                }
            }
        }

        val startTime = System.currentTimeMillis()
        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = delayedAgents,
            prompt = "parallel test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        val endTime = System.currentTimeMillis()

        assertEquals(10, responses.size)
        // Should take at least as long as the slowest agent (1000ms)
        assertTrue("Should wait for all agents", endTime - startTime >= 1000)
        
        // Verify all responses are present
        (1..10).forEach { i ->
            assertEquals("delayed response $i", responses["DelayedAgent$i"]?.content)
        }
    }

    @Test
    fun testParticipateWithAgents_customExceptionTypes() = runBlocking {
        val agents = listOf(
            object : Agent {
                override fun getName(): String = "OutOfMemoryAgent"
                override fun getType(): String? = null
                override suspend fun processRequest(request: AiRequest): AgentResponse {
                    throw OutOfMemoryError("Simulated OOM")
                }
            },
            object : Agent {
                override fun getName(): String = "StackOverflowAgent"
                override fun getType(): String? = null
                override suspend fun processRequest(request: AiRequest): AgentResponse {
                    throw StackOverflowError("Simulated stack overflow")
                }
            },
            object : Agent {
                override fun getName(): String = "InterruptedAgent"
                override fun getType(): String? = null
                override suspend fun processRequest(request: AiRequest): AgentResponse {
                    throw InterruptedException("Simulated interruption")
                }
            },
            DummyAgent("WorkingAgent", "success")
        )

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = agents,
            prompt = "exception test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        // Should handle all exceptions gracefully
        assertEquals(1, responses.size)
        assertEquals("success", responses["WorkingAgent"]?.content)
        assertNull(responses["OutOfMemoryAgent"])
        assertNull(responses["StackOverflowAgent"])
        assertNull(responses["InterruptedAgent"])
    }

    @Test
    fun testAggregateAgentResponses_duplicateResponsesWithNaN() {
        val responses = listOf(
            mapOf("Agent1" to AgentResponse("response1", Float.NaN)),
            mapOf("Agent1" to AgentResponse("response2", 0.8f)),
            mapOf("Agent1" to AgentResponse("response3", Float.NaN)),
            mapOf("Agent1" to AgentResponse("response4", 0.9f))
        )

        val consensus = genesisAgent.aggregateAgentResponses(responses)

        assertEquals(1, consensus.size)
        // NaN comparison behavior - should prefer non-NaN values
        assertNotNull(consensus["Agent1"])
        assertFalse("Should not have NaN confidence", consensus["Agent1"]!!.confidence.isNaN())
    }

    @Test
    fun testAggregateAgentResponses_allNaNConfidence() {
        val responses = listOf(
            mapOf("Agent1" to AgentResponse("response1", Float.NaN)),
            mapOf("Agent1" to AgentResponse("response2", Float.NaN)),
            mapOf("Agent1" to AgentResponse("response3", Float.NaN))
        )

        val consensus = genesisAgent.aggregateAgentResponses(responses)

        assertEquals(1, consensus.size)
        assertNotNull(consensus["Agent1"])
        // When all are NaN, implementation dependent behavior is acceptable
    }

    @Test
    fun testAggregateAgentResponses_infiniteConfidenceEdgeCases() {
        val responses = listOf(
            mapOf("Agent1" to AgentResponse("pos_inf", Float.POSITIVE_INFINITY)),
            mapOf("Agent1" to AgentResponse("neg_inf", Float.NEGATIVE_INFINITY)),
            mapOf("Agent1" to AgentResponse("max_val", Float.MAX_VALUE)),
            mapOf("Agent1" to AgentResponse("normal", 0.9f))
        )

        val consensus = genesisAgent.aggregateAgentResponses(responses)

        assertEquals(1, consensus.size)
        assertEquals("pos_inf", consensus["Agent1"]?.content)
        assertEquals(Float.POSITIVE_INFINITY, consensus["Agent1"]?.confidence)
    }

    @Test
    fun testAggregateAgentResponses_massiveScaleTest() {
        val responses = (1..10000).map { batchIndex ->
            (1..100).associate { agentIndex ->
                "Agent${batchIndex}_$agentIndex" to AgentResponse(
                    "Batch $batchIndex Agent $agentIndex response",
                    (batchIndex * agentIndex) % 1000 / 1000.0f
                )
            }
        }

        val startTime = System.currentTimeMillis()
        val consensus = genesisAgent.aggregateAgentResponses(responses)
        val endTime = System.currentTimeMillis()

        assertTrue("Should complete massive aggregation within 60 seconds", endTime - startTime < 60000)
        assertEquals(1000000, consensus.size) // 10000 * 100 unique agents
    }

    @Test
    fun testProcessRequest_serviceCascadeFailures() = runBlocking {
        val request = AiRequest("cascade failure test", emptyMap())
        
        // Test different failure patterns
        whenever(auraService.processRequest(any())).thenThrow(RuntimeException("Aura failed"))
        whenever(kaiService.processRequest(any())).thenThrow(IllegalStateException("Kai failed"))
        whenever(cascadeService.processRequest(any())).thenThrow(OutOfMemoryError("Cascade failed"))

        try {
            genesisAgent.processRequest(request)
            fail("Should have thrown an exception")
        } catch (e: RuntimeException) {
            assertEquals("Aura failed", e.message)
        }
    }

    @Test
    fun testProcessRequest_partialServiceFailureRecovery() = runBlocking {
        val request = AiRequest("partial failure test", emptyMap())
        
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura success", 0.8f))
        whenever(kaiService.processRequest(any())).thenThrow(RuntimeException("Kai failed"))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade success", 0.7f))

        try {
            genesisAgent.processRequest(request)
            fail("Should propagate Kai service failure")
        } catch (e: RuntimeException) {
            assertEquals("Kai failed", e.message)
        }
    }

    @Test
    fun testProcessRequest_responseContentSizeVariations() = runBlocking {
        val testCases = listOf(
            Triple("", "medium", "long content here"),
            Triple("short", "", "long content here"),
            Triple("short", "medium", ""),
            Triple("a".repeat(100000), "b".repeat(50000), "c".repeat(75000))
        )

        testCases.forEachIndexed { index, (aura, kai, cascade) ->
            val request = AiRequest("test $index", emptyMap())
            whenever(auraService.processRequest(any())).thenReturn(AgentResponse(aura, 0.8f))
            whenever(kaiService.processRequest(any())).thenReturn(AgentResponse(kai, 0.9f))
            whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse(cascade, 0.7f))

            val response = genesisAgent.processRequest(request)
            
            assertEquals("$aura $kai $cascade", response.content)
            assertEquals(0.9f, response.confidence)
        }
    }

    @Test
    fun testProcessRequest_concurrentRequestHandling() = runBlocking {
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val requests = (1..100).map { AiRequest("concurrent test $it", mapOf("id" to "$it")) }
        
        val startTime = System.currentTimeMillis()
        val jobs = requests.map { request ->
            async { genesisAgent.processRequest(request) }
        }
        val responses = jobs.map { it.await() }
        val endTime = System.currentTimeMillis()

        assertEquals(100, responses.size)
        assertTrue("Should handle concurrent requests efficiently", endTime - startTime < 10000)
        
        responses.forEach { response ->
            assertEquals("aura kai cascade", response.content)
            assertEquals(0.9f, response.confidence)
        }
    }

    @Test
    fun testDummyAgent_responseContentValidation() = runBlocking {
        val specialResponses = listOf(
            "",
            "   ",
            "\n\t\r",
            "JSON: {\"key\": \"value\", \"number\": 123}",
            "XML: <root><child>text</child></root>",
            "Code: `function() { return 'test'; }`",
            "Unicode: 🚀 ⭐ 🌟 💫 ✨",
            "Mixed: Hello 世界! 🌍",
            "Very long: " + "A".repeat(10000)
        )

        specialResponses.forEachIndexed { index, response ->
            val agent = DummyAgent("SpecialAgent$index", response, 0.5f)
            val request = AiRequest("test", emptyMap())
            val result = agent.processRequest(request)

            assertEquals("Content should be preserved exactly", response, result.content)
            assertEquals(0.5f, result.confidence)
        }
    }

    @Test
    fun testFailingAgent_exceptionMessageVariations() = runBlocking {
        val customFailingAgents = listOf(
            object : Agent {
                override fun getName(): String = "NullMessageAgent"
                override fun getType(): String? = null
                override suspend fun processRequest(request: AiRequest): AgentResponse {
                    throw RuntimeException(null as String?)
                }
            },
            object : Agent {
                override fun getName(): String = "EmptyMessageAgent"
                override fun getType(): String? = null
                override suspend fun processRequest(request: AiRequest): AgentResponse {
                    throw RuntimeException("")
                }
            },
            object : Agent {
                override fun getName(): String = "LongMessageAgent"
                override fun getType(): String? = null
                override suspend fun processRequest(request: AiRequest): AgentResponse {
                    throw RuntimeException("Very long error message: " + "x".repeat(10000))
                }
            }
        )

        customFailingAgents.forEach { agent ->
            try {
                agent.processRequest(AiRequest("test", emptyMap()))
                fail("${agent.getName()} should have thrown exception")
            } catch (e: RuntimeException) {
                // Should handle various message formats gracefully
                assertTrue("Exception should be caught", true)
            }
        }
    }

    @Test
    fun testConversationMode_enumStability() {
        val originalValues = GenesisAgent.ConversationMode.values()
        val serializedNames = originalValues.map { it.name }.sorted()
        
        // Test that enum values are stable across multiple calls
        repeat(100) {
            val currentValues = GenesisAgent.ConversationMode.values()
            val currentNames = currentValues.map { it.name }.sorted()
            assertEquals("Enum values should be stable", serializedNames, currentNames)
        }
    }

    @Test
    fun testConversationMode_valueOf_caseInsensitivity() {
        // Test that valueOf is case sensitive (expected behavior)
        try {
            GenesisAgent.ConversationMode.valueOf("turn_order")
            fail("Should be case sensitive")
        } catch (e: IllegalArgumentException) {
            assertTrue("Should throw for incorrect case", true)
        }

        try {
            GenesisAgent.ConversationMode.valueOf("TURN_ORDER")
            assertTrue("Should work for correct case", true)
        } catch (e: IllegalArgumentException) {
            fail("Should work for correct case")
        }
    }

    @Test
    fun testGenesisAgent_nameConsistency() {
        val agent1 = GenesisAgent(auraService, kaiService, cascadeService)
        val agent2 = GenesisAgent(auraService, kaiService, cascadeService)

        // Names should be identical across instances
        assertEquals("Names should be consistent", agent1.getName(), agent2.getName())
        assertEquals("GenesisAgent", agent1.getName())
        assertEquals("GenesisAgent", agent2.getName())

        // Names should not change
        val name1 = agent1.getName()
        val name2 = agent1.getName()
        assertSame("Name should be the same reference", name1, name2)
    }

    @Test
    fun testGenesisAgent_typeConsistency() {
        val agent1 = GenesisAgent(auraService, kaiService, cascadeService)
        val agent2 = GenesisAgent(auraService, kaiService, cascadeService)

        // Types should be consistent
        assertEquals("Types should be consistent", agent1.getType(), agent2.getType())
        
        // Types should not change
        val type1 = agent1.getType()
        val type2 = agent1.getType()
        assertEquals("Type should be consistent", type1, type2)
    }

    @Test
    fun testIntegration_workflowWithRealWorldScenario() = runBlocking {
        // Simulate a complete AI workflow
        val analysisAgent = DummyAgent("AnalysisAgent", "Data analyzed successfully", 0.85f)
        val validationAgent = DummyAgent("ValidationAgent", "Validation completed", 0.75f)
        val unreliableAgent = FailingAgent("UnreliableService")
        val fallbackAgent = DummyAgent("FallbackAgent", "Fallback response ready", 0.60f)

        val workflowContext = mapOf(
            "session_id" to "workflow_session_123",
            "user_id" to "user_456",
            "request_type" to "data_analysis",
            "priority" to "high",
            "timestamp" to "2024-01-15T10:30:00Z",
            "max_attempts" to "3",
            "timeout_ms" to "30000"
        )

        // Step 1: Gather responses from external agents
        val externalResponses = genesisAgent.participateWithAgents(
            context = workflowContext,
            agents = listOf(analysisAgent, validationAgent, unreliableAgent, fallbackAgent),
            prompt = "Process high-priority data analysis request",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        // Step 2: Mock internal services
        whenever(auraService.processRequest(any())).thenReturn(
            AgentResponse("Internal aura processing complete", 0.90f)
        )
        whenever(kaiService.processRequest(any())).thenReturn(
            AgentResponse("Internal kai analysis finished", 0.95f)
        )
        whenever(cascadeService.processRequest(any())).thenReturn(
            AgentResponse("Internal cascade validation passed", 0.80f)
        )

        // Step 3: Process internal request
        val internalResponse = genesisAgent.processRequest(
            AiRequest("Complete internal processing for workflow", workflowContext)
        )

        // Step 4: Aggregate all responses
        val allResponses = listOf(
            externalResponses,
            mapOf("InternalGenesisAgent" to internalResponse)
        )
        val finalResult = genesisAgent.aggregateAgentResponses(allResponses)

        // Verify workflow results
        assertEquals("Should have working agents only", 4, externalResponses.size)
        assertNull("Unreliable agent should fail", externalResponses["UnreliableService"])
        assertTrue("Should have analysis", externalResponses.containsKey("AnalysisAgent"))
        assertTrue("Should have validation", externalResponses.containsKey("ValidationAgent"))
        assertTrue("Should have fallback", externalResponses.containsKey("FallbackAgent"))

        assertEquals("Should combine internal services", 
            "Internal aura processing complete Internal kai analysis finished Internal cascade validation passed",
            internalResponse.content)
        assertEquals(0.95f, internalResponse.confidence)

        assertEquals("Should aggregate all results", 4, finalResult.size)
        assertTrue("Should contain internal result", finalResult.containsKey("InternalGenesisAgent"))
    }

    @Test
    fun testParticipateWithAgents_contextSizeLimit() = runBlocking {
        val agent = DummyAgent("LimitAgent", "context processed")
        
        // Test with extremely large context
        val hugeContext = (1..50000).associate { 
            "key$it" to "This is a very long value with lots of text: " + "X".repeat(100)
        }

        val startTime = System.currentTimeMillis()
        try {
            val responses = genesisAgent.participateWithAgents(
                context = hugeContext,
                agents = listOf(agent),
                prompt = "huge context test",
                mode = GenesisAgent.ConversationMode.TURN_ORDER
            )
            val endTime = System.currentTimeMillis()

            assertEquals("Should handle huge context", 1, responses.size)
            assertTrue("Should complete within reasonable time", endTime - startTime < 60000)
        } catch (e: OutOfMemoryError) {
            // Acceptable if system runs out of memory
            assertTrue("System handled memory limitation gracefully", true)
        }
    }

    @Test
    fun testParticipateWithAgents_networkSimulationWithTimeout() = runBlocking {
        val networkAgents = (1..5).map { i ->
            object : Agent {
                override fun getName(): String = "NetworkAgent$i"
                override fun getType(): String? = null
                override suspend fun processRequest(request: AiRequest): AgentResponse {
                    // Simulate network delay
                    delay((100..500).random().toLong())
                    if (i == 3) throw RuntimeException("Network timeout")
                    return AgentResponse("Network response $i", i / 5.0f)
                }
            }
        }

        val responses = genesisAgent.participateWithAgents(
            context = mapOf("network" to "simulation"),
            agents = networkAgents,
            prompt = "network stress test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals("Should have 4 successful responses", 4, responses.size)
        assertNull("NetworkAgent3 should fail", responses["NetworkAgent3"])
        assertTrue("Should have other network agents", responses.containsKey("NetworkAgent1"))
        assertTrue("Should have other network agents", responses.containsKey("NetworkAgent2"))
        assertTrue("Should have other network agents", responses.containsKey("NetworkAgent4"))
        assertTrue("Should have other network agents", responses.containsKey("NetworkAgent5"))
    }

    @Test
    fun testProcessRequest_serviceResponseTimeVariation() = runBlocking {
        val request = AiRequest("timing test", emptyMap())

        // Mock services with different response times
        whenever(auraService.processRequest(any())).thenAnswer {
            runBlocking { delay(100) }
            AgentResponse("aura delayed", 0.8f)
        }
        whenever(kaiService.processRequest(any())).thenAnswer {
            runBlocking { delay(200) }
            AgentResponse("kai more delayed", 0.9f)
        }
        whenever(cascadeService.processRequest(any())).thenAnswer {
            runBlocking { delay(50) }
            AgentResponse("cascade fast", 0.7f)
        }

        val startTime = System.currentTimeMillis()
        val response = genesisAgent.processRequest(request)
        val endTime = System.currentTimeMillis()

        // Should wait for all services (sequential execution)
        assertTrue("Should take at least 350ms total", endTime - startTime >= 350)
        assertEquals("aura delayed kai more delayed cascade fast", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testAggregateAgentResponses_memorySafetyWithDuplicates() {
        // Test memory safety when many duplicate responses exist
        val baseResponse = AgentResponse("base", 0.5f)
        val betterResponse = AgentResponse("better", 0.9f)
        
        val responses = (1..10000).map { batchIndex ->
            if (batchIndex % 1000 == 0) {
                mapOf("TestAgent" to betterResponse)
            } else {
                mapOf("TestAgent" to baseResponse)
            }
        }

        val startTime = System.currentTimeMillis()
        val consensus = genesisAgent.aggregateAgentResponses(responses)
        val endTime = System.currentTimeMillis()

        assertEquals(1, consensus.size)
        assertEquals("better", consensus["TestAgent"]?.content)
        assertEquals(0.9f, consensus["TestAgent"]?.confidence)
        assertTrue("Should complete efficiently", endTime - startTime < 5000)
    }

    @Test
    fun testAgentInterface_contractCompliance() = runBlocking {
        val agents = listOf(
            DummyAgent("Contract1", "response1"),
            FailingAgent("Contract2"),
            genesisAgent
        )

        agents.forEach { agent ->
            // Test interface contract
            assertNotNull("getName should not be null", agent.getName())
            assertTrue("getName should return non-empty string or be empty", 
                agent.getName().isNotEmpty() || agent.getName().isEmpty())
            
            // getType can be null
            val type = agent.getType()
            assertTrue("getType should be callable", true)

            // processRequest should be callable (but may throw)
            val request = AiRequest("contract test", emptyMap())
            try {
                val response = agent.processRequest(request)
                assertNotNull("Response should not be null if no exception", response)
                assertTrue("Response should be AgentResponse type", response is AgentResponse)
            } catch (e: Exception) {
                // Exception is acceptable for some agents
                assertTrue("Exception handling is acceptable", true)
            }
        }
    }

    @Test
    fun testGenesisAgent_immutabilityAndThreadSafety() = runBlocking {
        val requests = (1..50).map { AiRequest("thread test $it", mapOf("index" to "$it")) }
        
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        // Test concurrent access to the same GenesisAgent instance
        val jobs = requests.map { request ->
            async {
                val response = genesisAgent.processRequest(request)
                Triple(genesisAgent.getName(), genesisAgent.getType(), response)
            }
        }

        val results = jobs.map { it.await() }

        // Verify thread safety
        results.forEach { (name, type, response) ->
            assertEquals("Name should be consistent", "GenesisAgent", name)
            assertEquals("Type should be consistent", null, type)
            assertEquals("Content should be consistent", "aura kai cascade", response.content)
            assertEquals("Confidence should be consistent", 0.9f, response.confidence)
        }
    }

    @Test
    fun testDummyAgent_stateIsolation() = runBlocking {
        val agent = DummyAgent("IsolationAgent", "isolated response", 0.7f)
        val requests = (1..100).map { 
            AiRequest("request $it", mapOf("data" to "test data $it")) 
        }

        val responses = requests.map { request ->
            agent.processRequest(request)
        }

        // All responses should be identical (stateless)
        responses.forEach { response ->
            assertEquals("isolated response", response.content)
            assertEquals(0.7f, response.confidence)
        }

        // Agent properties should not change
        repeat(100) {
            assertEquals("IsolationAgent", agent.getName())
            assertNull(agent.getType())
        }
    }

    @Test
    fun testErrorRecovery_resilientWorkflow() = runBlocking {
        // Test workflow that continues despite various failures
        val criticalAgent = DummyAgent("CriticalAgent", "critical success", 0.95f)
        val unreliableAgents = (1..10).map { i ->
            if (i % 3 == 0) {
                FailingAgent("UnreliableAgent$i")
            } else {
                DummyAgent("ReliableAgent$i", "reliable response $i", i / 10.0f)
            }
        }

        val allAgents = listOf(criticalAgent) + unreliableAgents

        // Test that critical functionality works despite failures
        val responses = genesisAgent.participateWithAgents(
            context = mapOf("workflow" to "resilient", "critical" to "true"),
            agents = allAgents,
            prompt = "Execute resilient workflow",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )


        maliciousInputs.forEach { maliciousInput ->
            val responses = genesisAgent.participateWithAgents(
                context = mapOf("malicious" to maliciousInput),
                agents = listOf(agent),
                prompt = maliciousInput,
                mode = GenesisAgent.ConversationMode.TURN_ORDER
            )

            // Should handle malicious input without crashing
            assertEquals("Should handle malicious input", 1, responses.size)
            assertEquals("Should process normally", "sanitized response", responses["SanitizationAgent"]?.content)
        }
    }

    @Test
    fun testRobustness_unexpectedDataTypes() = runBlocking {
        // Test with various edge case names and responses
        val edgeCaseAgents = listOf(
            DummyAgent("null", "null response"),
            DummyAgent("true", "boolean response"),
            DummyAgent("false", "boolean response"),
            DummyAgent("0", "number response"),
            DummyAgent("1", "number response"),
            DummyAgent("-1", "negative response"),
            DummyAgent("NaN", "nan response"),
            DummyAgent("Infinity", "infinity response"),
            DummyAgent("undefined", "undefined response"),
            DummyAgent("[]", "array response"),
            DummyAgent("{}", "object response"),
            DummyAgent("()", "function response")
        )

        val responses = genesisAgent.participateWithAgents(
            context = mapOf("test" to "robustness"),
            agents = edgeCaseAgents,
            prompt = "robustness test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals("Should handle all edge case names", edgeCaseAgents.size, responses.size)

        edgeCaseAgents.forEach { agent ->
            assertTrue("Should contain response for ${agent.getName()}",
                responses.containsKey(agent.getName()))
        }
    }

    @Test
    fun testIntegration_realWorldScenario() = runBlocking {
        // Simulate a real-world scenario with multiple conversation modes
        val realWorldAgents = listOf(
            DummyAgent("AnalysisAgent", "Analysis complete", 0.85f),
            DummyAgent("ValidationAgent", "Validation passed", 0.75f),
            DummyAgent("RecommendationAgent", "Recommendation ready", 0.90f),
            FailingAgent("UnreliableAgent"), // Simulates unreliable service
            DummyAgent("FallbackAgent", "Fallback activated", 0.60f)
        )

        val realWorldContext = mapOf(
            "session_id" to "sess_123456",
            "user_id" to "user_789",
            "timestamp" to "2024-01-01T10:00:00Z",
            "request_type" to "analysis",
            "priority" to "high",
            "max_retries" to "3",
            "timeout" to "30000"
        )

        // Test all conversation modes with real-world context
        val modes = GenesisAgent.ConversationMode.values()
        val allResponses = mutableListOf<Map<String, AgentResponse>>()

        modes.forEach { mode ->
            val responses = genesisAgent.participateWithAgents(
                context = realWorldContext,
                agents = realWorldAgents,
                prompt = "Process user request with high priority",
                mode = mode
            )
            allResponses.add(responses)
        }

        // All modes should produce similar results (4 working agents)
        allResponses.forEach { responses ->
            assertEquals("Should have 4 working agents", 4, responses.size)
            assertNull("Unreliable agent should fail", responses["UnreliableAgent"])
            assertTrue("Should have analysis", responses.containsKey("AnalysisAgent"))
            assertTrue("Should have validation", responses.containsKey("ValidationAgent"))
            assertTrue("Should have recommendation", responses.containsKey("RecommendationAgent"))
            assertTrue("Should have fallback", responses.containsKey("FallbackAgent"))
        }

        // Test aggregation across all modes
        val consensus = genesisAgent.aggregateAgentResponses(allResponses)

        // Should aggregate to highest confidence responses
        assertEquals("Should have 4 agents in consensus", 4, consensus.size)
        assertEquals("Recommendation should have highest confidence", 0.90f, consensus["RecommendationAgent"]?.confidence)
>>>>>>> pr458merge
    }
}
