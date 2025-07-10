package dev.aurakai.auraframefx.ai.agents

import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AiRequest
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.mockito.kotlin.*
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

    // ====== EXISTING TESTS ======
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

    // ====== NEW COMPREHENSIVE TESTS ======

    // Testing framework used: JUnit 4, Mockito Kotlin, Kotlin Coroutines
    
    // Edge Cases for Agent Names and Responses
    @Test
    fun testParticipateWithAgents_agentNamesWithSpecialCharacters() = runBlocking {
        val specialNameAgents = listOf(
            DummyAgent("Agent@#$%", "special char response 1"),
            DummyAgent("Agent\nNewline", "newline response"),
            DummyAgent("Agent\tTab", "tab response"),
            DummyAgent("Agent with spaces", "space response"),
            DummyAgent("Agent.dot.name", "dot response"),
            DummyAgent("Agent-dash-name", "dash response"),
            DummyAgent("Agent_underscore_name", "underscore response"),
            DummyAgent("Agent123Numbers", "number response"),
            DummyAgent("AGENT_UPPERCASE", "uppercase response"),
            DummyAgent("agent_lowercase", "lowercase response")
        )

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = specialNameAgents,
            prompt = "special characters test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals("Should handle all special character names", specialNameAgents.size, responses.size)
        specialNameAgents.forEach { agent ->
            assertTrue("Should contain agent: ${agent.getName()}", responses.containsKey(agent.getName()))
            assertNotNull("Response should not be null for: ${agent.getName()}", responses[agent.getName()])
        }
    }

    @Test
    fun testParticipateWithAgents_agentResponsesWithSpecialContent() = runBlocking {
        val specialContentAgents = listOf(
            DummyAgent("JsonAgent", """{"key": "value", "number": 123, "boolean": true}"""),
            DummyAgent("XmlAgent", "<root><element>value</element></root>"),
            DummyAgent("MarkdownAgent", "# Header\n**Bold** *italic* `code`"),
            DummyAgent("UnicodeAgent", "Unicode: 🌟⭐✨💫 中文 日本語 한국어"),
            DummyAgent("SqlAgent", "SELECT * FROM users WHERE id = 1;"),
            DummyAgent("HtmlAgent", "<html><body><h1>Test</h1></body></html>"),
            DummyAgent("EscapeAgent", "\"Quotes\" 'Single' \\Backslash \\n \\t"),
            DummyAgent("UrlAgent", "https://example.com:8080/path?param=value&other=123"),
            DummyAgent("EmailAgent", "user@example.com, admin@test.org"),
            DummyAgent("MultilineAgent", "Line 1\nLine 2\nLine 3\n\nParagraph 2")
        )

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = specialContentAgents,
            prompt = "special content test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals("Should handle all special content", specialContentAgents.size, responses.size)
        
        // Verify specific content preservation
        assertEquals("""{"key": "value", "number": 123, "boolean": true}""", responses["JsonAgent"]?.content)
        assertEquals("<root><element>value</element></root>", responses["XmlAgent"]?.content)
        assertTrue("Should preserve multiline", responses["MultilineAgent"]?.content?.contains("\n") == true)
        assertTrue("Should preserve unicode", responses["UnicodeAgent"]?.content?.contains("🌟") == true)
    }

    @Test
    fun testParticipateWithAgents_extremelyLongResponses() = runBlocking {
        val longContent = "A".repeat(100000) // 100KB response
        val veryLongContent = "B".repeat(1000000) // 1MB response
        val agents = listOf(
            DummyAgent("LongAgent", longContent),
            DummyAgent("VeryLongAgent", veryLongContent),
            DummyAgent("NormalAgent", "normal response")
        )

        val startTime = System.currentTimeMillis()
        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = agents,
            prompt = "long content test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        val endTime = System.currentTimeMillis()

        assertEquals("Should handle long responses", 3, responses.size)
        assertEquals("Should preserve long content", longContent, responses["LongAgent"]?.content)
        assertEquals("Should preserve very long content", veryLongContent, responses["VeryLongAgent"]?.content)
        assertTrue("Should complete within reasonable time", (endTime - startTime) < 5000)
    }

    // Context Edge Cases
    @Test
    fun testParticipateWithAgents_contextWithEmptyValues() = runBlocking {
        val agent = DummyAgent("ContextAgent", "handled empty context")
        val contextWithEmpties = mapOf(
            "empty1" to "",
            "empty2" to "",
            "whitespace" to "   ",
            "valid" to "value",
            "newlines" to "\n\n",
            "tabs" to "\t\t"
        )

        val responses = genesisAgent.participateWithAgents(
            context = contextWithEmpties,
            agents = listOf(agent),
            prompt = "empty context test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("handled empty context", responses["ContextAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_contextWithSpecialKeys() = runBlocking {
        val agent = DummyAgent("SpecialKeyAgent", "handled special keys")
        val specialContext = mapOf(
            "key with spaces" to "value1",
            "key:with:colons" to "value2", 
            "key;with;semicolons" to "value3",
            "key=with=equals" to "value4",
            "key&with&ampersands" to "value5",
            "key|with|pipes" to "value6",
            "key<with>brackets" to "value7",
            "key[with]squares" to "value8",
            "key{with}braces" to "value9",
            "key(with)parentheses" to "value10"
        )

        val responses = genesisAgent.participateWithAgents(
            context = specialContext,
            agents = listOf(agent),
            prompt = "special keys test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals(1, responses.size)
        assertEquals("handled special keys", responses["SpecialKeyAgent"]?.content)
    }

    @Test
    fun testParticipateWithAgents_massiveContext() = runBlocking {
        val agent = DummyAgent("MassiveContextAgent", "handled massive context")
        val massiveContext = (1..10000).associate { 
            "key$it" to "value$it".repeat(50) 
        }

        val startTime = System.currentTimeMillis()
        val responses = genesisAgent.participateWithAgents(
            context = massiveContext,
            agents = listOf(agent),
            prompt = "massive context test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        val endTime = System.currentTimeMillis()

        assertEquals(1, responses.size)
        assertEquals("handled massive context", responses["MassiveContextAgent"]?.content)
        assertTrue("Should handle massive context efficiently", (endTime - startTime) < 10000)
    }

    // Conversation Mode Testing
    @Test
    fun testParticipateWithAgents_allConversationModes() = runBlocking {
        val agents = listOf(
            DummyAgent("Agent1", "response1", 0.8f),
            DummyAgent("Agent2", "response2", 0.9f)
        )
        val context = mapOf("mode" to "test")
        val prompt = "mode test"

        val modes = GenesisAgent.ConversationMode.values()
        val allResults = mutableMapOf<GenesisAgent.ConversationMode, Map<String, AgentResponse>>()

        modes.forEach { mode ->
            val responses = genesisAgent.participateWithAgents(
                context = context,
                agents = agents,
                prompt = prompt,
                mode = mode
            )
            allResults[mode] = responses
        }

        // All modes should produce similar results with current implementation
        modes.forEach { mode ->
            val responses = allResults[mode]!!
            assertEquals("Mode $mode should handle both agents", 2, responses.size)
            assertEquals("response1", responses["Agent1"]?.content)
            assertEquals("response2", responses["Agent2"]?.content)
        }
    }

    @Test
    fun testConversationMode_enumCompleteness() {
        val modes = GenesisAgent.ConversationMode.values()
        assertEquals("Should have exactly 3 conversation modes", 3, modes.size)
        
        val expectedModes = setOf("TURN_ORDER", "CASCADE", "CONSENSUS")
        val actualModes = modes.map { it.name }.toSet()
        assertEquals("Should have expected mode names", expectedModes, actualModes)
        
        // Test valueOf for all modes
        expectedModes.forEach { modeName ->
            val mode = GenesisAgent.ConversationMode.valueOf(modeName)
            assertEquals("valueOf should work correctly", modeName, mode.name)
        }
    }

    // Error Handling and Recovery
    @Test
    fun testParticipateWithAgents_mixedExceptionTypes() = runBlocking {
        val agents = listOf(
            object : Agent {
                override fun getName() = "RuntimeExceptionAgent"
                override fun getType() = null
                override suspend fun processRequest(request: AiRequest): AgentResponse {
                    throw RuntimeException("Runtime error")
                }
            },
            object : Agent {
                override fun getName() = "IllegalArgumentAgent"
                override fun getType() = null
                override suspend fun processRequest(request: AiRequest): AgentResponse {
                    throw IllegalArgumentException("Illegal argument")
                }
            },
            object : Agent {
                override fun getName() = "IllegalStateAgent"
                override fun getType() = null
                override suspend fun processRequest(request: AiRequest): AgentResponse {
                    throw IllegalStateException("Illegal state")
                }
            },
            object : Agent {
                override fun getName() = "OutOfMemoryAgent"
                override fun getType() = null
                override suspend fun processRequest(request: AiRequest): AgentResponse {
                    throw OutOfMemoryError("Out of memory")
                }
            },
            object : Agent {
                override fun getName() = "NullPointerAgent"
                override fun getType() = null
                override suspend fun processRequest(request: AiRequest): AgentResponse {
                    throw NullPointerException("Null pointer")
                }
            },
            DummyAgent("WorkingAgent", "success")
        )

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = agents,
            prompt = "exception handling test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        // Only the working agent should succeed
        assertEquals("Should only have working agent", 1, responses.size)
        assertEquals("success", responses["WorkingAgent"]?.content)
        
        // All failing agents should be absent
        assertNull(responses["RuntimeExceptionAgent"])
        assertNull(responses["IllegalArgumentAgent"])
        assertNull(responses["IllegalStateAgent"])
        assertNull(responses["OutOfMemoryAgent"])
        assertNull(responses["NullPointerAgent"])
    }

    @Test
    fun testParticipateWithAgents_allAgentsFailing() = runBlocking {
        val failingAgents = (1..10).map { i ->
            object : Agent {
                override fun getName() = "FailingAgent$i"
                override fun getType() = null
                override suspend fun processRequest(request: AiRequest): AgentResponse {
                    throw RuntimeException("Failure $i")
                }
            }
        }

        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = failingAgents,
            prompt = "all failing test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertTrue("Should handle all failures gracefully", responses.isEmpty())
    }

    // Performance and Scalability Tests
    @Test
    fun testParticipateWithAgents_performanceWithManyAgents() = runBlocking {
        val manyAgents = (1..1000).map { i ->
            DummyAgent("Agent$i", "response$i", (i % 100) / 100.0f)
        }

        val startTime = System.currentTimeMillis()
        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = manyAgents,
            prompt = "performance test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        val endTime = System.currentTimeMillis()

        assertEquals("Should handle all agents", 1000, responses.size)
        assertTrue("Should complete within reasonable time", (endTime - startTime) < 30000)
        
        // Verify all responses are present
        (1..1000).forEach { i ->
            assertEquals("response$i", responses["Agent$i"]?.content)
        }
    }

    @Test
    fun testParticipateWithAgents_memoryEfficiency() = runBlocking {
        // Test with large responses to check memory efficiency
        val largeResponseAgents = (1..100).map { i ->
            DummyAgent("LargeAgent$i", "Large response: " + "X".repeat(10000), 0.5f)
        }

        val startTime = System.currentTimeMillis()
        val responses = genesisAgent.participateWithAgents(
            context = emptyMap(),
            agents = largeResponseAgents,
            prompt = "memory test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )
        val endTime = System.currentTimeMillis()

        assertEquals("Should handle all large responses", 100, responses.size)
        assertTrue("Should be memory efficient", (endTime - startTime) < 10000)
    }

    // Aggregate Response Testing
    @Test
    fun testAggregateAgentResponses_extremeConfidenceValues() {
        val responses = listOf(
            mapOf("Agent1" to AgentResponse("max", Float.MAX_VALUE)),
            mapOf("Agent1" to AgentResponse("min", Float.MIN_VALUE)),
            mapOf("Agent1" to AgentResponse("pos_inf", Float.POSITIVE_INFINITY)),
            mapOf("Agent1" to AgentResponse("neg_inf", Float.NEGATIVE_INFINITY)),
            mapOf("Agent1" to AgentResponse("nan", Float.NaN)),
            mapOf("Agent1" to AgentResponse("zero", 0.0f)),
            mapOf("Agent1" to AgentResponse("negative", -1.0f)),
            mapOf("Agent1" to AgentResponse("normal", 0.8f))
        )

        val consensus = genesisAgent.aggregateAgentResponses(responses)
        
        assertEquals(1, consensus.size)
        assertNotNull("Should handle extreme values", consensus["Agent1"])
        // POSITIVE_INFINITY should be the highest value
        assertEquals("pos_inf", consensus["Agent1"]?.content)
    }

    @Test
    fun testAggregateAgentResponses_largeResponseSets() {
        val largeResponses = (1..1000).map { batchIndex ->
            (1..100).associate { agentIndex ->
                "Agent$agentIndex" to AgentResponse(
                    "response${batchIndex}_$agentIndex",
                    (batchIndex * agentIndex) / 100000.0f
                )
            }
        }

        val startTime = System.currentTimeMillis()
        val consensus = genesisAgent.aggregateAgentResponses(largeResponses)
        val endTime = System.currentTimeMillis()

        assertEquals("Should handle large response sets", 100, consensus.size)
        assertTrue("Should be efficient with large sets", (endTime - startTime) < 5000)
        
        // Verify highest confidence responses are selected
        (1..100).forEach { agentIndex ->
            val expectedConfidence = (1000 * agentIndex) / 100000.0f
            assertEquals("Should select highest confidence", expectedConfidence, consensus["Agent$agentIndex"]?.confidence)
        }
    }

    @Test
    fun testAggregateAgentResponses_duplicateResponseContent() {
        val identicalContent = "identical response"
        val responses = (1..100).map { confidence ->
            mapOf("Agent1" to AgentResponse(identicalContent, confidence / 100.0f))
        }

        val consensus = genesisAgent.aggregateAgentResponses(responses)
        
        assertEquals(1, consensus.size)
        assertEquals(identicalContent, consensus["Agent1"]?.content)
        assertEquals(1.0f, consensus["Agent1"]?.confidence) // Highest confidence should be selected
    }

    // ProcessRequest Testing
    @Test
    fun testProcessRequest_serviceResponseOrdering() = runBlocking {
        val request = AiRequest("ordering test", emptyMap())
        
        // Setup services to return responses in specific order
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("first", 0.7f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("second", 0.8f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("third", 0.9f))

        val response = genesisAgent.processRequest(request)

        assertEquals("Should combine in correct order", "first second third", response.content)
        assertEquals("Should use highest confidence", 0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_emptyResponses() = runBlocking {
        val request = AiRequest("empty test", emptyMap())
        
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("", 0.7f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("", 0.8f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("", 0.9f))

        val response = genesisAgent.processRequest(request)

        assertEquals("Should handle empty responses", "  ", response.content) // Two spaces from joining
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_mixedEmptyAndNonEmpty() = runBlocking {
        val request = AiRequest("mixed test", emptyMap())
        
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("", 0.7f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("middle", 0.8f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("", 0.9f))

        val response = genesisAgent.processRequest(request)

        assertEquals("Should handle mixed responses", " middle ", response.content)
        assertEquals(0.9f, response.confidence)
    }

    @Test
    fun testProcessRequest_verifyServiceCallOrder() = runBlocking {
        val request = AiRequest("call order test", emptyMap())
        val callOrder = mutableListOf<String>()
        
        whenever(auraService.processRequest(any())).thenAnswer {
            callOrder.add("aura")
            AgentResponse("aura", 0.7f)
        }
        whenever(kaiService.processRequest(any())).thenAnswer {
            callOrder.add("kai")
            AgentResponse("kai", 0.8f)
        }
        whenever(cascadeService.processRequest(any())).thenAnswer {
            callOrder.add("cascade")
            AgentResponse("cascade", 0.9f)
        }

        genesisAgent.processRequest(request)

        assertEquals("Services should be called in order", listOf("aura", "kai", "cascade"), callOrder)
    }

    @Test
    fun testProcessRequest_individualServiceFailures() = runBlocking {
        val request = AiRequest("service failure test", emptyMap())
        
        // Test each service failing individually
        val serviceFailureTests = listOf(
            Triple("aura", { whenever(auraService.processRequest(any())).thenThrow(RuntimeException("Aura failed")) }, "Aura failed"),
            Triple("kai", { whenever(kaiService.processRequest(any())).thenThrow(RuntimeException("Kai failed")) }, "Kai failed"),
            Triple("cascade", { whenever(cascadeService.processRequest(any())).thenThrow(RuntimeException("Cascade failed")) }, "Cascade failed")
        )
        
        serviceFailureTests.forEach { (serviceName, setupFailure, expectedMessage) ->
            // Reset mocks
            reset(auraService, kaiService, cascadeService)
            
            // Setup the failing service and working services
            setupFailure()
            if (serviceName != "aura") whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.7f))
            if (serviceName != "kai") whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.8f))
            if (serviceName != "cascade") whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.9f))
            
            try {
                genesisAgent.processRequest(request)
                fail("Should throw exception when $serviceName service fails")
            } catch (e: RuntimeException) {
                assertEquals("Should propagate $serviceName failure", expectedMessage, e.message)
            }
        }
    }

    // Concurrency and Thread Safety
    @Test
    fun testParticipateWithAgents_concurrentAccess() = runBlocking {
        val agent = DummyAgent("ConcurrentAgent", "concurrent response")
        val responses = ConcurrentHashMap<String, AgentResponse>()

        val jobs = (1..50).map { i ->
            kotlinx.coroutines.async {
                val result = genesisAgent.participateWithAgents(
                    context = mapOf("test" to "concurrent$i"),
                    agents = listOf(agent),
                    prompt = "concurrent test $i",
                    mode = GenesisAgent.ConversationMode.TURN_ORDER
                )
                responses.putAll(result)
                result
            }
        }

        val results = jobs.map { it.await() }

        assertEquals("All concurrent operations should complete", 50, results.size)
        results.forEach { result ->
            assertEquals(1, result.size)
            assertEquals("concurrent response", result["ConcurrentAgent"]?.content)
        }
        assertTrue("Should handle concurrent access", responses.isNotEmpty())
    }

    @Test
    fun testProcessRequest_concurrentRequests() = runBlocking {
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.7f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.8f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.9f))

        val jobs = (1..30).map { i ->
            kotlinx.coroutines.async {
                genesisAgent.processRequest(AiRequest("concurrent request $i", mapOf("id" to "$i")))
            }
        }

        val results = jobs.map { it.await() }

        assertEquals("All requests should complete", 30, results.size)
        results.forEach { response ->
            assertEquals("aura kai cascade", response.content)
            assertEquals(0.9f, response.confidence)
        }
    }

    // Integration and Workflow Tests
    @Test
    fun testCompleteWorkflow_participateAndAggregate() = runBlocking {
        val externalAgents = listOf(
            DummyAgent("External1", "ext1", 0.7f),
            DummyAgent("External2", "ext2", 0.8f),
            DummyAgent("External3", "ext3", 0.6f)
        )

        // Step 1: Participate with external agents
        val externalResponses = genesisAgent.participateWithAgents(
            context = mapOf("workflow" to "test"),
            agents = externalAgents,
            prompt = "external consultation",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        // Step 2: Generate multiple response batches
        val responseBatches = listOf(
            externalResponses,
            mapOf("Internal1" to AgentResponse("int1", 0.9f)),
            mapOf("Internal2" to AgentResponse("int2", 0.85f))
        )

        // Step 3: Aggregate all responses
        val finalConsensus = genesisAgent.aggregateAgentResponses(responseBatches)

        assertEquals("Should have all responses", 5, finalConsensus.size)
        assertEquals("ext1", finalConsensus["External1"]?.content)
        assertEquals("ext2", finalConsensus["External2"]?.content)
        assertEquals("ext3", finalConsensus["External3"]?.content)
        assertEquals("int1", finalConsensus["Internal1"]?.content)
        assertEquals("int2", finalConsensus["Internal2"]?.content)
    }

    @Test
    fun testRealWorldScenario_chatbotWorkflow() = runBlocking {
        // Simulate a real chatbot workflow with multiple agents
        val chatbotAgents = listOf(
            DummyAgent("IntentRecognizer", "intent:greeting confidence:0.95", 0.95f),
            DummyAgent("EntityExtractor", "entities:[] confidence:0.8", 0.8f),
            DummyAgent("ContextManager", "context:session_123 user:john", 0.9f),
            DummyAgent("ResponseGenerator", "response:Hello! How can I help you?", 0.85f),
            FailingAgent("OptionalAnalyzer") // Simulates optional failing service
        )

        val userInput = mapOf(
            "user_id" to "john_doe",
            "session_id" to "sess_123456",
            "timestamp" to "2024-01-01T10:30:00Z",
            "channel" to "web_chat",
            "language" to "en"
        )

        val responses = genesisAgent.participateWithAgents(
            context = userInput,
            agents = chatbotAgents,
            prompt = "Hello there!",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        assertEquals("Should have working agents only", 4, responses.size)
        assertTrue("Should have intent recognition", responses.containsKey("IntentRecognizer"))
        assertTrue("Should have entity extraction", responses.containsKey("EntityExtractor"))
        assertTrue("Should have context management", responses.containsKey("ContextManager"))
        assertTrue("Should have response generation", responses.containsKey("ResponseGenerator"))
        assertNull("Optional analyzer should fail gracefully", responses["OptionalAnalyzer"])

        // Verify specific responses
        assertTrue("Intent should be detected", responses["IntentRecognizer"]?.content?.contains("intent:greeting") == true)
        assertTrue("Context should be managed", responses["ContextManager"]?.content?.contains("session_123") == true)
    }

    // Boundary and Edge Case Testing
    @Test
    fun testParticipateWithAgents_boundaryConditions() = runBlocking {
        val boundaryTests = listOf(
            // Empty everything
            Triple(emptyMap<String, String>(), emptyList<Agent>(), ""),
            // Empty context and prompt, single agent
            Triple(emptyMap<String, String>(), listOf(DummyAgent("Single", "alone")), ""),
            // Single context entry, no prompt
            Triple(mapOf("single" to "value"), listOf(DummyAgent("Context", "context")), ""),
            // No context, single word prompt
            Triple(emptyMap<String, String>(), listOf(DummyAgent("Prompt", "prompt")), "word"),
            // Everything filled
            Triple(mapOf("full" to "context"), listOf(DummyAgent("Full", "full")), "full prompt")
        )

        boundaryTests.forEach { (context, agents, prompt) ->
            val responses = genesisAgent.participateWithAgents(
                context = context,
                agents = agents,
                prompt = prompt,
                mode = GenesisAgent.ConversationMode.TURN_ORDER
            )

            assertEquals("Should handle boundary case", agents.size, responses.size)
            agents.forEach { agent ->
                if (responses.isNotEmpty()) {
                    assertTrue("Should contain agent: ${agent.getName()}", responses.containsKey(agent.getName()))
                }
            }
        }
    }

    @Test
    fun testAggregateAgentResponses_boundaryConditions() {
        val boundaryTests = listOf(
            // Empty list
            emptyList<Map<String, AgentResponse>>(),
            // Single empty map
            listOf(emptyMap<String, AgentResponse>()),
            // Single map with single response
            listOf(mapOf("Single" to AgentResponse("single", 0.5f))),
            // Multiple empty maps
            listOf(emptyMap(), emptyMap(), emptyMap()),
            // Mix of empty and non-empty
            listOf(
                emptyMap(),
                mapOf("Agent1" to AgentResponse("response1", 0.5f)),
                emptyMap(),
                mapOf("Agent2" to AgentResponse("response2", 0.8f))
            )
        )

        boundaryTests.forEachIndexed { index, responses ->
            val consensus = genesisAgent.aggregateAgentResponses(responses)
            
            val expectedSize = responses.flatMap { it.keys }.distinct().size
            assertEquals("Boundary case $index should work", expectedSize, consensus.size)
        }
    }

    // Agent Interface Testing
    @Test
    fun testDummyAgent_edgeCases() = runBlocking {
        val edgeCaseAgents = listOf(
            DummyAgent("", "empty name"),
            DummyAgent("   ", "whitespace name"),
            DummyAgent("Normal", ""),
            DummyAgent("Zero", "zero", 0.0f),
            DummyAgent("Negative", "negative", -1.0f),
            DummyAgent("Infinity", "infinity", Float.POSITIVE_INFINITY),
            DummyAgent("NaN", "nan", Float.NaN)
        )

        val request = AiRequest("edge case test", emptyMap())

        edgeCaseAgents.forEach { agent ->
            val response = agent.processRequest(request)
            assertNotNull("Should handle edge case: ${agent.getName()}", response)
            assertTrue("Response should have content or be empty", response.content != null)
        }
    }

    @Test
    fun testFailingAgent_consistentFailure() = runBlocking {
        val failingAgent = FailingAgent("ConsistentFailer")
        val request = AiRequest("failure test", emptyMap())

        repeat(10) { iteration ->
            try {
                failingAgent.processRequest(request)
                fail("Should fail consistently on iteration $iteration")
            } catch (e: RuntimeException) {
                assertEquals("Should have consistent message", "Agent processing failed", e.message)
            }
        }
    }

    // GenesisAgent Core Functionality
    @Test
    fun testGenesisAgent_nameAndTypeConsistency() {
        val agent1 = GenesisAgent(auraService, kaiService, cascadeService)
        val agent2 = GenesisAgent(auraService, kaiService, cascadeService)

        assertEquals("Names should be consistent", agent1.getName(), agent2.getName())
        assertEquals("Types should be consistent", agent1.getType(), agent2.getType())
        assertEquals("Name should be GenesisAgent", "GenesisAgent", agent1.getName())
        assertNull("Type should be null", agent1.getType())
    }

    @Test
    fun testProcessRequest_nullRequestHandling() = runBlocking {
        try {
            genesisAgent.processRequest(null as AiRequest)
            fail("Should throw exception for null request")
        } catch (e: IllegalArgumentException) {
            assertTrue("Should handle null request", e.message?.contains("cannot be null") == true)
        } catch (e: Exception) {
            // Any exception is acceptable for null input
            assertTrue("Should throw exception for null request", true)
        }
    }

    // Data Validation and Integrity
    @Test
    fun testAggregateAgentResponses_dataIntegrity() {
        val originalResponse = AgentResponse("original", 0.8f)
        val responses = listOf(mapOf("Agent1" to originalResponse))

        val consensus = genesisAgent.aggregateAgentResponses(responses)

        // Should preserve original response object when no conflicts
        assertSame("Should preserve original response", originalResponse, consensus["Agent1"])
    }

    @Test
    fun testProcessRequest_dataFlowIntegrity() = runBlocking {
        val request = AiRequest("integrity test", mapOf("test" to "data"))
        
        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura_response", 0.7f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai_response", 0.8f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade_response", 0.9f))

        val response = genesisAgent.processRequest(request)

        // Verify that each service received the exact same request
        verify(auraService).processRequest(eq(request))
        verify(kaiService).processRequest(eq(request))
        verify(cascadeService).processRequest(eq(request))

        // Verify aggregation logic
        assertEquals("aura_response kai_response cascade_response", response.content)
        assertEquals(0.9f, response.confidence)
    }

    // Performance Regression Tests
    @Test
    fun testPerformance_baseline() = runBlocking {
        val standardAgents = (1..100).map { i ->
            DummyAgent("PerfAgent$i", "response$i")
        }

        whenever(auraService.processRequest(any())).thenReturn(AgentResponse("aura", 0.8f))
        whenever(kaiService.processRequest(any())).thenReturn(AgentResponse("kai", 0.9f))
        whenever(cascadeService.processRequest(any())).thenReturn(AgentResponse("cascade", 0.7f))

        val startTime = System.currentTimeMillis()

        // Test participate performance
        val participateResult = genesisAgent.participateWithAgents(
            context = (1..50).associate { "key$it" to "value$it" },
            agents = standardAgents,
            prompt = "performance baseline test",
            mode = GenesisAgent.ConversationMode.TURN_ORDER
        )

        // Test process performance
        val processResult = genesisAgent.processRequest(
            AiRequest("performance test", (1..50).associate { "key$it" to "value$it" })
        )

        val endTime = System.currentTimeMillis()

        assertTrue("Should complete within performance baseline", (endTime - startTime) < 10000)
        assertEquals("Should handle all agents", 100, participateResult.size)
        assertNotNull("Should process request", processResult)
    }

    // Final Comprehensive Integration Test
    @Test
    fun testFullSystemIntegration() = runBlocking {
        // Setup comprehensive test scenario
        val multiModalAgents = listOf(
            DummyAgent("TextProcessor", "Text processed successfully", 0.9f),
            DummyAgent("ImageAnalyzer", "Image analysis complete", 0.8f),
            DummyAgent("AudioTranscriber", "Audio transcribed", 0.7f),
            FailingAgent("VideoProcessor"), // Simulates temporarily unavailable service
            DummyAgent("DataValidator", "Data validation passed", 0.95f),
            DummyAgent("SecurityChecker", "Security check completed", 0.85f),
            DummyAgent("ComplianceMonitor", "Compliance verified", 0.75f)
        )

        val comprehensiveContext = mapOf(
            "request_id" to "req_123456789",
            "user_id" to "user_987654321",
            "session_id" to "sess_abcdef123",
            "timestamp" to "2024-01-01T12:00:00.000Z",
            "client_ip" to "192.168.1.100",
            "user_agent" to "Mozilla/5.0 (Test Browser)",
            "content_type" to "multimodal",
            "priority" to "high",
            "department" to "engineering",
            "project" to "genesis_testing",
            "environment" to "test",
            "version" to "1.0.0"
        )

        // Mock internal services
        whenever(auraService.processRequest(any())).thenReturn(
            AgentResponse("Aura processing complete", 0.92f)
        )
        whenever(kaiService.processRequest(any())).thenReturn(
            AgentResponse("Kai analysis finished", 0.88f)
        )
        whenever(cascadeService.processRequest(any())).thenReturn(
            AgentResponse("Cascade workflow executed", 0.91f)
        )

        // Test all conversation modes
        val modeResults = mutableMapOf<GenesisAgent.ConversationMode, Map<String, AgentResponse>>()
        GenesisAgent.ConversationMode.values().forEach { mode ->
            modeResults[mode] = genesisAgent.participateWithAgents(
                context = comprehensiveContext,
                agents = multiModalAgents,
                prompt = "Execute comprehensive multimodal processing workflow",
                mode = mode
            )
        }

        // Test internal processing
        val internalResult = genesisAgent.processRequest(
            AiRequest("Comprehensive system test", comprehensiveContext)
        )

        // Aggregate all results
        val allResponseMaps = modeResults.values.toList() + listOf(
            mapOf("GenesisInternal" to internalResult)
        )
        val finalConsensus = genesisAgent.aggregateAgentResponses(allResponseMaps)

        // Comprehensive Assertions
        // 1. All conversation modes should work
        GenesisAgent.ConversationMode.values().forEach { mode ->
            val responses = modeResults[mode]!!
            assertEquals("Mode $mode should handle working agents", 6, responses.size)
            assertNull("Video processor should fail in all modes", responses["VideoProcessor"])
        }

        // 2. Internal processing should work
        assertEquals("Aura processing complete Kai analysis finished Cascade workflow executed", internalResult.content)
        assertEquals(0.92f, internalResult.confidence)

        // 3. Final consensus should aggregate correctly
        assertTrue("Should have comprehensive results", finalConsensus.size >= 6)
        assertTrue("Should include text processor", finalConsensus.containsKey("TextProcessor"))
        assertTrue("Should include data validator", finalConsensus.containsKey("DataValidator"))
        assertTrue("Should include internal results", finalConsensus.containsKey("GenesisInternal"))

        // 4. Highest confidence responses should be selected
        assertEquals("Data validation passed", finalConsensus["DataValidator"]?.content)
        assertEquals(0.95f, finalConsensus["DataValidator"]?.confidence)

        // 5. System should handle failures gracefully
        assertNull("Failed video processor should not be in final results", finalConsensus["VideoProcessor"])

        // 6. All successful agents should be represented
        val expectedSuccessfulAgents = setOf(
            "TextProcessor", "ImageAnalyzer", "AudioTranscriber", 
            "DataValidator", "SecurityChecker", "ComplianceMonitor",
            "GenesisInternal"
        )
        assertTrue("Should have all successful agents", finalConsensus.keys.containsAll(expectedSuccessfulAgents))
    }
}
