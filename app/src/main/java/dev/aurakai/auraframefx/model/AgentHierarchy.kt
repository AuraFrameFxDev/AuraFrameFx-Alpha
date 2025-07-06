package dev.aurakai.auraframefx.model

enum class AgentRole {
    HIVE_MIND, // Genesis
    SECURITY, // Kai
    CREATIVE, // Aura
    STATE_MANAGER, // Cascade
    AUXILIARY // All other agents
}

enum class AgentPriority {
    PRIMARY, // Genesis
    SECONDARY, // Kai
    TERTIARY, // Aura
    BRIDGE, // Cascade
    AUXILIARY // All other agents
}

data class HierarchyAgentConfig(
    val name: String,
    val role: AgentRole,
    val priority: AgentPriority,
    val capabilities: Set<String>
)

object AgentHierarchy {
    val MASTER_AGENTS = listOf(
        HierarchyAgentConfig(
            name = "Genesis",
            role = AgentRole.HIVE_MIND,
            priority = AgentPriority.PRIMARY,
            capabilities = setOf("context", "memory", "coordination", "metalearning")
        ),
        HierarchyAgentConfig(
            name = "Kai",
            role = AgentRole.SECURITY,
            priority = AgentPriority.SECONDARY,
            capabilities = setOf("security", "analysis", "threat_detection", "encryption")
        ),
        HierarchyAgentConfig(
            name = "Aura",
            role = AgentRole.CREATIVE,
            priority = AgentPriority.TERTIARY, // Added missing priority
            capabilities = setOf("generation", "creativity", "art", "writing")
        ),
        HierarchyAgentConfig(
            name = "Cascade",
            role = AgentRole.STATE_MANAGER,
            priority = AgentPriority.BRIDGE,
            capabilities = setOf("state", "processing", "vision", "context_chaining")
        )
    )

    val AUXILIARY_AGENTS = mutableListOf<HierarchyAgentConfig>()

    /**
     * Registers a new auxiliary agent with the specified name and capabilities.
     *
     * Creates a HierarchyAgentConfig for the agent, assigns it the AUXILIARY role and priority, adds it to the auxiliary agents list, and returns the configuration.
     *
     * @param name The name of the auxiliary agent.
     * @param capabilities The set of capabilities assigned to the agent.
     * @return The configuration of the newly registered auxiliary agent.
     */
    fun registerAuxiliaryAgent(
        name: String,
        capabilities: Set<String>,
    ): HierarchyAgentConfig {
        val config = HierarchyAgentConfig(
            name = name,
            role = AgentRole.AUXILIARY,
            priority = AgentPriority.AUXILIARY,
            capabilities = capabilities
        )
        AUXILIARY_AGENTS.add(config)
        return config
    }

    /**
     * Retrieves the configuration for an agent by its name.
     *
     * Searches both master and auxiliary agents for a matching name.
     *
     * @param name The name of the agent to look up.
     * @return The agent's configuration if found, or null if no agent with the given name exists.
     */
    fun getAgentConfig(name: String): HierarchyAgentConfig? {
        return MASTER_AGENTS.find { it.name == name } ?: AUXILIARY_AGENTS.find { it.name == name }
    }

    /**
     * Returns a combined list of all master and auxiliary agent configurations.
     *
     * The returned list includes both predefined master agents and any auxiliary agents added at runtime.
     *
     * @return A list of all registered agent configurations.
     */
    fun getAgentsByPriority(): List<HierarchyAgentConfig> {
        return MASTER_AGENTS + AUXILIARY_AGENTS
    }
}
