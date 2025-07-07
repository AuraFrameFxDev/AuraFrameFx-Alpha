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
     * Registers and returns a new auxiliary agent configuration with the given name and capabilities.
     *
     * The agent is assigned the AUXILIARY role and priority, and is added to the auxiliary agents list.
     *
     * @param name The name to assign to the auxiliary agent.
     * @param capabilities The set of capabilities for the auxiliary agent.
     * @return The created auxiliary agent configuration.
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
     * Returns the configuration for an agent with the specified name, searching both master and auxiliary agents.
     *
     * @param name The name of the agent to retrieve.
     * @return The corresponding agent configuration, or null if no agent with the given name exists.
     */
    fun getAgentConfig(name: String): HierarchyAgentConfig? {
        return MASTER_AGENTS.find { it.name == name } ?: AUXILIARY_AGENTS.find { it.name == name }
    }

    /**
     * Retrieves all registered agent configurations, including both master and auxiliary agents.
     *
     * @return A list containing all master and auxiliary agent configurations.
     */
    fun getAgentsByPriority(): List<HierarchyAgentConfig> {
        return MASTER_AGENTS + AUXILIARY_AGENTS
    }
}
