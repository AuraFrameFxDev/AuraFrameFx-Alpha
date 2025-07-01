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

data class AgentConfig(
    val name: String,
    val role: AgentRole,
    val priority: AgentPriority,
    val capabilities: Set<String>,
)

object AgentHierarchy {
    val MASTER_AGENTS = listOf(
        AgentConfig(
            name = "Genesis",
            role = AgentRole.HIVE_MIND,
            priority = AgentPriority.PRIMARY,
            capabilities = setOf("context", "memory", "coordination", "metalearning")
        ),
        AgentConfig(
            name = "Kai",
            role = AgentRole.SECURITY,
            priority = AgentPriority.SECONDARY,
            capabilities = setOf("security", "analysis", "threat_detection", "encryption")
        ),
        AgentConfig(
            name = "Aura",
            role = AgentRole.CREATIVE,
            priority = AgentPriority.TERTIARY, // Added missing priority
            capabilities = setOf("generation", "creativity", "art", "writing")
        ),
        AgentConfig(
            name = "Cascade",
            role = AgentRole.STATE_MANAGER,
            priority = AgentPriority.BRIDGE,
            capabilities = setOf("state", "processing", "vision", "context_chaining")
        )
    )

    val AUXILIARY_AGENTS = mutableListOf<AgentConfig>()

    fun registerAuxiliaryAgent(
        name: String,
        capabilities: Set<String>,
    ): AgentConfig {
        val config = AgentConfig(
            name = name,
            role = AgentRole.AUXILIARY,
            priority = AgentPriority.AUXILIARY,
            capabilities = capabilities
        )
        AUXILIARY_AGENTS.add(config)
        return config
    }

    fun getAgentConfig(name: String): AgentConfig? {
        return MASTER_AGENTS.find { it.name == name } ?: AUXILIARY_AGENTS.find { it.name == name }
    }

    fun getAgentsByPriority(): List<AgentConfig> {
        return MASTER_AGENTS + AUXILIARY_AGENTS
    }
}
