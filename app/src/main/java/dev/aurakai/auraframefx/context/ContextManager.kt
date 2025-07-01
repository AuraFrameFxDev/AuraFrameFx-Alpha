package dev.aurakai.auraframefx.context

import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ContextManager @Inject constructor() {
    private val contexts = mutableListOf<String>()

    /**
     * Creates and stores a new context.
     * For now, this is a simple in-memory list.
     * TODO: Implement more sophisticated context storage (e.g., persistent learning, session memory)
     *
     * @param context The context string to add.
     */
    fun createContext(context: String) {
        if (context.isNotBlank() && !contexts.contains(context)) {
            contexts.add(context)
        }
    }

    /**
     * Retrieves the current list of contexts.
     *
     * @return A new list containing all current contexts.
     */
    fun getAllContexts(): List<String> {
        return contexts.toList()
    }
}

@Singleton
class ContextChain @Inject constructor() {
    private val contextLinks = mutableMapOf<String, String>()

    /**
     * Links two contexts together, implying a sequential or causal relationship.
     * TODO: Implement more complex context linking logic (e.g., weighted links, graph database)
     *
     * @param contextA The source context.
     * @param contextB The target context to link to.
     */
    fun linkContexts(contextA: String, contextB: String) {
        if (contextA.isNotBlank() && contextB.isNotBlank()) {
            contextLinks[contextA] = contextB
        }
    }

    /**
     * Finds the context that follows the given context.
     *
     * @param context The context to find the link for.
     * @return The linked context, or null if no link exists.
     */
    fun getNextInChain(context: String): String? {
        return contextLinks[context]
    }
}
