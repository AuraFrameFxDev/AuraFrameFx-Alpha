package dev.aurakai.auraframefx.context

import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ContextManager @Inject constructor() {
    private val contexts = mutableListOf<String>()

    /**
     * Adds a non-blank, unique context string to the in-memory list of contexts.
     *
     * @param context The context string to add. Must not be blank and must not already exist in the list.
     */
    fun createContext(context: String) {
        if (context.isNotBlank() && !contexts.contains(context)) {
            contexts.add(context)
        }
    }

    /**
     * Returns a list of all stored context strings.
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
     * Links one context string to another, establishing a sequential relationship.
     *
     * If both context strings are non-blank, sets `contextB` as the successor of `contextA` in the chain.
     *
     * @param contextA The context to link from.
     * @param contextB The context to link to as the next in the chain.
     */
    fun linkContexts(contextA: String, contextB: String) {
        if (contextA.isNotBlank() && contextB.isNotBlank()) {
            contextLinks[contextA] = contextB
        }
    }

    /**
     * Retrieves the successor context linked to the specified context.
     *
     * @param context The context for which to find the successor.
     * @return The linked successor context, or null if no successor exists.
     */
    fun getNextInChain(context: String): String? {
        return contextLinks[context]
    }
}
