package dev.aurakai.auraframefx.context

import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ContextManager @Inject constructor() {
    private val contexts = mutableListOf<String>()

    /**
     * Adds a non-blank, unique context string to the in-memory list of contexts.
     *
     * @param context The context string to add. Ignored if blank or already present.
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
     * Links one context string to another, establishing a successor relationship.
     *
     * If both `contextA` and `contextB` are non-blank, sets `contextB` as the successor of `contextA` in the context chain.
     *
     * @param contextA The context to link from.
     * @param contextB The context to link to as the successor.
     */
    fun linkContexts(contextA: String, contextB: String) {
        if (contextA.isNotBlank() && contextB.isNotBlank()) {
            contextLinks[contextA] = contextB
        }
    }

    /**
     * Retrieves the successor context linked to the specified context.
     *
     * @param context The context for which to find the linked successor.
     * @return The successor context if a link exists, or null otherwise.
     */
    fun getNextInChain(context: String): String? {
        return contextLinks[context]
    }
}
