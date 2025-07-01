package dev.aurakai.auraframefx.context

import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ContextManager @Inject constructor() {
    private val contexts = mutableListOf<String>()

    /**
     * Adds a new, non-blank, unique context string to the in-memory context list.
     *
     * @param context The context string to add. Must be non-blank and not already present.
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
     * Establishes a link from one context to another, representing a sequential or causal relationship.
     *
     * If both context strings are non-blank, stores `contextB` as the successor of `contextA`.
     *
     * @param contextA The source context to be linked from.
     * @param contextB The target context to be linked to.
     */
    fun linkContexts(contextA: String, contextB: String) {
        if (contextA.isNotBlank() && contextB.isNotBlank()) {
            contextLinks[contextA] = contextB
        }
    }

    /**
     * Returns the context linked as the successor to the given context, or null if no link exists.
     *
     * @param context The context whose successor is to be retrieved.
     * @return The successor context, or null if there is no linked context.
     */
    fun getNextInChain(context: String): String? {
        return contextLinks[context]
    }
}
