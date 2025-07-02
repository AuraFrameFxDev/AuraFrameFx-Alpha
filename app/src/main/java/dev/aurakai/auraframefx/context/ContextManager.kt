package dev.aurakai.auraframefx.context

import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ContextManager @Inject constructor() {
    private val contexts = mutableListOf<String>()

    /**
     * Adds the given context string to the in-memory list if it is non-blank and not already present.
     *
     * @param context The context string to add. Ignored if blank or duplicate.
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
     * Establishes a successor relationship between two context strings.
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
     * Returns the successor context linked to the given context, or null if no link exists.
     *
     * @param context The context whose successor is to be retrieved.
     * @return The successor context, or null if there is no linked successor.
     */
    fun getNextInChain(context: String): String? {
        return contextLinks[context]
    }
}
