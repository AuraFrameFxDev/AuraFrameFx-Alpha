package dev.aurakai.auraframefx.context

import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ContextManager @Inject constructor() {
    private val contexts = mutableListOf<String>()

    /**
     * Adds a context string to the in-memory list if it is non-blank and not already present.
     *
     * Ignores the input if it is blank or already exists in the list.
     *
     * @param context The context string to add.
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
     * Sets one context string as the successor of another in the context chain.
     *
     * Links `contextB` as the successor of `contextA` if both strings are non-blank.
     */
    fun linkContexts(contextA: String, contextB: String) {
        if (contextA.isNotBlank() && contextB.isNotBlank()) {
            contextLinks[contextA] = contextB
        }
    }

    /**
     * Returns the successor context linked to the specified context, or null if no successor exists.
     *
     * @param context The context whose successor is to be retrieved.
     * @return The successor context, or null if none is linked.
     */
    fun getNextInChain(context: String): String? {
        return contextLinks[context]
    }
}
