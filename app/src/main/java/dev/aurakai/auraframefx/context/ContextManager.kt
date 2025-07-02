package dev.aurakai.auraframefx.context

import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ContextManager @Inject constructor() {
    private val contexts = mutableListOf<String>()

    /**
     * Adds a non-blank context string to the list if it does not already exist.
     *
     * @param context The context string to add; ignored if blank or already present.
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
     * Assigns `contextB` as the successor to `contextA` if both strings are non-blank. No action is taken if either string is blank.
     *
     * @param contextA The context string to link from.
     * @param contextB The context string to set as the successor.
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
