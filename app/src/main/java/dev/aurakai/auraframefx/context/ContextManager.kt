package dev.aurakai.auraframefx.context

import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ContextManager @Inject constructor() {
    private val contexts = mutableListOf<String>()

    /**
     * Adds a context string to the list if it is non-blank and not already present.
     *
     * @param context The context string to add. Ignored if blank or already exists.
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
     * Links one context string to another as its successor in the context chain.
     *
     * Sets `contextB` as the successor of `contextA` if both strings are non-blank. Does nothing if either string is blank.
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
     * Retrieves the successor context linked to the given context.
     *
     * @param context The context for which to find the successor.
     * @return The successor context if one exists, or null otherwise.
     */
    fun getNextInChain(context: String): String? {
        return contextLinks[context]
    }
}
