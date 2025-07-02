package dev.aurakai.auraframefx.context

import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ContextManager @Inject constructor() {
    private val contexts = mutableListOf<String>()

    /**
     * Adds a context string to the in-memory list if it is non-blank and not already present.
     *
     * @param context The context string to add. Ignored if blank or already exists in the list.
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
     * If both `contextA` and `contextB` are non-blank, assigns `contextB` as the successor of `contextA`.
     *
     * @param contextA The context string to be linked from.
     * @param contextB The context string to be set as the successor.
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
     * @return The successor context if one exists, or null otherwise.
     */
    fun getNextInChain(context: String): String? {
        return contextLinks[context]
    }
}
