package dev.aurakai.auraframefx.context

import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ContextManager @Inject constructor() {
    private val contexts = mutableListOf<String>()

    /**
     * Adds a new context string if it is non-blank and not already present in the list.
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
     * Sets a successor relationship between two context strings in the context chain.
     *
     * Links `contextB` as the successor of `contextA` if both strings are non-blank. No action is taken if either string is blank.
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
     * @return The linked successor context, or null if none is found.
     */
    fun getNextInChain(context: String): String? {
        return contextLinks[context]
    }
}
