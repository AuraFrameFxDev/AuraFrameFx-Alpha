package dev.aurakai.auraframefx.context

import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ContextManager @Inject constructor() {
    private val contexts = mutableListOf<String>()

    /**
     * Adds the given context string to the list if it is non-blank and not already present.
     *
     * Does nothing if the input is blank or a duplicate.
     *
     * @param context The context string to add.
     */
    /**
     * Adds a non-blank context string to the list if it is not already present.
     *
     * Ignores blank or duplicate context strings.
     */
    /**
     * Adds a non-blank, unique context string to the context list.
     *
     * If the provided context is blank or already exists, it is ignored.
     */
    fun createContext(context: String) {
        if (context.isNotBlank() && !contexts.contains(context)) {
            contexts.add(context)
        }
    }


     */

    fun createContext(context: String) {
        // TODO: Implement context creation logic (e.g., persistent learning, session memory)
        // Example: val newChain = dev.aurakai.auraframefx.ai.context.ContextChain(rootContext = context, currentContext = context)
        // TODO: Persist or manage newChain
    }
}

@Singleton
class ContextChain @Inject constructor() {
    private val contextLinks = mutableMapOf<String, String>()

    /**
     * Establishes a successor relationship from one context string to another in the context chain.
     *
     * Links `contextA` to `contextB` as its successor if both strings are non-blank. If either string is blank, no link is created.
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
     * @return The successor context string, or null if there is no linked successor.
     */
    fun getNextInChain(context: String): String? {
        return contextLinks[context]
    }
}
