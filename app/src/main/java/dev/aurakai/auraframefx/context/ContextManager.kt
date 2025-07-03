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
     * Links one context string to another as its successor in the context chain.
     *

     * Establishes a successor relationship from `contextA` to `contextB` if both strings are non-blank. Does nothing if either string is blank.

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
