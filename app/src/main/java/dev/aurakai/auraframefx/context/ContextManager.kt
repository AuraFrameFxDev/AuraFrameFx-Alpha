package dev.aurakai.auraframefx.context

import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ContextManager @Inject constructor() {
    private val contexts = mutableListOf<String>()

    /**
     * Adds a context string to the list if it is non-blank and not already present.
     *
     * @param context The context string to add. Ignored if blank or already exists in the list.
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
     * Sets one context string as the successor of another in the context chain.
     *
     * Assigns `contextB` as the successor of `contextA` if both strings are non-blank. No action is taken if either string is blank.
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
     * Returns the successor context linked to the given context, or null if no successor exists.
     *
     * @param context The context whose successor is to be retrieved.
     * @return The successor context, or null if none is linked.
     */
    fun getNextInChain(context: String): String? {
        return contextLinks[context]
    }
}
