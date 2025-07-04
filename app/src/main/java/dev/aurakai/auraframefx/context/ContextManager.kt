package dev.aurakai.auraframefx.context

import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ContextManager @Inject constructor() {
    private val contexts = mutableListOf<String>()

    /**
     * Registers a new context string if it is non-blank and not already present.
     *
     * The actual logic for adding and managing the context is not yet implemented.
     *
     * @param context The context string to register.
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
     * Assigns `contextB` as the successor to `contextA` if both strings are non-blank. No action is taken if either string is blank.
     *
     * @param contextA The context string to link from.
     * @param contextB The context string to designate as the successor.
     */
    fun linkContexts(contextA: String, contextB: String) {
        if (contextA.isNotBlank() && contextB.isNotBlank()) {
            contextLinks[contextA] = contextB
        }
    }

    /**
     * Returns the successor context string linked to the given context, or null if no successor exists.
     *
     * @param context The context whose successor is to be retrieved.
     * @return The successor context string, or null if none is linked.
     */
    fun getNextInChain(context: String): String? {
        return contextLinks[context]
    }
}

