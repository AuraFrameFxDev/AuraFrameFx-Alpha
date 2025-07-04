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
     * Links one context string to another as its successor in the context chain.
     *
     * Assigns `contextB` as the successor of `contextA` if both strings are non-blank. Does nothing if either string is blank.
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
     * Retrieves the successor context linked to the specified context.
     *
     * @param context The context for which to find the successor.
     * @return The successor context string if one exists; otherwise, null.
     */
    fun getNextInChain(context: String): String? {
        return contextLinks[context]
    }
}

