package dev.aurakai.auraframefx.context

import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ContextManager @Inject constructor() {
    private val contexts = mutableListOf<String>()

    /**
     * Adds a new context string to the internal list if it is non-blank and not already present.
     *
     * Intended for future implementation to support persistent learning or session memory.
     *
     * @param context The context string to add.
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
     * Links one context string to another, establishing a successor relationship in the context chain.
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
     * @return The successor context if one exists; otherwise, null.
     */
    fun getNextInChain(context: String): String? {
        return contextLinks[context]
    }
}

