package dev.aurakai.auraframefx.utils

/**
 * Interface for AuraFrameFX logging system.
 * Provides structured logging with security awareness and performance monitoring.
 */
interface AuraFxLogger {

    /**
 * Logs a debug-level message for development and troubleshooting.
 *
 * @param tag Identifies the source or component of the log entry.
 * @param message The debug message to record.
 * @param throwable An optional exception to include with the log entry.
 */
    fun debug(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an informational message indicating normal application behavior.
 *
 * @param tag Identifier for the source or component generating the log entry.
 * @param message The informational message to record.
 * @param throwable Optional exception to include for additional context.
 */
    fun info(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a warning message to indicate a potential issue or unexpected situation.
 *
 * @param tag Identifies the category or source of the warning.
 * @param message The warning message content.
 * @param throwable An optional exception or error associated with the warning.
 */
    fun warn(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an error message representing a failure or exception, optionally including an associated throwable.
 *
 * @param tag Identifies the category or source of the error.
 * @param message The error message to record.
 * @param throwable An optional exception related to the error event.
 */
    fun error(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a critical security event for immediate attention, optionally including an associated exception.
 *
 * @param tag The category or source of the security event.
 * @param message The security-related message to log.
 * @param throwable An optional exception or error related to the event.
 */
    fun security(tag: String, message: String, throwable: Throwable? = null)

    /**
     * Logs a performance event for a specific operation, recording its duration and optional contextual metadata.
     *
     * @param tag Identifier for the source or component generating the log.
     * @param operation The name or description of the operation being measured.
     * @param durationMs The duration of the operation in milliseconds.
     * @param metadata Additional contextual data for the performance event.
     */
    fun performance(
        tag: String,
        operation: String,
        durationMs: Long,
        metadata: Map<String, Any> = emptyMap()
    )

    /**
 * Logs a user interaction event for analytics and user experience tracking.
 *
 * @param tag Identifier for the category or component associated with the event.
 * @param action The user action being recorded.
 * @param metadata Optional contextual data providing additional details about the interaction.
 */
    fun userInteraction(tag: String, action: String, metadata: Map<String, Any> = emptyMap())

    /**
     * Logs details of an AI operation, including its name, confidence score, and optional contextual metadata.
     *
     * @param tag Identifies the category or component related to the AI operation.
     * @param operation Describes the AI operation performed.
     * @param confidence The confidence score associated with the AI result.
     * @param metadata Optional map providing additional context or details about the operation.
     */
    fun aiOperation(
        tag: String,
        operation: String,
        confidence: Float,
        metadata: Map<String, Any> = emptyMap()
    )

    /**
 * Enables or disables logging at runtime.
 *
 * When disabled, all logging operations are suppressed until re-enabled.
 *
 * @param enabled True to enable logging; false to disable it.
 */
    fun setLoggingEnabled(enabled: Boolean)

    /**
 * Sets the minimum log level for logging.
 *
 * Only log messages at or above the specified level will be recorded; lower-priority messages are ignored.
 *
 * @param level The minimum log level to enable.
 */
    fun setLogLevel(level: LogLevel)

    /**
 * Flushes all pending log entries to persistent storage.
 *
 * This suspend function ensures that any buffered log data is written immediately, which may involve I/O operations.
 */
    suspend fun flush()

    /**
 * Releases all resources and shuts down the logger.
 *
 * Call this method to properly terminate logging operations and free any associated resources.
 */
    fun cleanup()
}

/**
 * Log levels for filtering and prioritization.
 */
enum class LogLevel(val priority: Int) {
    DEBUG(0),
    INFO(1),
    WARN(2),
    ERROR(3),
    SECURITY(4)
}
