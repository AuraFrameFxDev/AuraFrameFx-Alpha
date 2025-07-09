package dev.aurakai.auraframefx.utils

/**
 * Interface for AuraFrameFX logging system.
 * Provides structured logging with security awareness and performance monitoring.
 */
interface AuraFxLogger {

    /**
 * Logs a debug-level message for development and troubleshooting purposes.
 *
 * @param tag Identifies the source or component generating the log entry.
 * @param message The debug message to log.
 * @param throwable Optional exception to include for additional context.
 */
    fun debug(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an informational message representing standard application activity.
 *
 * @param tag Identifies the source or component generating the log entry.
 * @param message The informational message to log.
 * @param throwable Optional exception providing additional context.
 */
    fun info(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a warning message to highlight a potential issue or unexpected situation.
 *
 * @param tag The category or source of the warning.
 * @param message The warning message.
 * @param throwable An optional exception or error related to the warning.
 */
    fun warn(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an error message indicating a failure or exception, with optional exception details.
 *
 * @param tag The category or source of the error.
 * @param message The error message to log.
 * @param throwable An optional exception associated with the error.
 */
    fun error(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a critical security event, optionally including an associated exception.
 *
 * Use this method to record security-related incidents that require immediate attention, such as authentication failures, access violations, or suspicious activities.
 *
 * @param tag Identifies the category or source of the security event.
 * @param message Describes the security event being logged.
 * @param throwable An optional exception or error associated with the event.
 */
    fun security(tag: String, message: String, throwable: Throwable? = null)

    /**
     * Logs a performance event for a specific operation, including its duration and optional metadata.
     *
     * @param tag Identifies the source or component generating the log entry.
     * @param operation The name or description of the operation being measured.
     * @param durationMs The duration of the operation in milliseconds.
     * @param metadata Optional contextual data providing additional details about the performance event.
     */
    fun performance(
        tag: String,
        operation: String,
        durationMs: Long,
        metadata: Map<String, Any> = emptyMap()
    )

    /**
 * Records a user interaction event for analytics and user experience monitoring.
 *
 * @param tag Identifies the category or component related to the interaction.
 * @param action Describes the specific user action performed.
 * @param metadata Additional contextual information about the interaction, if any.
 */
    fun userInteraction(tag: String, action: String, metadata: Map<String, Any> = emptyMap())

    /**
     * Logs an AI operation event with its name, confidence score, and optional contextual metadata.
     *
     * @param tag The category or component associated with the AI operation.
     * @param operation The specific AI operation performed.
     * @param confidence The confidence score of the AI result.
     * @param metadata Additional context or details about the operation.
     */
    fun aiOperation(
        tag: String,
        operation: String,
        confidence: Float,
        metadata: Map<String, Any> = emptyMap()
    )

    /**
 * Enables or disables all logging at runtime.
 *
 * When logging is disabled, no log messages are recorded until logging is re-enabled.
 *
 * @param enabled If true, logging is enabled; if false, all logging is suppressed.
 */
    fun setLoggingEnabled(enabled: Boolean)

    /**
 * Sets the minimum log level for logging.
 *
 * Only messages at or above the specified log level will be recorded; messages below this level are ignored until the threshold is changed.
 *
 * @param level The lowest log level that will be processed.
 */
    fun setLogLevel(level: LogLevel)

    /**
 * Forces all buffered log entries to be written to persistent storage.
 *
 * This suspend function ensures that any pending log data is immediately flushed, which may involve I/O operations.
 */
    suspend fun flush()

    /**
 * Shuts down the logger and releases all associated resources.
 *
 * Call this method to terminate logging operations and ensure proper cleanup before application exit or logger replacement.
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
