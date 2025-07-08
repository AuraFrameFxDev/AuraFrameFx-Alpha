package dev.aurakai.auraframefx.utils

/**
 * Interface for AuraFrameFX logging system.
 * Provides structured logging with security awareness and performance monitoring.
 */
interface AuraFxLogger {

    /**
 * Logs a debug-level message for development or troubleshooting purposes.
 *
 * @param tag Identifies the source or component generating the log entry.
 * @param message The debug message to record.
 * @param throwable An optional exception to include with the log entry.
 */
    fun debug(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an informational message representing standard application activity.
 *
 * @param tag Identifier for the source or component generating the log entry.
 * @param message The informational message to record.
 * @param throwable Optional exception to include for additional context.
 */
    fun info(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a warning message indicating a potential issue or abnormal condition.
 *
 * @param tag The category or source of the warning.
 * @param message The warning message content.
 * @param throwable An optional exception or error related to the warning.
 */
    fun warn(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an error message indicating a failure or exception, optionally including an associated exception.
 *
 * @param tag The category or source of the error event.
 * @param message The error message describing the failure or issue.
 * @param throwable An optional exception to include with the error log.
 */
    fun error(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a security-related event that requires immediate attention.
 *
 * @param tag Identifies the source or category of the security event.
 * @param message The message describing the security event.
 * @param throwable An optional exception associated with the event.
 */
    fun security(tag: String, message: String, throwable: Throwable? = null)

    /**
     * Logs a performance event for an operation, including its duration and optional contextual metadata.
     *
     * @param tag Identifier for the source or component generating the log entry.
     * @param operation Name or description of the operation being measured.
     * @param durationMs Duration of the operation in milliseconds.
     * @param metadata Additional context for the performance event.
     */
    fun performance(
        tag: String,
        operation: String,
        durationMs: Long,
        metadata: Map<String, Any> = emptyMap()
    )

    /**
 * Logs a user interaction event for analytics and user experience monitoring.
 *
 * @param tag The category or component associated with the interaction.
 * @param action The specific user action performed.
 * @param metadata Optional contextual details about the interaction.
 */
    fun userInteraction(tag: String, action: String, metadata: Map<String, Any> = emptyMap())

    /**
     * Logs an AI operation event, including the operation name, confidence score, and optional metadata for monitoring or analytics purposes.
     *
     * @param tag The category or component associated with the AI operation.
     * @param operation The name or description of the AI operation performed.
     * @param confidence The confidence score of the AI result.
     * @param metadata Optional additional details or context about the operation.
     */
    fun aiOperation(
        tag: String,
        operation: String,
        confidence: Float,
        metadata: Map<String, Any> = emptyMap()
    )

    /**
 * Enables or disables the logging system at runtime.
 *
 * When disabled, no log entries will be recorded until re-enabled.
 *
 * @param enabled If true, logging is enabled; if false, logging is disabled.
 */
    fun setLoggingEnabled(enabled: Boolean)

    /**
 * Sets the minimum log level for which log entries will be recorded.
 *
 * Log entries below the specified level are ignored.
 *
 * @param level The lowest log level to be processed.
 */
    fun setLogLevel(level: LogLevel)

    /**
 * Suspends execution to write all buffered log entries to persistent storage.
 *
 * Ensures that any pending log data is fully persisted, which may involve I/O operations.
 */
    suspend fun flush()

    /**
 * Shuts down the logging system and releases any associated resources.
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
