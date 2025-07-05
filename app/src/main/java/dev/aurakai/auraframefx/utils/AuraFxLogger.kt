package dev.aurakai.auraframefx.utils

/**
 * Interface for AuraFrameFX logging system.
 * Provides structured logging with security awareness and performance monitoring.
 */
interface AuraFxLogger {
    
    /**
 * Logs a debug-level message for diagnostic or development purposes.
 *
 * @param tag Identifies the source or component generating the log entry.
 * @param message The message to record at the debug level.
 * @param throwable An optional exception or error to include with the log entry.
 */
    fun debug(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs an informational message representing standard application activity.
 *
 * @param tag Identifies the source or category of the log entry.
 * @param message The informational message to log.
 * @param throwable Optional exception or error to associate with the log entry.
 */
    fun info(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs a warning message indicating a potential issue or abnormal situation.
 *
 * @param tag The category or source of the warning.
 * @param message The warning message describing the issue.
 * @param throwable An optional exception or error related to the warning.
 */
    fun warn(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs an error message indicating a failure or exception within the application.
 *
 * @param tag The category or source of the error.
 * @param message A description of the error condition.
 * @param throwable An optional exception or error object related to the failure.
 */
    fun error(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs a critical security event or potential threat.
 *
 * @param tag The source or category of the security event.
 * @param message Details about the security incident or concern.
 * @param throwable An optional exception associated with the security event.
 */
    fun security(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Records performance metrics for a specific operation, including its duration and optional metadata.
 *
 * @param tag The source or category associated with the performance event.
 * @param operation The name or description of the operation being measured.
 * @param durationMs The time taken to complete the operation, in milliseconds.
 * @param metadata Optional contextual information related to the operation.
 */
    fun performance(tag: String, operation: String, durationMs: Long, metadata: Map<String, Any> = emptyMap())
    
    /**
 * Logs a user interaction event with optional contextual metadata.
 *
 * @param tag The category or source of the interaction.
 * @param action The specific user action to record.
 * @param metadata Additional contextual data about the interaction, if any.
 */
    fun userInteraction(tag: String, action: String, metadata: Map<String, Any> = emptyMap())
    
    /**
 * Records an AI-related operation, including its name, confidence score, and optional contextual metadata.
 *
 * @param tag Identifies the source or component generating the log entry.
 * @param operation Describes the AI operation being logged.
 * @param confidence Indicates the certainty or confidence level of the AI operation.
 * @param metadata Additional contextual data related to the AI operation.
 */
    fun aiOperation(tag: String, operation: String, confidence: Float, metadata: Map<String, Any> = emptyMap())
    
    /**
 * Enables or disables all logging output at runtime.
 *
 * When disabled, no log messages will be recorded or emitted until re-enabled.
 *
 * @param enabled True to enable logging; false to disable it.
 */
    fun setLoggingEnabled(enabled: Boolean)
    
    /**
 * Sets the minimum log level for recording log entries.
 *
 * Only messages at the specified level or higher will be logged; lower-priority messages are ignored.
 *
 * @param level The minimum log level to enable.
 */
    fun setLogLevel(level: LogLevel)
    
    /**
 * Forces all buffered or pending log entries to be written to persistent storage.
 *
 * Suspends execution until all in-memory or queued logs are durably persisted, ensuring no log data is lost up to the point of invocation.
 */
    suspend fun flush()
    
    /**
 * Releases resources and terminates logging operations for proper shutdown.
 *
 * Call this method when logging is no longer needed to ensure all resources are freed.
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
