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
 * @param message The message describing the debug information.
 * @param throwable An optional exception or error to include with the log entry.
 */
    fun debug(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs an informational message about normal application operation.
 *
 * @param tag Identifies the source or category of the log entry.
 * @param message The message describing the informational event.
 * @param throwable An optional exception or error to include with the log entry.
 */
    fun info(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs a warning message to highlight a potential issue or abnormal situation.
 *
 * @param tag The category or source of the warning.
 * @param message The warning message describing the situation.
 * @param throwable An optional exception or error related to the warning.
 */
    fun warn(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs an error-level message indicating a failure or exception within the application.
 *
 * @param tag The category or source of the error event.
 * @param message A description of the error condition.
 * @param throwable An optional exception or error object related to the failure.
 */
    fun error(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs a security-related event to record critical issues or potential threats.
 *
 * @param tag The source or category of the security event.
 * @param message Details about the security incident or concern.
 * @param throwable An optional exception associated with the security event.
 */
    fun security(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Records performance metrics for a specific operation, including its duration and optional contextual metadata.
 *
 * @param tag The source or category associated with the performance event.
 * @param operation The name or description of the operation being measured.
 * @param durationMs The time taken to complete the operation, in milliseconds.
 * @param metadata Optional additional data providing context about the operation.
 */
    fun performance(tag: String, operation: String, durationMs: Long, metadata: Map<String, Any> = emptyMap())
    
    /**
 * Logs a user interaction event for analytics and user experience tracking.
 *
 * @param tag The category or source of the interaction.
 * @param action The specific user action being recorded.
 * @param metadata Additional contextual data about the interaction.
 */
    fun userInteraction(tag: String, action: String, metadata: Map<String, Any> = emptyMap())
    
    /**
 * Records an AI operation event, including its name, confidence score, and optional contextual metadata, for monitoring or auditing AI-driven actions.
 *
 * @param tag Identifier for the source or component generating the log entry.
 * @param operation The name or description of the AI operation performed.
 * @param confidence The confidence score indicating the certainty of the AI operation.
 * @param metadata Optional additional information providing context about the AI operation.
 */
    fun aiOperation(tag: String, operation: String, confidence: Float, metadata: Map<String, Any> = emptyMap())
    
    /**
 * Enables or disables the logger at runtime.
 *
 * When disabled, all logging operations are ignored until re-enabled.
 *
 * @param enabled True to enable logging; false to disable it.
 */
    fun setLoggingEnabled(enabled: Boolean)
    
    /**
 * Sets the minimum severity level for logging.
 *
 * Only log messages at the specified level or higher will be recorded; messages below this threshold are ignored.
 *
 * @param level The lowest log level that will be processed.
 */
    fun setLogLevel(level: LogLevel)
    
    /**
 * Suspends execution to flush all buffered or pending log entries to persistent storage.
 *
 * Ensures that all logs up to the point of invocation are durably written, making them persistent and not lost in case of failure.
 */
    suspend fun flush()
    
    /**
 * Releases resources and terminates all logging operations.
 *
 * Call this method to properly shut down the logger and ensure that all associated resources are freed when logging is no longer required.
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
