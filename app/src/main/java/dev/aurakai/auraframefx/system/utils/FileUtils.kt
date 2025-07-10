package dev.aurakai.auraframefx.system.utils

import android.content.Context
import android.util.Log
import java.io.*

/**
 * Utility class for file operations.
 */
object FileUtils {
    private const val TAG = "FileUtils"
    
    /**
     * Writes the specified string content to a file in the application's internal storage.
     *
     * @param fileName The name of the file to write to.
     * @param content The string content to be saved.
     * @return `true` if the content was successfully written; `false` if an I/O error occurred.
     */
    fun saveToFile(context: Context, fileName: String, content: String): Boolean {
        return try {
            context.openFileOutput(fileName, Context.MODE_PRIVATE).use { output ->
                output.write(content.toByteArray())
                true
            }
        } catch (e: IOException) {
            Log.e(TAG, "Error writing to file $fileName", e)
            false
        }
    }
    
    /**
     * Reads the contents of a file from the app's internal storage as a string.
     *
     * @param fileName The name of the file to read.
     * @return The file content as a string, or null if the file does not exist or an I/O error occurs.
     */
    fun readFromFile(context: Context, fileName: String): String? {
        return try {
            context.openFileInput(fileName).bufferedReader().use { it.readText() }
        } catch (e: FileNotFoundException) {
            Log.d(TAG, "File not found: $fileName")
            null
        } catch (e: IOException) {
            Log.e(TAG, "Error reading from file $fileName", e)
            null
        }
    }
    
    /****
     * Determines whether a file with the specified name exists in the application's internal storage directory.
     *
     * @param fileName The name of the file to check for existence.
     * @return `true` if the file exists, `false` otherwise.
     */
    fun fileExists(context: Context, fileName: String): Boolean {
        return File(context.filesDir, fileName).exists()
    }
    
    /**
     * Deletes the specified file from the app's internal storage.
     *
     * @param fileName The name of the file to delete.
     * @return `true` if the file was successfully deleted, `false` otherwise.
     */
    fun deleteFile(context: Context, fileName: String): Boolean {
        return try {
            context.deleteFile(fileName)
        } catch (e: Exception) {
            Log.e(TAG, "Error deleting file $fileName", e)
            false
        }
    }
    
    /**
     * Returns the absolute file system path for a file in the app's internal storage.
     *
     * @param fileName The name of the file whose path is to be retrieved.
     * @return The absolute path to the specified file within the app's internal storage directory.
     */
    fun getFilePath(context: Context, fileName: String): String {
        return File(context.filesDir, fileName).absolutePath
    }
}
