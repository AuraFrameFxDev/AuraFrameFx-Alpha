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
     * Saves a string to a file.
     * @param context The application context.
     * @param fileName The name of the file to save to.
     * @param content The content to save.
     * @return True if the operation was successful, false otherwise.
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
     * Reads a string from a file.
     * @param context The application context.
     * @param fileName The name of the file to read from.
     * @return The file content as a string, or null if the file doesn't exist or an error occurred.
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
    
    /**
     * Checks if a file exists in the app's internal storage.
     * @param context The application context.
     * @param fileName The name of the file to check.
     * @return True if the file exists, false otherwise.
     */
    fun fileExists(context: Context, fileName: String): Boolean {
        return File(context.filesDir, fileName).exists()
    }
    
    /**
     * Deletes a file from the app's internal storage.
     * @param context The application context.
     * @param fileName The name of the file to delete.
     * @return True if the file was successfully deleted, false otherwise.
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
     * Gets the absolute path of a file in the app's internal storage.
     * @param context The application context.
     * @param fileName The name of the file.
     * @return The absolute path of the file.
     */
    fun getFilePath(context: Context, fileName: String): String {
        return File(context.filesDir, fileName).absolutePath
    }
}
