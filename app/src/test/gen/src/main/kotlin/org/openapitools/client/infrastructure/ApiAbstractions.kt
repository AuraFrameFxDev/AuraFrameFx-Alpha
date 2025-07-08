package org.openapitools.client.infrastructure

<<<<<<< HEAD
typealias MultiValueMap = MutableMap<String, List<String>>

fun collectionDelimiter(collectionFormat: String) = when (collectionFormat) {
=======
typealias MultiValueMap = MutableMap<String,List<String>>

fun collectionDelimiter(collectionFormat: String) = when(collectionFormat) {
>>>>>>> origin/coderabbitai/docstrings/78f34ad
    "csv" -> ","
    "tsv" -> "\t"
    "pipe" -> "|"
    "space" -> " "
    else -> ""
}

val defaultMultiValueConverter: (item: Any?) -> String = { item -> "$item" }

<<<<<<< HEAD
fun <T : Any?> toMultiValue(
    items: Array<T>,
    collectionFormat: String,
    map: (item: T) -> String = defaultMultiValueConverter
) = toMultiValue(items.asIterable(), collectionFormat, map)

fun <T : Any?> toMultiValue(
    items: Iterable<T>,
    collectionFormat: String,
    map: (item: T) -> String = defaultMultiValueConverter
): List<String> {
    return when (collectionFormat) {
        "multi" -> items.map(map)
        else -> listOf(
            items.joinToString(
                separator = collectionDelimiter(collectionFormat),
                transform = map
            )
        )
=======
fun <T : Any?> toMultiValue(items: Array<T>, collectionFormat: String, map: (item: T) -> String = defaultMultiValueConverter)
        = toMultiValue(items.asIterable(), collectionFormat, map)

fun <T : Any?> toMultiValue(items: Iterable<T>, collectionFormat: String, map: (item: T) -> String = defaultMultiValueConverter): List<String> {
    return when(collectionFormat) {
        "multi" -> items.map(map)
        else -> listOf(items.joinToString(separator = collectionDelimiter(collectionFormat), transform = map))
>>>>>>> origin/coderabbitai/docstrings/78f34ad
    }
}
