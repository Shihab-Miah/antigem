package ai.antigem.data.local

import android.content.Context
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

data class LocalAiModel(
    val name: String,
    val path: String,
    val sizeBytes: Long,
    val sizeFormatted: String,
    val quantization: String
)

sealed interface LocalAiStatus {
    object Stopped : LocalAiStatus
    data class Starting(val modelName: String) : LocalAiStatus
    data class Running(val modelName: String, val endpoint: String, val pid: Int) : LocalAiStatus
    data class Error(val message: String) : LocalAiStatus
}

object LocalAiManager {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val _status = MutableStateFlow<LocalAiStatus>(LocalAiStatus.Stopped)
    val status: StateFlow<LocalAiStatus> = _status.asStateFlow()

    private val _availableModels = MutableStateFlow<List<LocalAiModel>>(emptyList())
    val availableModels: StateFlow<List<LocalAiModel>> = _availableModels.asStateFlow()

    private val defaultSearchPaths = listOf(
        "/storage/emulated/0/Download/Ai Files",
        "/storage/emulated/0/Download",
        "/sdcard/Download/Ai Files",
        "/sdcard/Download"
    )

    fun scanModels() {
        scope.launch {
            val list = mutableListOf<LocalAiModel>()
            val seen = mutableSetOf<String>()

            for (dirPath in defaultSearchPaths) {
                val dir = File(dirPath)
                if (!dir.exists() || !dir.isDirectory) continue

                dir.listFiles { file -> file.isFile && file.name.endsWith(".gguf", ignoreCase = true) }?.forEach { f ->
                    if (seen.add(f.absolutePath)) {
                        list.add(
                            LocalAiModel(
                                name = f.nameWithoutExtension,
                                path = f.absolutePath,
                                sizeBytes = f.length(),
                                sizeFormatted = formatSize(f.length()),
                                quantization = parseQuant(f.name)
                            )
                        )
                    }
                }
            }
            _availableModels.value = list
        }
    }

    private fun formatSize(bytes: Long): String {
        val mb = bytes / (1024.0 * 1024.0)
        return if (mb >= 1024) String.format("%.2f GB", mb / 1024.0) else String.format("%.1f MB", mb)
    }

    private fun parseQuant(name: String): String {
        val upper = name.uppercase()
        val quants = listOf("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q6_K")
        return quants.firstOrNull { upper.contains(it) } ?: ""
    }

    fun checkHealth(endpoint: String = "http://127.0.0.1:8080/v1"): Boolean {
        return try {
            val url = URL("$endpoint/models")
            val conn = url.openConnection() as HttpURLConnection
            conn.connectTimeout = 1500
            conn.readTimeout = 1500
            conn.responseCode == 200
        } catch (_: Exception) {
            false
        }
    }
}
