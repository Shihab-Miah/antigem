package ai.antigem.data.agent.local

import android.content.Context
import ai.antigem.data.agent.AgentChatBackend
import ai.antigem.data.local.LocalAiManager
import ai.antigem.data.local.LocalAiModel
import ai.antigem.domain.model.AgentKind
import ai.antigem.domain.model.ChatMessage
import ai.antigem.domain.model.Conversation
import ai.antigem.domain.model.MessageRole
import ai.antigem.domain.model.ToolCall
import ai.antigem.domain.model.ToolType
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID

/**
 * On-device Local AI Agent Chat Backend.
 * Connects to llama-server (OpenAI-compatible) running locally at http://127.0.0.1:8080/v1
 * with full autonomous tool-calling capabilities (Bash, File Read, File Write).
 */
class LocalChatBackend(private val context: Context) : AgentChatBackend {
    override val kind = AgentKind.LOCAL

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var streamingJob: Job? = null

    private val _conversations = MutableStateFlow<List<Conversation>>(emptyList())
    override val conversations: StateFlow<List<Conversation>> = _conversations.asStateFlow()

    private val _currentConversation = MutableStateFlow<Conversation?>(null)
    override val currentConversation: StateFlow<Conversation?> = _currentConversation.asStateFlow()

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    override val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _isStreaming = MutableStateFlow(false)
    override val isStreaming: StateFlow<Boolean> = _isStreaming.asStateFlow()

    private val _isLoadingConversation = MutableStateFlow(false)
    override val isLoadingConversation: StateFlow<Boolean> = _isLoadingConversation.asStateFlow()

    private val _conversationError = MutableStateFlow<String?>(null)
    override val conversationError: StateFlow<String?> = _conversationError.asStateFlow()

    private val _selectedModel = MutableStateFlow<LocalAiModel?>(null)
    val selectedModel: StateFlow<LocalAiModel?> = _selectedModel.asStateFlow()

    private val apiBase = "http://127.0.0.1:8080/v1"

    init {
        startNewChat()
        LocalAiManager.scanModels()
    }

    fun selectModel(model: LocalAiModel) {
        _selectedModel.value = model
        _currentConversation.value = _currentConversation.value?.copy(modelId = model.name)
    }

    override fun startNewChat() {
        stopStreaming()
        val conv = Conversation(
            id = UUID.randomUUID().toString(),
            title = "Local AI Chat",
            agent = AgentKind.LOCAL,
            modelId = _selectedModel.value?.name ?: "Local Model"
        )
        _conversations.value = listOf(conv) + _conversations.value
        _currentConversation.value = conv
        _messages.value = emptyList()
        _conversationError.value = null
    }

    override fun selectConversation(id: String) {
        val conv = _conversations.value.find { it.id == id } ?: return
        _currentConversation.value = conv
    }

    override fun deleteConversation(id: String) {
        _conversations.value = _conversations.value.filter { it.id != id }
        if (_currentConversation.value?.id == id) {
            _conversations.value.firstOrNull()?.let { selectConversation(it.id) } ?: startNewChat()
        }
    }

    override fun forkConversation(id: String) {
        val conv = _conversations.value.find { it.id == id } ?: return
        val forked = conv.copy(id = UUID.randomUUID().toString(), title = "${conv.title} (Fork)")
        _conversations.value = listOf(forked) + _conversations.value
        _currentConversation.value = forked
    }

    override fun stopStreaming() {
        streamingJob?.cancel()
        streamingJob = null
        _isStreaming.value = false
    }

    override fun retryConnections() {
        LocalAiManager.scanModels()
    }

    override fun onAppForegrounded() {
        LocalAiManager.scanModels()
    }

    override fun sendMessage(content: String) {
        val conv = _currentConversation.value ?: return
        val userMsg = ChatMessage(
            id = UUID.randomUUID().toString(),
            conversationId = conv.id,
            role = MessageRole.USER,
            content = content
        )
        val updatedMessages = _messages.value + userMsg
        _messages.value = updatedMessages

        // Update conversation title from first message
        if (conv.title == "Local AI Chat" && content.isNotBlank()) {
            val title = if (content.length > 28) content.take(28) + "…" else content
            _currentConversation.value = conv.copy(title = title)
        }

        executeAgentTurn(conv.id, updatedMessages)
    }

    private fun executeAgentTurn(convId: String, history: List<ChatMessage>) {
        stopStreaming()
        _isStreaming.value = true
        _conversationError.value = null

        val assistantMsgId = UUID.randomUUID().toString()
        val initialAssistantMsg = ChatMessage(
            id = assistantMsgId,
            conversationId = convId,
            role = MessageRole.ASSISTANT,
            content = "",
            isStreaming = true
        )
        _messages.value = history + initialAssistantMsg

        streamingJob = scope.launch {
            try {
                var currentTurnHistory = history.toMutableList()
                var keepGoing = true
                var steps = 0
                val maxSteps = 6

                while (keepGoing && steps < maxSteps) {
                    steps++
                    val payload = buildOpenAiPayload(currentTurnHistory)
                    val responseJson = postChatCompletion(payload)

                    val choice = responseJson.getJSONArray("choices").getJSONObject(0)
                    val messageObj = choice.getJSONObject("message")
                    val textContent = messageObj.optString("content", "")
                    val toolCallsJson = messageObj.optJSONArray("tool_calls")

                    if (toolCallsJson != null && toolCallsJson.length() > 0) {
                        val executedCalls = mutableListOf<ToolCall>()
                        for (i in 0 until toolCallsJson.length()) {
                            val tc = toolCallsJson.getJSONObject(i)
                            val callId = tc.optString("id", UUID.randomUUID().toString())
                            val fn = tc.getJSONObject("function")
                            val fnName = fn.getString("name")
                            val fnArgs = fn.optString("arguments", "{}")

                            val toolCall = ToolCall(
                                id = callId,
                                name = fnName,
                                toolType = when (fnName) {
                                    "execute_bash" -> ToolType.BASH
                                    "read_file" -> ToolType.VIEW_FILE
                                    "write_file" -> ToolType.EDIT_FILE
                                    else -> ToolType.BASH
                                },
                                command = fnArgs,
                                status = "RUNNING"
                            )
                            executedCalls.add(toolCall)

                            // Execute on device
                            val output = runLocalTool(fnName, fnArgs)
                            val completedCall = toolCall.copy(status = "SUCCESS", output = output)
                            executedCalls[i] = completedCall
                        }

                        val assistantWithTools = ChatMessage(
                            id = UUID.randomUUID().toString(),
                            conversationId = convId,
                            role = MessageRole.ASSISTANT,
                            content = textContent,
                            toolCalls = executedCalls
                        )
                        _messages.value = _messages.value.filter { it.id != assistantMsgId } + assistantWithTools
                        currentTurnHistory.add(assistantWithTools)
                    } else {
                        // Check for inline markdown tool calls like ```json {"name": "execute_bash"...} ```
                        val regex = Regex("```(?:json)?\\s*(\\{.*?\\})\\s*```", RegexOption.DOT_MATCHES_ALL)
                        val match = regex.find(textContent)

                        if (match != null) {
                            try {
                                val inlineJson = JSONObject(match.groupValues[1])
                                if (inlineJson.has("name")) {
                                    val fnName = inlineJson.getString("name")
                                    val fnArgs = inlineJson.optJSONObject("arguments")?.toString()
                                        ?: inlineJson.optString("arguments", "{}")

                                    val executedCall = ToolCall(
                                        id = UUID.randomUUID().toString(),
                                        name = fnName,
                                        toolType = ToolType.BASH,
                                        command = fnArgs,
                                        status = "RUNNING"
                                    )
                                    val output = runLocalTool(fnName, fnArgs)
                                    val completedCall = executedCall.copy(status = "SUCCESS", output = output)

                                    val assistantMsg = ChatMessage(
                                        id = assistantMsgId,
                                        conversationId = convId,
                                        role = MessageRole.ASSISTANT,
                                        content = textContent,
                                        toolCalls = listOf(completedCall)
                                    )
                                    _messages.value = _messages.value.filter { it.id != assistantMsgId } + assistantMsg
                                    currentTurnHistory.add(assistantMsg)
                                    continue
                                }
                            } catch (_: Exception) {}
                        }

                        // Final Assistant response
                        val finalMsg = ChatMessage(
                            id = assistantMsgId,
                            conversationId = convId,
                            role = MessageRole.ASSISTANT,
                            content = textContent,
                            isStreaming = false
                        )
                        _messages.value = _messages.value.filter { it.id != assistantMsgId } + finalMsg
                        keepGoing = false
                    }
                }
            } catch (e: Exception) {
                _conversationError.value = "Local AI Error: ${e.message}"
                val errorMsg = ChatMessage(
                    id = assistantMsgId,
                    conversationId = convId,
                    role = MessageRole.ASSISTANT,
                    content = "Local AI Error: ${e.localizedMessage ?: "Failed to connect to local server"}",
                    isStreaming = false
                )
                _messages.value = _messages.value.filter { it.id != assistantMsgId } + errorMsg
            } finally {
                _isStreaming.value = false
            }
        }
    }

    private fun buildOpenAiPayload(history: List<ChatMessage>): JSONObject {
        val messagesArray = JSONArray()
        val sysObj = JSONObject().apply {
            put("role", "system")
            put("content", "You are AntiGem Local Agent, a fast and helpful on-device AI coding assistant. You have full access to execute terminal commands, view files, and write files. Call tools whenever you need to inspect or modify the workspace.")
        }
        messagesArray.put(sysObj)

        for (m in history) {
            val roleStr = when (m.role) {
                MessageRole.USER -> "user"
                MessageRole.ASSISTANT -> "assistant"
                else -> "user"
            }
            val mObj = JSONObject().apply {
                put("role", roleStr)
                put("content", m.content)
            }
            messagesArray.put(mObj)
        }

        val toolsArray = JSONArray().apply {
            put(JSONObject().apply {
                put("type", "function")
                put("function", JSONObject().apply {
                    put("name", "execute_bash")
                    put("description", "Execute a bash shell command in Termux/Android and return output.")
                    put("parameters", JSONObject().apply {
                        put("type", "object")
                        put("properties", JSONObject().apply {
                            put("command", JSONObject().apply {
                                put("type", "string")
                                put("description", "The shell command to execute")
                            })
                        })
                        put("required", JSONArray().apply { put("command") })
                    })
                })
            })
            put(JSONObject().apply {
                put("type", "function")
                put("function", JSONObject().apply {
                    put("name", "read_file")
                    put("description", "Read contents of a file on device.")
                    put("parameters", JSONObject().apply {
                        put("type", "object")
                        put("properties", JSONObject().apply {
                            put("path", JSONObject().apply {
                                put("type", "string")
                                put("description", "Path to file")
                            })
                        })
                        put("required", JSONArray().apply { put("path") })
                    })
                })
            })
        }

        return JSONObject().apply {
            put("model", "local-model")
            put("messages", messagesArray)
            put("tools", toolsArray)
            put("tool_choice", "auto")
            put("temperature", 0.2)
            put("max_tokens", 1024)
        }
    }

    private fun postChatCompletion(payload: JSONObject): JSONObject {
        val url = URL("$apiBase/chat/completions")
        val conn = url.openConnection() as HttpURLConnection
        conn.requestMethod = "POST"
        conn.setRequestProperty("Content-Type", "application/json")
        conn.doOutput = true
        conn.connectTimeout = 60000
        conn.readTimeout = 60000

        OutputStreamWriter(conn.outputStream).use { it.write(payload.toString()) }

        val responseCode = conn.responseCode
        val stream = if (responseCode in 200..299) conn.inputStream else conn.errorStream
        val responseText = BufferedReader(InputStreamReader(stream)).use { it.readText() }

        if (responseCode !in 200..299) {
            throw RuntimeException("HTTP $responseCode: $responseText")
        }
        return JSONObject(responseText)
    }

    private fun runLocalTool(name: String, argsJson: String): String {
        return try {
            val json = JSONObject(argsJson)
            when (name) {
                "execute_bash" -> {
                    val cmd = json.optString("command", "")
                    val process = Runtime.getRuntime().exec(arrayOf("bash", "-c", cmd))
                    val out = BufferedReader(InputStreamReader(process.inputStream)).use { it.readText() }
                    val err = BufferedReader(InputStreamReader(process.errorStream)).use { it.readText() }
                    process.waitFor()
                    val combined = (out + "\n" + err).trim()
                    if (combined.isEmpty()) "(command completed with no output)" else combined
                }
                "read_file" -> {
                    val path = json.optString("path", "")
                    val file = File(path)
                    if (!file.exists()) "Error: File does not exist: $path"
                    else file.readText().take(4000)
                }
                "write_file" -> {
                    val path = json.optString("path", "")
                    val content = json.optString("content", "")
                    val file = File(path)
                    file.parentFile?.mkdirs()
                    file.writeText(content)
                    "Successfully wrote ${content.length} characters to $path"
                }
                else -> "Unknown tool: $name"
            }
        } catch (e: Exception) {
            "Error executing tool $name: ${e.message}"
        }
    }
}
