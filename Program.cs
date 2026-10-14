using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace MyRagWiki
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            Console.WriteLine("=== 🧠 Мой персональный RAG (Ollama + C#) ===\n");

            var ollama = new OllamaClient("http://localhost:11434");

            // Проверяем, что Ollama вообще отвечает
            Console.WriteLine("[0/4] Проверка подключения к Ollama...");
            var isAlive = await ollama.IsAliveAsync();
            if (!isAlive)
            {
                Console.WriteLine("❌ Ollama не отвечает на http://localhost:11434");
                Console.WriteLine("   Запустите: ollama serve");
                Console.ReadKey();
                return;
            }
            Console.WriteLine("✅ Ollama отвечает.\n");

            // Проверяем список моделей
            Console.WriteLine("[0.5/4] Доступные модели:");
            var models = await ollama.ListModelsAsync();
            foreach (var m in models)
                Console.WriteLine($"   - {m}");
            Console.WriteLine();

            // 1. ИНГЕСТЦИЯ
            var chunks = new List<DocumentChunk>
            {
                new DocumentChunk("profile", "Backend-разработчик с опытом коммерческой разработки на .NET более 13 лет. Релевантный опыт в финтехе (Банк ПСБ) и телекоме (Amdocs). Специализируюсь на DDD, микросервисах, оптимизации SQL и Kafka."),
                new DocumentChunk("tern", "TERN (2025-н.в.): PoC микросервиса на .NET (C#) с DDD для Row-Level Security (RLS). Оптимизация SQL-запросов к PostgreSQL для OLAP. Стек: C#, .NET, PostgreSQL, DDD, Docker, C++."),
                new DocumentChunk("amdocs", "Amdocs (2022-2025): Миграция бизнес-логики из хранимых процедур MS SQL Server на уровень приложения через Entity Framework Core для перехода на PostgreSQL. Рефакторинг OCS, код-ревью, e2e тесты для gRPC API и Redis."),
                new DocumentChunk("psb", "Банк ПСБ (2021-2022): WebAPI для мобильного и веб-банка. Логика банковского процессинга. Database First: оптимизация сложных T-SQL скриптов и хранимых процедур MS SQL Server."),
                new DocumentChunk("ibs", "IBS (2019-2020): Настройка CDC-пайплайна: Docker + Kafka + Debezium для стриминга событий из SQL Server в PostgreSQL в реальном времени. Оптимизация SQL под высокой нагрузкой."),
                new DocumentChunk("mont", "Mont (2015-2019): SSO на базе IdentityServer4. Внедрение DDD и Clean Architecture. Стек: C#, .NET Framework, DDD, ElasticSearch (ELK), Redis, MediatR.")
            };

            Console.WriteLine("[1/4] Векторизация базы знаний (nomic-embed-text)...");
            foreach (var chunk in chunks)
            {
                chunk.Embedding = await ollama.GetEmbeddingAsync(chunk.Text, "nomic-embed-text");
                Console.WriteLine($"   [{chunk.Id}] вектор размер: {chunk.Embedding.Length}");
            }
            Console.WriteLine("✅ База знаний проиндексирована.\n");

            // 2. ЗАПРОС
            Console.WriteLine("Введите вопрос о вашем опыте (или нажмите Enter для теста):");
            
            var input = Console.ReadLine();
            var query = string.IsNullOrWhiteSpace(input)
                ? "Расскажи про мой опыт с Kafka и миграцией баз данных"
                : input;
            Console.WriteLine("\n[2/4] Векторизация запроса...");
            var queryEmbedding = await ollama.GetEmbeddingAsync(query, "nomic-embed-text");
            Console.WriteLine($"   Вектор запроса размер: {queryEmbedding.Length}");

            // 3. RETRIEVAL
            Console.WriteLine("\n[3/4] Поиск релевантного контекста...");

            // Фильтруем чанки с валидными эмбеддингами
            var validChunks = chunks.Where(c => c.Embedding.Length > 0).ToList();

            if (validChunks.Count == 0)
            {
                Console.WriteLine("❌ Нет валидных эмбеддингов. Используем первый чанк как fallback.");
                // Fallback: просто берём первый чанк
                var fallback = chunks.First();
                var prompt = $@"Ты — помощник по анализу резюме Senior .NET разработчика.
Используй ТОЛЬКО следующий факт из базы знаний кандидата:

ФАКТ: {fallback.Text}

ВОПРОС: {query}

Дай краткий, структурированный ответ на русском языке. Не выдумывай факты.";

                Console.WriteLine("\n[4/4] Генерация ответа (qwen2.5-coder:14b)...");
                var response = await ollama.GenerateAsync(prompt, "qwen2.5-coder:14b");
                Console.WriteLine($"\n ОТВЕТ ИИ:\n{response}");
                Console.ReadKey();
                return;
            }

            var bestMatch = validChunks
                .Select(c => new { Chunk = c, Similarity = CosineSimilarity(queryEmbedding, c.Embedding) })
                .OrderByDescending(x => x.Similarity)
                .First();

            Console.WriteLine($"✅ Найдено совпадение: {bestMatch.Chunk.Id} (Сходство: {bestMatch.Similarity:P1})");
            Console.WriteLine($"   Текст: {bestMatch.Chunk.Text}\n");

            // 4. GENERATION
            var finalPrompt = $@"Ты — помощник по анализу резюме Senior .NET разработчика.
Используй ТОЛЬКО следующий факт из базы знаний кандидата:

ФАКТ: {bestMatch.Chunk.Text}

ВОПРОС: {query}

Дай краткий, структурированный ответ на русском языке. Не выдумывай факты. Если в факте нет ответа, так и скажи.";

            Console.WriteLine("[4/4] Генерация ответа (qwen2.5-coder:14b)...");
            var finalResponse = await ollama.GenerateAsync(finalPrompt, "qwen2.5-coder:14b");

            Console.WriteLine("\n" + new string('=', 50));
            Console.WriteLine("🤖 ОТВЕТ ИИ:");
            Console.WriteLine(new string('=', 50));
            Console.WriteLine(finalResponse);
            Console.WriteLine(new string('=', 50) + "\n");

            Console.WriteLine("Нажмите любую клавишу для выхода...");
            Console.ReadKey();
        }

        static float CosineSimilarity(float[] a, float[] b)
        {
            if (a.Length == 0 || b.Length == 0 || a.Length != b.Length)
            {
                Console.WriteLine($"   ⚠️ Невалидные векторы: a.Length={a.Length}, b.Length={b.Length}");
                return 0f;
            }

            float dotProduct = 0, magnitudeA = 0, magnitudeB = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dotProduct += a[i] * b[i];
                magnitudeA += a[i] * a[i];
                magnitudeB += b[i] * b[i];
            }

            float denominator = MathF.Sqrt(magnitudeA) * MathF.Sqrt(magnitudeB);
            if (denominator == 0) return 0f;

            return dotProduct / denominator;
        }
    }

    public class DocumentChunk
    {
        public string Id { get; }
        public string Text { get; }
        public float[] Embedding { get; set; } = Array.Empty<float>();

        public DocumentChunk(string id, string text) { Id = id; Text = text; }
    }

    public class OllamaClient
    {
        private readonly HttpClient _http = new();
        private readonly string _url;

        public OllamaClient(string baseUrl) => _url = baseUrl;

        public async Task<bool> IsAliveAsync()
        {
            try
            {
                var resp = await _http.GetAsync($"{_url}/api/tags");
                return resp.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }

        public async Task<List<string>> ListModelsAsync()
        {
            try
            {
                var resp = await _http.GetAsync($"{_url}/api/tags");
                var json = await resp.Content.ReadAsStringAsync();
                var doc = JsonDocument.Parse(json);
                var models = new List<string>();
                if (doc.RootElement.TryGetProperty("models", out var modelsArray))
                {
                    foreach (var m in modelsArray.EnumerateArray())
                    {
                        if (m.TryGetProperty("name", out var name))
                            models.Add(name.GetString() ?? "");
                    }
                }
                return models;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   Ошибка получения списка моделей: {ex.Message}");
                return new List<string>();
            }
        }

        public async Task<float[]> GetEmbeddingAsync(string text, string model)
        {
            try
            {
                // Пробуем новый формат API (/api/embed) — Ollama 0.3+
                var reqNew = new { model, input = new[] { text } };
                var contentNew = new StringContent(JsonSerializer.Serialize(reqNew), Encoding.UTF8, "application/json");
                var respNew = await _http.PostAsync($"{_url}/api/embed", contentNew);

                if (respNew.IsSuccessStatusCode)
                {
                    var json = await respNew.Content.ReadAsStringAsync();
                    Console.WriteLine($"   [DEBUG /api/embed] ответ: {json.Substring(0, Math.Min(200, json.Length))}...");

                    var doc = JsonDocument.Parse(json);
                    if (doc.RootElement.TryGetProperty("embeddings", out var embeddings) &&
                        embeddings.GetArrayLength() > 0)
                    {
                        var firstEmbedding = embeddings[0];
                        var result = new float[firstEmbedding.GetArrayLength()];
                        for (int i = 0; i < result.Length; i++)
                            result[i] = firstEmbedding[i].GetSingle();
                        return result;
                    }
                }

                // Fallback на старый формат (/api/embeddings)
                var reqOld = new { model, prompt = text };
                var contentOld = new StringContent(JsonSerializer.Serialize(reqOld), Encoding.UTF8, "application/json");
                var respOld = await _http.PostAsync($"{_url}/api/embeddings", contentOld);

                if (respOld.IsSuccessStatusCode)
                {
                    var json = await respOld.Content.ReadAsStringAsync();
                    Console.WriteLine($"   [DEBUG /api/embeddings] ответ: {json.Substring(0, Math.Min(200, json.Length))}...");

                    var doc = JsonDocument.Parse(json);
                    if (doc.RootElement.TryGetProperty("embedding", out var embedding))
                    {
                        var result = new float[embedding.GetArrayLength()];
                        for (int i = 0; i < result.Length; i++)
                            result[i] = embedding[i].GetSingle();
                        return result;
                    }
                }
                else
                {
                    Console.WriteLine($"   ⚠️ Ошибка эмбеддинга: {(int)respOld.StatusCode} {respOld.ReasonPhrase}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ❌ Исключение при получении эмбеддинга: {ex.Message}");
            }
            return Array.Empty<float>();
        }

        public async Task<string> GenerateAsync(string prompt, string model)
        {
            try
            {
                var req = new { model, prompt, stream = false };
                var content = new StringContent(JsonSerializer.Serialize(req), Encoding.UTF8, "application/json");
                var resp = await _http.PostAsync($"{_url}/api/generate", content);
                var json = await resp.Content.ReadAsStringAsync();

                Console.WriteLine($"   [DEBUG /api/generate] статус: {(int)resp.StatusCode}");
                Console.WriteLine($"   [DEBUG /api/generate] ответ (первые 500 символов): {json.Substring(0, Math.Min(500, json.Length))}");

                if (!resp.IsSuccessStatusCode)
                {
                    return $"Ошибка HTTP {(int)resp.StatusCode}: {resp.ReasonPhrase}";
                }

                var doc = JsonDocument.Parse(json);
                if (doc.RootElement.TryGetProperty("response", out var response))
                {
                    return response.GetString() ?? "Пустой ответ";
                }

                // Если нет поля "response", возвращаем весь JSON
                return json;
            }
            catch (Exception ex)
            {
                return $"❌ Исключение при генерации: {ex.Message}";
            }
        }
    }
}