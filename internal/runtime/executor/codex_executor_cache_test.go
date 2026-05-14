package executor

import (
	"context"
	"io"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

func TestCodexExecutorCacheHelper_OpenAIChatCompletions_StablePromptCacheKeyFromAPIKey(t *testing.T) {
	recorder := httptest.NewRecorder()
	ginCtx, _ := gin.CreateTestContext(recorder)
	ginCtx.Set("userApiKey", "test-api-key")

	ctx := context.WithValue(context.Background(), "gin", ginCtx)
	executor := &CodexExecutor{}
	auth := &cliproxyauth.Auth{ID: "auth-a", Provider: "codex", ProxyURL: "socks5://proxy-a.example:443"}
	rawJSON := []byte(`{"model":"gpt-5.3-codex","stream":true}`)
	req := cliproxyexecutor.Request{
		Model:   "gpt-5.3-codex",
		Payload: []byte(`{"model":"gpt-5.3-codex"}`),
	}
	url := "https://example.com/responses"

	httpReq, err := executor.cacheHelper(ctx, auth, sdktranslator.FromString("openai"), url, req, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper error: %v", err)
	}

	body, errRead := io.ReadAll(httpReq.Body)
	if errRead != nil {
		t.Fatalf("read request body: %v", errRead)
	}

	expectedKey := codexNamespacedPromptCacheKey(codexAuthIsolationKey(auth, url), "test-api-key")
	gotKey := gjson.GetBytes(body, "prompt_cache_key").String()
	if gotKey != expectedKey {
		t.Fatalf("prompt_cache_key = %q, want %q", gotKey, expectedKey)
	}
	if gotConversation := httpReq.Header.Get("Conversation_id"); gotConversation != "" {
		t.Fatalf("Conversation_id = %q, want empty", gotConversation)
	}
	if gotSession := httpReq.Header.Get("Session_id"); gotSession != expectedKey {
		t.Fatalf("Session_id = %q, want %q", gotSession, expectedKey)
	}

	httpReq2, err := executor.cacheHelper(ctx, auth, sdktranslator.FromString("openai"), url, req, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper error (second call): %v", err)
	}
	body2, errRead2 := io.ReadAll(httpReq2.Body)
	if errRead2 != nil {
		t.Fatalf("read request body (second call): %v", errRead2)
	}
	gotKey2 := gjson.GetBytes(body2, "prompt_cache_key").String()
	if gotKey2 != expectedKey {
		t.Fatalf("prompt_cache_key (second call) = %q, want %q", gotKey2, expectedKey)
	}
}

func TestCodexExecutorCacheHelper_OpenAIChatCompletions_UsesMetadataUserIDWhenAvailable(t *testing.T) {
	recorder := httptest.NewRecorder()
	ginCtx, _ := gin.CreateTestContext(recorder)
	ginCtx.Set("userApiKey", "worker-api-key")

	ctx := context.WithValue(context.Background(), "gin", ginCtx)
	executor := &CodexExecutor{}
	auth := &cliproxyauth.Auth{ID: "auth-openai-meta", Provider: "codex", ProxyURL: "socks5://proxy.example:443"}
	rawJSON := []byte(`{"model":"gpt-5.4","stream":true}`)
	req := cliproxyexecutor.Request{
		Model:   "gpt-5.4(high)",
		Payload: []byte(`{"model":"gpt-5.4","metadata":{"user_id":"forwarded-session-user"}}`),
	}
	url := "https://example.com/responses"

	httpReq0, err := executor.cacheHelper(ctx, auth, sdktranslator.FromString("openai"), url, req, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper first error: %v", err)
	}
	body0, err := io.ReadAll(httpReq0.Body)
	if err != nil {
		t.Fatalf("read first body: %v", err)
	}
	key0 := gjson.GetBytes(body0, "prompt_cache_key").String()
	apiKeyFallback := codexNamespacedPromptCacheKey(codexAuthIsolationKey(auth, url), "worker-api-key")
	if key0 == "" {
		t.Fatal("expected metadata-backed prompt cache key")
	}
	if key0 == apiKeyFallback {
		t.Fatalf("metadata.user_id should override worker API-key fallback cache key %q", apiKeyFallback)
	}

	observeCodexPromptCacheUsage(auth, sdktranslator.FromString("openai"), req.Model, req.Payload, 24000, 18432)
	httpReq1, err := executor.cacheHelper(ctx, auth, sdktranslator.FromString("openai"), url, req, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper rolled error: %v", err)
	}
	body1, err := io.ReadAll(httpReq1.Body)
	if err != nil {
		t.Fatalf("read rolled body: %v", err)
	}
	key1 := gjson.GetBytes(body1, "prompt_cache_key").String()
	if key1 == "" || key1 == key0 {
		t.Fatalf("expected metadata-backed OpenAI cache key to roll, key0=%q key1=%q", key0, key1)
	}
}

func TestCodexExecutorCacheHelper_OpenAIChatCompletions_UsesForwardedPromptCacheKey(t *testing.T) {
	recorder := httptest.NewRecorder()
	ginCtx, _ := gin.CreateTestContext(recorder)
	ginCtx.Set("userApiKey", "worker-api-key")

	ctx := context.WithValue(context.Background(), "gin", ginCtx)
	executor := &CodexExecutor{}
	auth := &cliproxyauth.Auth{ID: "auth-openai-forwarded-cache", Provider: "codex", ProxyURL: "socks5://proxy.example:443"}
	rawJSON := []byte(`{"model":"gpt-5.4","stream":true}`)
	reqA := cliproxyexecutor.Request{
		Model:   "gpt-5.4(high)",
		Payload: []byte(`{"model":"gpt-5.4","prompt_cache_key":"session-cache-a","metadata":{"user_id":"same-user"}}`),
	}
	reqB := cliproxyexecutor.Request{
		Model:   "gpt-5.4(high)",
		Payload: []byte(`{"model":"gpt-5.4","prompt_cache_key":"session-cache-b","metadata":{"user_id":"same-user"}}`),
	}
	url := "https://example.com/responses"

	httpReqA, err := executor.cacheHelper(ctx, auth, sdktranslator.FromString("openai"), url, reqA, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper A error: %v", err)
	}
	httpReqB, err := executor.cacheHelper(ctx, auth, sdktranslator.FromString("openai"), url, reqB, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper B error: %v", err)
	}
	bodyA, err := io.ReadAll(httpReqA.Body)
	if err != nil {
		t.Fatalf("read body A: %v", err)
	}
	bodyB, err := io.ReadAll(httpReqB.Body)
	if err != nil {
		t.Fatalf("read body B: %v", err)
	}
	keyA := gjson.GetBytes(bodyA, "prompt_cache_key").String()
	keyB := gjson.GetBytes(bodyB, "prompt_cache_key").String()
	if keyA == "" || keyB == "" {
		t.Fatalf("expected forwarded prompt cache keys, got A=%q B=%q", keyA, keyB)
	}
	if keyA == keyB {
		t.Fatalf("different forwarded prompt cache keys shared generated key %q", keyA)
	}

	observeCodexPromptCacheUsage(auth, sdktranslator.FromString("openai"), reqA.Model, reqA.Payload, 36387, 18432)
	httpReqA2, err := executor.cacheHelper(ctx, auth, sdktranslator.FromString("openai"), url, reqA, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper A after cached growth error: %v", err)
	}
	bodyA2, err := io.ReadAll(httpReqA2.Body)
	if err != nil {
		t.Fatalf("read body A after cached growth: %v", err)
	}
	keyA2 := gjson.GetBytes(bodyA2, "prompt_cache_key").String()
	if keyA2 != keyA {
		t.Fatalf("forwarded prompt cache key should remain stable after cached growth: got %q want %q", keyA2, keyA)
	}
}

func TestCodexExecutorCacheHelper_ClaudePromptCacheSeparatedByAuth(t *testing.T) {
	executor := &CodexExecutor{}
	rawJSON := []byte(`{"model":"gpt-5.3-codex","stream":true}`)
	req := cliproxyexecutor.Request{
		Model:   "gpt-5.3-codex",
		Payload: []byte(`{"metadata":{"user_id":"same-user"}}`),
	}
	url := "https://example.com/responses"
	authA := &cliproxyauth.Auth{ID: "auth-a", Provider: "codex", ProxyURL: "socks5://proxy-a.example:443"}
	authB := &cliproxyauth.Auth{ID: "auth-b", Provider: "codex", ProxyURL: "socks5://proxy-b.example:443"}

	httpReqA, err := executor.cacheHelper(context.Background(), authA, sdktranslator.FromString("claude"), url, req, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper auth A error: %v", err)
	}
	httpReqB, err := executor.cacheHelper(context.Background(), authB, sdktranslator.FromString("claude"), url, req, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper auth B error: %v", err)
	}

	bodyA, err := io.ReadAll(httpReqA.Body)
	if err != nil {
		t.Fatalf("read auth A body: %v", err)
	}
	bodyB, err := io.ReadAll(httpReqB.Body)
	if err != nil {
		t.Fatalf("read auth B body: %v", err)
	}
	keyA := gjson.GetBytes(bodyA, "prompt_cache_key").String()
	keyB := gjson.GetBytes(bodyB, "prompt_cache_key").String()
	if keyA == "" || keyB == "" {
		t.Fatalf("prompt cache keys should be set, got A=%q B=%q", keyA, keyB)
	}
	if keyA == keyB {
		t.Fatalf("prompt cache key should differ by auth isolation key, got %q", keyA)
	}
}

func TestCodexExecutorCacheHelper_ClaudePromptCacheUsesBaseModel(t *testing.T) {
	executor := &CodexExecutor{}
	rawJSON := []byte(`{"model":"gpt-5.4","stream":true}`)
	reqHigh := cliproxyexecutor.Request{
		Model:   "gpt-5.4(high)",
		Payload: []byte(`{"metadata":{"user_id":"same-base-model-user"}}`),
	}
	reqMedium := cliproxyexecutor.Request{
		Model:   "gpt-5.4(medium)",
		Payload: []byte(`{"metadata":{"user_id":"same-base-model-user"}}`),
	}
	url := "https://example.com/responses"
	auth := &cliproxyauth.Auth{ID: "auth-base-model", Provider: "codex", ProxyURL: "socks5://proxy.example:443"}

	httpReqHigh, err := executor.cacheHelper(context.Background(), auth, sdktranslator.FromString("claude"), url, reqHigh, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper high error: %v", err)
	}
	httpReqMedium, err := executor.cacheHelper(context.Background(), auth, sdktranslator.FromString("claude"), url, reqMedium, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper medium error: %v", err)
	}

	bodyHigh, err := io.ReadAll(httpReqHigh.Body)
	if err != nil {
		t.Fatalf("read high body: %v", err)
	}
	bodyMedium, err := io.ReadAll(httpReqMedium.Body)
	if err != nil {
		t.Fatalf("read medium body: %v", err)
	}
	keyHigh := gjson.GetBytes(bodyHigh, "prompt_cache_key").String()
	keyMedium := gjson.GetBytes(bodyMedium, "prompt_cache_key").String()
	if keyHigh == "" || keyMedium == "" {
		t.Fatalf("prompt cache keys should be set, got high=%q medium=%q", keyHigh, keyMedium)
	}
	if keyHigh != keyMedium {
		t.Fatalf("same base model should share prompt cache key, high=%q medium=%q", keyHigh, keyMedium)
	}
}

func TestCodexExecutorCacheHelper_ClaudePromptCacheRollsAfterCachedGrowth(t *testing.T) {
	executor := &CodexExecutor{}
	rawJSON := []byte(`{"model":"gpt-5.4","stream":true}`)
	req := cliproxyexecutor.Request{
		Model:   "gpt-5.4(medium)",
		Payload: []byte(`{"metadata":{"user_id":"rolling-cache-user"}}`),
	}
	url := "https://example.com/responses"
	auth := &cliproxyauth.Auth{ID: "auth-rolling-cache", Provider: "codex", ProxyURL: "socks5://proxy.example:443"}

	httpReq0, err := executor.cacheHelper(context.Background(), auth, sdktranslator.FromString("claude"), url, req, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper first error: %v", err)
	}
	body0, err := io.ReadAll(httpReq0.Body)
	if err != nil {
		t.Fatalf("read first body: %v", err)
	}
	key0 := gjson.GetBytes(body0, "prompt_cache_key").String()
	if key0 == "" {
		t.Fatal("expected initial prompt cache key")
	}

	observeCodexPromptCacheUsage(auth, sdktranslator.FromString("claude"), req.Model, req.Payload, 33859, 0)
	httpReqNoCache, err := executor.cacheHelper(context.Background(), auth, sdktranslator.FromString("claude"), url, req, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper no-cache observation error: %v", err)
	}
	bodyNoCache, err := io.ReadAll(httpReqNoCache.Body)
	if err != nil {
		t.Fatalf("read no-cache observation body: %v", err)
	}
	if keyNoCache := gjson.GetBytes(bodyNoCache, "prompt_cache_key").String(); keyNoCache != key0 {
		t.Fatalf("cache key changed before cached tokens appeared: got %q want %q", keyNoCache, key0)
	}

	observeCodexPromptCacheUsage(auth, sdktranslator.FromString("claude"), req.Model, req.Payload, 15808, 18432)
	httpReq1, err := executor.cacheHelper(context.Background(), auth, sdktranslator.FromString("claude"), url, req, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper rolled error: %v", err)
	}
	body1, err := io.ReadAll(httpReq1.Body)
	if err != nil {
		t.Fatalf("read rolled body: %v", err)
	}
	key1 := gjson.GetBytes(body1, "prompt_cache_key").String()
	if key1 == "" || key1 == key0 {
		t.Fatalf("expected rolling cache key to advance, key0=%q key1=%q", key0, key1)
	}

	observeCodexPromptCacheUsage(auth, sdktranslator.FromString("claude"), req.Model, req.Payload, 20000, 18432)
	httpReqBelowStep, err := executor.cacheHelper(context.Background(), auth, sdktranslator.FromString("claude"), url, req, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper below-step error: %v", err)
	}
	bodyBelowStep, err := io.ReadAll(httpReqBelowStep.Body)
	if err != nil {
		t.Fatalf("read below-step body: %v", err)
	}
	if keyBelowStep := gjson.GetBytes(bodyBelowStep, "prompt_cache_key").String(); keyBelowStep != key1 {
		t.Fatalf("cache key changed below rolling step: got %q want %q", keyBelowStep, key1)
	}
}
