package helps

import (
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

const (
	codexPromptCacheTTL         = 1 * time.Hour
	codexRollingCacheStepTokens = int64(16_000)
)

type CodexCache struct {
	ID                   string
	Expire               time.Time
	Generation           int
	LastRollPromptTokens int64
	LastRollCachedTokens int64
}

// codexCacheMap stores prompt cache IDs keyed by model+user_id.
// Protected by codexCacheMu. Entries expire after 1 hour.
var (
	codexCacheMap = make(map[string]CodexCache)
	codexCacheMu  sync.RWMutex
)

// codexCacheCleanupInterval controls how often expired entries are purged.
const codexCacheCleanupInterval = 15 * time.Minute

// codexCacheCleanupOnce ensures the background cleanup goroutine starts only once.
var codexCacheCleanupOnce sync.Once

// startCodexCacheCleanup launches a background goroutine that periodically
// removes expired entries from codexCacheMap to prevent memory leaks.
func startCodexCacheCleanup() {
	go func() {
		ticker := time.NewTicker(codexCacheCleanupInterval)
		defer ticker.Stop()
		for range ticker.C {
			purgeExpiredCodexCache()
		}
	}()
}

// purgeExpiredCodexCache removes entries that have expired.
func purgeExpiredCodexCache() {
	now := time.Now()
	codexCacheMu.Lock()
	defer codexCacheMu.Unlock()
	for key, cache := range codexCacheMap {
		if cache.Expire.Before(now) {
			delete(codexCacheMap, key)
		}
	}
}

// GetCodexCache retrieves a cached entry, returning ok=false if not found or expired.
func GetCodexCache(key string) (CodexCache, bool) {
	codexCacheCleanupOnce.Do(startCodexCacheCleanup)
	codexCacheMu.RLock()
	cache, ok := codexCacheMap[key]
	codexCacheMu.RUnlock()
	if !ok || cache.Expire.Before(time.Now()) {
		return CodexCache{}, false
	}
	return cache, true
}

// SetCodexCache stores a cache entry.
func SetCodexCache(key string, cache CodexCache) {
	codexCacheCleanupOnce.Do(startCodexCacheCleanup)
	codexCacheMu.Lock()
	codexCacheMap[key] = cache
	codexCacheMu.Unlock()
}

// GetOrCreateCodexRollingCache returns a stable prompt cache key for a scope.
func GetOrCreateCodexRollingCache(scope string) CodexCache {
	scope = strings.TrimSpace(scope)
	if scope == "" {
		return CodexCache{}
	}
	codexCacheCleanupOnce.Do(startCodexCacheCleanup)
	now := time.Now()
	codexCacheMu.Lock()
	defer codexCacheMu.Unlock()

	cache, ok := codexCacheMap[scope]
	if !ok || cache.Expire.Before(now) || strings.TrimSpace(cache.ID) == "" {
		cache = newCodexRollingCache(scope, 0, 0, now)
	} else {
		cache.Expire = now.Add(codexPromptCacheTTL)
	}
	codexCacheMap[scope] = cache
	return cache
}

// ObserveCodexRollingCacheUsage advances a scope after a larger cached prefix is usable.
func ObserveCodexRollingCacheUsage(scope string, inputTokens, cachedTokens int64) {
	scope = strings.TrimSpace(scope)
	if scope == "" || cachedTokens <= 0 {
		return
	}
	promptTokens := inputTokens + cachedTokens
	if promptTokens <= 0 {
		return
	}

	codexCacheCleanupOnce.Do(startCodexCacheCleanup)
	now := time.Now()
	codexCacheMu.Lock()
	defer codexCacheMu.Unlock()

	cache, ok := codexCacheMap[scope]
	if !ok || cache.Expire.Before(now) || strings.TrimSpace(cache.ID) == "" {
		cache = newCodexRollingCache(scope, 0, 0, now)
	}
	if cache.LastRollCachedTokens == 0 {
		cache.LastRollCachedTokens = cachedTokens
		cache.LastRollPromptTokens = promptTokens
		cache.Expire = now.Add(codexPromptCacheTTL)
		codexCacheMap[scope] = cache
		return
	}
	if cachedTokens-cache.LastRollCachedTokens < codexRollingCacheStepTokens {
		cache.Expire = now.Add(codexPromptCacheTTL)
		codexCacheMap[scope] = cache
		return
	}

	cache.Generation++
	cache.ID = codexRollingCacheID(scope, cache.Generation)
	cache.LastRollPromptTokens = promptTokens
	cache.LastRollCachedTokens = cachedTokens
	cache.Expire = now.Add(codexPromptCacheTTL)
	codexCacheMap[scope] = cache
}

func newCodexRollingCache(scope string, generation int, promptTokens int64, now time.Time) CodexCache {
	return CodexCache{
		ID:                   codexRollingCacheID(scope, generation),
		Expire:               now.Add(codexPromptCacheTTL),
		Generation:           generation,
		LastRollPromptTokens: promptTokens,
		LastRollCachedTokens: 0,
	}
}

func codexRollingCacheID(scope string, generation int) string {
	return uuid.NewSHA1(uuid.NameSpaceOID, []byte("cli-proxy-api:codex:prompt-cache:"+scope+":generation:"+strconv.Itoa(generation))).String()
}
