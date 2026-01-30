/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package router

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
)

// APIKeyInfo contains metadata about an API key
type APIKeyInfo struct {
	Hash           string   `json:"hash"`
	Owner          string   `json:"owner,omitempty"`
	Namespaces     []string `json:"namespaces,omitempty"`
	MaxConnections int      `json:"maxConnections,omitempty"`
	CreatedAt      string   `json:"createdAt,omitempty"`
}

// GPUResources specifies GPU resource requirements for external connections
type GPUResources struct {
	TflopsRequest string `json:"tflopsRequest,omitempty"` // e.g., "1"
	TflopsLimit   string `json:"tflopsLimit,omitempty"`   // e.g., "10"
	VramRequest   string `json:"vramRequest,omitempty"`   // e.g., "1Gi"
	VramLimit     string `json:"vramLimit,omitempty"`     // e.g., "8Gi"
}

// CreateExternalConnectionRequest is the request body for creating an external connection
type CreateExternalConnectionRequest struct {
	// Option 1: Use existing workload
	WorkloadName string `json:"workloadName,omitempty"`

	// Option 2: Create workload automatically with these settings
	PoolName  string        `json:"poolName,omitempty"`  // Required if workloadName not provided
	Resources *GPUResources `json:"resources,omitempty"` // GPU resource requirements

	// Common fields
	Namespace  string `json:"namespace" binding:"required"`
	ClientID   string `json:"clientId,omitempty"`
	TTLSeconds int64  `json:"ttlSeconds" binding:"required,min=60,max=86400"`
}

// ExternalConnectionResponse is the response for external connection operations
type ExternalConnectionResponse struct {
	Name          string  `json:"name"`
	Namespace     string  `json:"namespace"`
	Status        string  `json:"status"`
	ConnectionURL string  `json:"connectionURL,omitempty"`
	WorkerName    string  `json:"workerName,omitempty"`
	ExpiresAt     *string `json:"expiresAt,omitempty"`
	ClientID      string  `json:"clientId,omitempty"`
}

// ExternalConnectionRouter handles external connection API requests
type ExternalConnectionRouter struct {
	client          client.Client
	apiKeysCache    map[string]*APIKeyInfo
	apiKeysCacheMu  sync.RWMutex
	cacheExpiration time.Time
	cacheTTL        time.Duration
}

// NewExternalConnectionRouter creates a new ExternalConnectionRouter
func NewExternalConnectionRouter(c client.Client) *ExternalConnectionRouter {
	return &ExternalConnectionRouter{
		client:       c,
		apiKeysCache: make(map[string]*APIKeyInfo),
		cacheTTL:     5 * time.Minute,
	}
}

// Create creates a new external connection
func (r *ExternalConnectionRouter) Create(ctx *gin.Context) {
	logger := log.FromContext(ctx.Request.Context())

	// Validate API key
	keyID, ok := r.validateAPIKey(ctx)
	if !ok {
		ctx.JSON(http.StatusUnauthorized, gin.H{"error": "invalid or missing API key"})
		return
	}

	var req CreateExternalConnectionRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid request body: %v", err)})
		return
	}

	// Check namespace access
	if !r.canAccessNamespace(keyID, req.Namespace) {
		ctx.JSON(http.StatusForbidden, gin.H{"error": "API key does not have access to this namespace"})
		return
	}

	// Generate connection name early so we can use it as a stable owner.
	connName := fmt.Sprintf("ext-%s", uuid.New().String()[:8])

	var (
		workloadName     string
		autoCreateWorkload bool
		pool             *tfv1.GPUPool
	)

	if req.WorkloadName != "" {
		// Option 1: Use existing workload
		workload := &tfv1.TensorFusionWorkload{}
		if err := r.client.Get(ctx.Request.Context(), types.NamespacedName{
			Name:      req.WorkloadName,
			Namespace: req.Namespace,
		}, workload); err != nil {
			if errors.IsNotFound(err) {
				ctx.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("workload %s not found in namespace %s", req.WorkloadName, req.Namespace)})
				return
			}
			logger.Error(err, "Failed to get workload", "workload", req.WorkloadName)
			ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to verify workload"})
			return
		}
		workloadName = req.WorkloadName
	} else {
		// Option 2: Auto-create workload
		if req.PoolName == "" {
			ctx.JSON(http.StatusBadRequest, gin.H{"error": "either workloadName or poolName must be provided"})
			return
		}

		// Verify pool exists
		pool = &tfv1.GPUPool{}
		if err := r.client.Get(ctx.Request.Context(), types.NamespacedName{Name: req.PoolName}, pool); err != nil {
			if errors.IsNotFound(err) {
				ctx.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("GPU pool %s not found", req.PoolName)})
				return
			}
			logger.Error(err, "Failed to get GPU pool", "pool", req.PoolName)
			ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to verify GPU pool"})
			return
		}

		// Create a unique workload name; we'll create it after the connection is created
		// so we can set an ownerReference and allow GC to clean up on connection deletion.
		workloadName = fmt.Sprintf("ext-wl-%s", uuid.New().String()[:8])
		autoCreateWorkload = true
	}

	// Calculate expiration time
	expiresAt := metav1.NewTime(time.Now().Add(time.Duration(req.TTLSeconds) * time.Second))

	// Create the connection
	connection := &tfv1.TensorFusionConnection{
		ObjectMeta: metav1.ObjectMeta{
			Name:      connName,
			Namespace: req.Namespace,
			Labels: map[string]string{
				constants.WorkloadKey:                workloadName,
				constants.ExternalConnectionLabelKey: constants.TrueStringValue,
			},
		},
		Spec: tfv1.TensorFusionConnectionSpec{
			WorkloadName: workloadName,
			ExternalClient: &tfv1.ExternalClientSpec{
				ClientID:   req.ClientID,
				TTLSeconds: req.TTLSeconds,
			},
		},
	}

	if req.ClientID != "" {
		connection.Labels[constants.ExternalConnectionClientIDKey] = req.ClientID
	}

	if err := r.client.Create(ctx.Request.Context(), connection); err != nil {
		logger.Error(err, "Failed to create external connection", "name", connName)
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to create connection"})
		return
	}

	// Auto-created workload should be garbage-collected when the connection is deleted (TTL/DELETE).
	if autoCreateWorkload {
		if err := r.createExternalWorkload(ctx.Request.Context(), req, pool, workloadName, connection); err != nil {
			logger.Error(err, "Failed to create workload for external connection", "connection", connection.Name, "workload", workloadName)
			// Best-effort cleanup: delete the connection to avoid leaks.
			_ = r.client.Delete(ctx.Request.Context(), connection)
			ctx.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to create workload: %v", err)})
			return
		}
	}

	// Set the expiration time in status
	connection.Status.Phase = tfv1.WorkerPending
	connection.Status.ExpiresAt = &expiresAt
	if err := r.client.Status().Update(ctx.Request.Context(), connection); err != nil {
		logger.Error(err, "Failed to update connection status", "name", connName)
		// Don't fail the request, connection is created
	}

	expiresAtStr := expiresAt.Format(time.RFC3339)
	ctx.JSON(http.StatusCreated, ExternalConnectionResponse{
		Name:      connName,
		Namespace: req.Namespace,
		Status:    string(tfv1.WorkerPending),
		ExpiresAt: &expiresAtStr,
		ClientID:  req.ClientID,
	})
}

// Get returns the status of an external connection
func (r *ExternalConnectionRouter) Get(ctx *gin.Context) {
	logger := log.FromContext(ctx.Request.Context())

	// Validate API key
	keyID, ok := r.validateAPIKey(ctx)
	if !ok {
		ctx.JSON(http.StatusUnauthorized, gin.H{"error": "invalid or missing API key"})
		return
	}

	namespace := ctx.Param("namespace")
	name := ctx.Param("name")

	// Check namespace access
	if !r.canAccessNamespace(keyID, namespace) {
		ctx.JSON(http.StatusForbidden, gin.H{"error": "API key does not have access to this namespace"})
		return
	}

	connection := &tfv1.TensorFusionConnection{}
	if err := r.client.Get(ctx.Request.Context(), types.NamespacedName{
		Name:      name,
		Namespace: namespace,
	}, connection); err != nil {
		if errors.IsNotFound(err) {
			ctx.JSON(http.StatusNotFound, gin.H{"error": "connection not found"})
			return
		}
		logger.Error(err, "Failed to get connection", "name", name, "namespace", namespace)
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get connection"})
		return
	}

	// Verify this is an external connection
	if connection.Spec.ExternalClient == nil {
		ctx.JSON(http.StatusForbidden, gin.H{"error": "not an external connection"})
		return
	}

	response := ExternalConnectionResponse{
		Name:          connection.Name,
		Namespace:     connection.Namespace,
		Status:        string(connection.Status.Phase),
		ConnectionURL: connection.Status.ConnectionURL,
		WorkerName:    connection.Status.WorkerName,
		ClientID:      connection.Spec.ExternalClient.ClientID,
	}

	if connection.Status.ExpiresAt != nil {
		expiresAtStr := connection.Status.ExpiresAt.Format(time.RFC3339)
		response.ExpiresAt = &expiresAtStr
	}

	ctx.JSON(http.StatusOK, response)
}

// Delete terminates an external connection
func (r *ExternalConnectionRouter) Delete(ctx *gin.Context) {
	logger := log.FromContext(ctx.Request.Context())

	// Validate API key
	keyID, ok := r.validateAPIKey(ctx)
	if !ok {
		ctx.JSON(http.StatusUnauthorized, gin.H{"error": "invalid or missing API key"})
		return
	}

	namespace := ctx.Param("namespace")
	name := ctx.Param("name")

	// Check namespace access
	if !r.canAccessNamespace(keyID, namespace) {
		ctx.JSON(http.StatusForbidden, gin.H{"error": "API key does not have access to this namespace"})
		return
	}

	connection := &tfv1.TensorFusionConnection{}
	if err := r.client.Get(ctx.Request.Context(), types.NamespacedName{
		Name:      name,
		Namespace: namespace,
	}, connection); err != nil {
		if errors.IsNotFound(err) {
			ctx.JSON(http.StatusNotFound, gin.H{"error": "connection not found"})
			return
		}
		logger.Error(err, "Failed to get connection", "name", name, "namespace", namespace)
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get connection"})
		return
	}

	// Verify this is an external connection
	if connection.Spec.ExternalClient == nil {
		ctx.JSON(http.StatusForbidden, gin.H{"error": "not an external connection"})
		return
	}

	if err := r.client.Delete(ctx.Request.Context(), connection); err != nil {
		logger.Error(err, "Failed to delete connection", "name", name, "namespace", namespace)
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to delete connection"})
		return
	}

	ctx.Status(http.StatusNoContent)
}

// Refresh extends the TTL of an external connection
func (r *ExternalConnectionRouter) Refresh(ctx *gin.Context) {
	logger := log.FromContext(ctx.Request.Context())

	// Validate API key
	keyID, ok := r.validateAPIKey(ctx)
	if !ok {
		ctx.JSON(http.StatusUnauthorized, gin.H{"error": "invalid or missing API key"})
		return
	}

	namespace := ctx.Param("namespace")
	name := ctx.Param("name")

	// Check namespace access
	if !r.canAccessNamespace(keyID, namespace) {
		ctx.JSON(http.StatusForbidden, gin.H{"error": "API key does not have access to this namespace"})
		return
	}

	connection := &tfv1.TensorFusionConnection{}
	if err := r.client.Get(ctx.Request.Context(), types.NamespacedName{
		Name:      name,
		Namespace: namespace,
	}, connection); err != nil {
		if errors.IsNotFound(err) {
			ctx.JSON(http.StatusNotFound, gin.H{"error": "connection not found"})
			return
		}
		logger.Error(err, "Failed to get connection", "name", name, "namespace", namespace)
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get connection"})
		return
	}

	// Verify this is an external connection
	if connection.Spec.ExternalClient == nil {
		ctx.JSON(http.StatusForbidden, gin.H{"error": "not an external connection"})
		return
	}

	// Extend the TTL
	newExpiresAt := metav1.NewTime(time.Now().Add(time.Duration(connection.Spec.ExternalClient.TTLSeconds) * time.Second))
	connection.Status.ExpiresAt = &newExpiresAt

	if err := r.client.Status().Update(ctx.Request.Context(), connection); err != nil {
		logger.Error(err, "Failed to update connection status", "name", name, "namespace", namespace)
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to refresh TTL"})
		return
	}

	expiresAtStr := newExpiresAt.Format(time.RFC3339)
	ctx.JSON(http.StatusOK, gin.H{
		"name":      connection.Name,
		"namespace": connection.Namespace,
		"expiresAt": expiresAtStr,
	})
}

// List returns all external connections for the authenticated key
func (r *ExternalConnectionRouter) List(ctx *gin.Context) {
	logger := log.FromContext(ctx.Request.Context())

	// Validate API key
	keyID, ok := r.validateAPIKey(ctx)
	if !ok {
		ctx.JSON(http.StatusUnauthorized, gin.H{"error": "invalid or missing API key"})
		return
	}

	namespace := ctx.Query("namespace")
	if namespace != "" && !r.canAccessNamespace(keyID, namespace) {
		ctx.JSON(http.StatusForbidden, gin.H{"error": "API key does not have access to this namespace"})
		return
	}

	listOpts := []client.ListOption{
		client.MatchingLabels{
			constants.ExternalConnectionLabelKey: constants.TrueStringValue,
		},
	}
	if namespace != "" {
		listOpts = append(listOpts, client.InNamespace(namespace))
	}

	connectionList := &tfv1.TensorFusionConnectionList{}
	if err := r.client.List(ctx.Request.Context(), connectionList, listOpts...); err != nil {
		logger.Error(err, "Failed to list connections")
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to list connections"})
		return
	}

	var responses []ExternalConnectionResponse
	for _, conn := range connectionList.Items {
		// Filter by namespace access
		if !r.canAccessNamespace(keyID, conn.Namespace) {
			continue
		}

		response := ExternalConnectionResponse{
			Name:          conn.Name,
			Namespace:     conn.Namespace,
			Status:        string(conn.Status.Phase),
			ConnectionURL: conn.Status.ConnectionURL,
			WorkerName:    conn.Status.WorkerName,
		}

		if conn.Spec.ExternalClient != nil {
			response.ClientID = conn.Spec.ExternalClient.ClientID
		}

		if conn.Status.ExpiresAt != nil {
			expiresAtStr := conn.Status.ExpiresAt.Format(time.RFC3339)
			response.ExpiresAt = &expiresAtStr
		}

		responses = append(responses, response)
	}

	ctx.JSON(http.StatusOK, gin.H{"connections": responses})
}

// createExternalWorkload creates a TensorFusionWorkload for an external connection.
// The workload is owned by the connection so that GC cleans it up when the connection is deleted.
func (r *ExternalConnectionRouter) createExternalWorkload(
	ctx context.Context,
	req CreateExternalConnectionRequest,
	pool *tfv1.GPUPool,
	workloadName string,
	connection *tfv1.TensorFusionConnection,
) error {

	// Set default resources if not provided
	tflopsReq := "0"
	tflopsLim := "0"
	vramReq := "4Gi"
	vramLim := "4Gi"

	if req.Resources != nil {
		if req.Resources.TflopsRequest != "" {
			tflopsReq = req.Resources.TflopsRequest
		}
		if req.Resources.TflopsLimit != "" {
			tflopsLim = req.Resources.TflopsLimit
		}
		if req.Resources.VramRequest != "" {
			vramReq = req.Resources.VramRequest
		}
		if req.Resources.VramLimit != "" {
			vramLim = req.Resources.VramLimit
		}
	}

	// Determine GPU vendor from pool spec
	gpuVendor := "AMD"
	if pool.Spec.NodeManagerConfig != nil && pool.Spec.NodeManagerConfig.DefaultVendor != "" {
		gpuVendor = pool.Spec.NodeManagerConfig.DefaultVendor
	}

	workload := &tfv1.TensorFusionWorkload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      workloadName,
			Namespace: req.Namespace,
			Labels: map[string]string{
				constants.GpuPoolKey:                 req.PoolName,
				constants.ExternalConnectionLabelKey: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				constants.WorkloadModeAnnotation: constants.WorkloadModeDynamic,
			},
		},
		Spec: tfv1.WorkloadProfileSpec{
			PoolName:  req.PoolName,
			Replicas:  nil, // Dynamic replicas for dedicated workers
			GPUCount:  1,
			Isolation: tfv1.IsolationModeShared,
			Qos:       tfv1.QoSHigh,
			GPUVendor: gpuVendor,
			Resources: tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse(tflopsReq),
					Vram:   resource.MustParse(vramReq),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse(tflopsLim),
					Vram:   resource.MustParse(vramLim),
				},
			},
		},
	}

	// Set owner reference to allow GC cleanup with connection deletion.
	controller := true
	blockOwnerDeletion := true
	workload.OwnerReferences = []metav1.OwnerReference{
		{
			APIVersion:         tfv1.GroupVersion.String(),
			Kind:               "TensorFusionConnection",
			Name:               connection.Name,
			UID:                connection.UID,
			Controller:         &controller,
			BlockOwnerDeletion: &blockOwnerDeletion,
		},
	}

	if err := r.client.Create(ctx, workload); err != nil {
		return fmt.Errorf("create workload: %w", err)
	}

	log.FromContext(ctx).Info("Created external workload", "name", workloadName, "pool", req.PoolName, "vendor", gpuVendor, "ownerConnection", connection.Name)
	return nil
}

// validateAPIKey validates the API key from the request header
// Returns the key ID if valid, or empty string and false if invalid
func (r *ExternalConnectionRouter) validateAPIKey(ctx *gin.Context) (string, bool) {
	apiKey := ctx.GetHeader(constants.ExternalConnectionAPIKeyHeader)
	if apiKey == "" {
		return "", false
	}

	// Refresh cache if needed
	if err := r.refreshAPIKeysCache(ctx.Request.Context()); err != nil {
		log.FromContext(ctx.Request.Context()).Error(err, "Failed to refresh API keys cache")
		return "", false
	}

	r.apiKeysCacheMu.RLock()
	defer r.apiKeysCacheMu.RUnlock()

	for keyID, keyInfo := range r.apiKeysCache {
		if err := bcrypt.CompareHashAndPassword([]byte(keyInfo.Hash), []byte(apiKey)); err == nil {
			return keyID, true
		}
	}

	return "", false
}

// canAccessNamespace checks if the API key can access the given namespace
func (r *ExternalConnectionRouter) canAccessNamespace(keyID, namespace string) bool {
	r.apiKeysCacheMu.RLock()
	defer r.apiKeysCacheMu.RUnlock()

	keyInfo, ok := r.apiKeysCache[keyID]
	if !ok {
		return false
	}

	// If no namespaces specified, allow all
	if len(keyInfo.Namespaces) == 0 {
		return true
	}

	for _, ns := range keyInfo.Namespaces {
		if ns == namespace || ns == "*" {
			return true
		}
	}

	return false
}

// refreshAPIKeysCache refreshes the API keys cache from the Kubernetes secret
func (r *ExternalConnectionRouter) refreshAPIKeysCache(ctx context.Context) error {
	r.apiKeysCacheMu.RLock()
	if time.Now().Before(r.cacheExpiration) {
		r.apiKeysCacheMu.RUnlock()
		return nil
	}
	r.apiKeysCacheMu.RUnlock()

	r.apiKeysCacheMu.Lock()
	defer r.apiKeysCacheMu.Unlock()

	// Double-check after acquiring write lock
	if time.Now().Before(r.cacheExpiration) {
		return nil
	}

	secret := &corev1.Secret{}
	if err := r.client.Get(ctx, types.NamespacedName{
		Name:      constants.ExternalConnectionSecretName,
		Namespace: constants.ExternalConnectionSecretNamespace,
	}, secret); err != nil {
		if errors.IsNotFound(err) {
			// No API keys configured, clear cache
			r.apiKeysCache = make(map[string]*APIKeyInfo)
			r.cacheExpiration = time.Now().Add(r.cacheTTL)
			return nil
		}
		return fmt.Errorf("get API keys secret: %w", err)
	}

	keysData, ok := secret.Data[constants.ExternalConnectionSecretDataKey]
	if !ok {
		r.apiKeysCache = make(map[string]*APIKeyInfo)
		r.cacheExpiration = time.Now().Add(r.cacheTTL)
		return nil
	}

	var keys map[string]*APIKeyInfo
	if err := json.Unmarshal(keysData, &keys); err != nil {
		return fmt.Errorf("unmarshal API keys: %w", err)
	}

	r.apiKeysCache = keys
	r.cacheExpiration = time.Now().Add(r.cacheTTL)
	return nil
}
